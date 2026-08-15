#include "searcher.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace cascade {
namespace {

constexpr uint32_t kNoDocid = UINT32_MAX;

// Bounded min-heap of the best k results seen so far. Its smallest score is the
// threshold every pruning decision is made against.
class TopK {
 public:
  explicit TopK(uint32_t k) : k_(k) {}

  float threshold() const {
    return heap_.size() < k_ ? 0.0f : heap_.top().score;
  }

  bool would_enter(float score) const {
    return heap_.size() < k_ || score >= heap_.top().score;
  }

  void push(uint32_t docid, float score) {
    const Result candidate{docid, score};
    if (heap_.size() < k_) {
      heap_.push(candidate);
      return;
    }
    // Compare(a, b) is a full strict order over (score, docid) — docids are
    // unique, so no two distinct documents ever compare equal under it. An
    // exact score tie at the k-th boundary is common in this corpus (BM25
    // over a small synthetic vocabulary), and checking raw score here as
    // the original code did left ties resolved by arrival order: whichever
    // algorithm happened to visit a tied document first kept it, so
    // exhaustive DAAT-OR (ascending docid order) and WAND/BlockMax-WAND
    // (pivot-jump order) could retain different — but equally
    // score-correct — top-k sets. Routing the replacement decision through
    // the same comparator the heap already orders by makes retention
    // depend only on (score, docid), not traversal order.
    if (Compare{}(candidate, heap_.top())) {
      heap_.pop();
      heap_.push(candidate);
    }
  }

  std::vector<Result> drain() {
    std::vector<Result> out;
    out.reserve(heap_.size());
    while (!heap_.empty()) {
      out.push_back(heap_.top());
      heap_.pop();
    }
    // Descending score; ties broken by ascending docid so runs are deterministic.
    std::sort(out.begin(), out.end(), [](const Result& a, const Result& b) {
      if (a.score != b.score) return a.score > b.score;
      return a.docid < b.docid;
    });
    return out;
  }

 private:
  struct Compare {
    bool operator()(const Result& a, const Result& b) const {
      if (a.score != b.score) return a.score > b.score;
      return a.docid < b.docid;
    }
  };
  uint32_t k_;
  std::priority_queue<Result, std::vector<Result>, Compare> heap_;
};

// Ties on docid are broken by pointer identity — equivalently, by each
// cursor's position in the fixed cursors vector built by open_cursors(),
// since that vector is never resized during a search. This isn't just for
// determinism: the scoring loops sum contributions over exactly the cursors
// at pivot_docid, in whatever order this puts them in, and float addition
// isn't associative. daat_or() always sums in that same fixed (term-sorted)
// order. Without a matching tie-break here, std::sort could leave same-docid
// cursors in a different relative order on a different call, so WAND/BMW's
// summed score for a document could come out a few ULPs off from
// exhaustive's for that identical document — enough, on documents whose true
// scores are already extremely close, to flip which one lands in the top-k.
void sort_by_docid(std::vector<PostingCursor*>* cursors) {
  std::sort(cursors->begin(), cursors->end(),
            [](const PostingCursor* a, const PostingCursor* b) {
              if (a->docid() != b->docid()) return a->docid() < b->docid();
              return a < b;
            });
}

}  // namespace

// ---------------------------------------------------------------- cursor ----

PostingCursor::PostingCursor(const BlockMeta* blocks, uint32_t num_blocks,
                             const uint8_t* payload, float idf, float max_impact,
                             SearchStats* stats)
    : blocks_(blocks),
      num_blocks_(num_blocks),
      payload_(payload),
      idf_(idf),
      max_impact_(max_impact),
      stats_(stats) {
  if (num_blocks_ == 0) {
    exhausted_ = true;
    docid_ = kNoDocid;
    return;
  }
  decode_block();
  docid_ = docids_[0];
  freq_ = freqs_[0];
}

void PostingCursor::decode_block() {
  const BlockMeta& meta = blocks_[block_index_];
  const uint8_t* data = payload_ + meta.payload_offset;
  size_t pos = 0;
  uint32_t previous = 0;
  for (uint32_t i = 0; i < meta.count; i++) {
    const uint32_t gap = get_varint(data, &pos);
    previous = (i == 0) ? gap : previous + gap;
    docids_[i] = previous;
  }
  for (uint32_t i = 0; i < meta.count; i++) {
    freqs_[i] = get_varint(data, &pos);
  }
  position_ = 0;
  stats_->blocks_decoded++;
}

void PostingCursor::next() {
  if (exhausted_) return;
  position_++;
  if (position_ >= blocks_[block_index_].count) {
    block_index_++;
    if (block_index_ >= num_blocks_) {
      exhausted_ = true;
      docid_ = kNoDocid;
      return;
    }
    decode_block();
  }
  docid_ = docids_[position_];
  freq_ = freqs_[position_];
}

void PostingCursor::skip_to_block(uint32_t target) {
  if (exhausted_) return;
  const uint32_t entry_block = block_index_;
  while (block_index_ < num_blocks_ && blocks_[block_index_].max_docid < target) {
    block_index_++;
  }
  if (block_index_ >= num_blocks_) {
    exhausted_ = true;
    docid_ = kNoDocid;
    return;
  }
  // Whole blocks between entry_block and the landed one are discarded on
  // max_docid alone, without their payload ever being decoded — that's the
  // skip list. But leaving docid()/freq() pointed at the old (pre-skip)
  // entry once we've moved on is what let callers observe a stale docid:
  // block_max_wand mixes skip_to_block with sort_by_docid, and a cursor
  // whose block advanced but whose cached docid_ didn't allowed the "live"
  // array to go out of the order that later code assumes. Decoding just the
  // one block we land on (not the ones skipped past) keeps docid() honest at
  // the cost of one bounded decode, only when the block actually changes.
  if (block_index_ != entry_block) {
    decode_block();
    docid_ = docids_[0];
    freq_ = freqs_[0];
  }
}

void PostingCursor::next_geq(uint32_t target) {
  if (exhausted_ || docid_ >= target) return;

  // Block-level skip first: skip_to_block leaves the landed block decoded,
  // so no separate re-decode is needed here.
  skip_to_block(target);
  if (exhausted_) return;

  const uint32_t count = blocks_[block_index_].count;
  while (position_ < count && docids_[position_] < target) position_++;
  if (position_ >= count) {
    // target sits in the gap after this block's last docid.
    block_index_++;
    if (block_index_ >= num_blocks_) {
      exhausted_ = true;
      docid_ = kNoDocid;
      return;
    }
    decode_block();
  }
  docid_ = docids_[position_];
  freq_ = freqs_[position_];
}

// ----------------------------------------------------------------- index ----

void* Index::map_file(const std::string& path, size_t* size) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  *size = static_cast<size_t>(st.st_size);
  void* addr = ::mmap(nullptr, *size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (addr == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
  mappings_.emplace_back(addr, *size);
  return addr;
}

Index::Index(const std::string& directory) {
  size_t size = 0;

  const auto* docs = static_cast<const uint8_t*>(map_file(directory + "/index.docs", &size));
  std::memcpy(&doc_count_, docs, 4);
  std::memcpy(&avg_dl_, docs + 4, 4);
  doc_lengths_ = reinterpret_cast<const uint32_t*>(docs + 8);

  const auto* terms = static_cast<const uint8_t*>(map_file(directory + "/index.terms", &size));
  std::memcpy(&term_count_, terms, 4);
  term_records_ = reinterpret_cast<const TermRecord*>(terms + 4);
  term_strings_ = reinterpret_cast<const char*>(terms + 4 + sizeof(TermRecord) * term_count_);

  postings_ = static_cast<const uint8_t*>(map_file(directory + "/index.post", &size));

  const auto* ids = static_cast<const uint8_t*>(map_file(directory + "/index.docids", &size));
  uint32_t id_count = 0;
  std::memcpy(&id_count, ids, 4);
  docid_offsets_ = reinterpret_cast<const uint64_t*>(ids + 4);
  docid_blob_ = reinterpret_cast<const char*>(ids + 4 + 8 * (static_cast<size_t>(id_count) + 1));
}

Index::~Index() {
  for (auto& [addr, size] : mappings_) ::munmap(addr, size);
}

std::string Index::external_id(uint32_t docid) const {
  const uint64_t start = docid_offsets_[docid];
  const uint64_t end = docid_offsets_[docid + 1];
  return std::string(docid_blob_ + start, end - start);
}

const TermRecord* Index::find_term(const std::string& term) const {
  uint32_t low = 0;
  uint32_t high = term_count_;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const TermRecord& record = term_records_[mid];
    const int cmp = term.compare(
        0, term.size(), term_strings_ + record.str_offset, record.str_len);
    if (cmp == 0) return &record;
    if (cmp < 0) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return nullptr;
}

const BlockMeta* Index::blocks_for(const TermRecord& record) const {
  return reinterpret_cast<const BlockMeta*>(postings_ + record.post_offset);
}

const uint8_t* Index::payload_for(const TermRecord& record) const {
  return postings_ + record.post_offset + sizeof(BlockMeta) * record.num_blocks;
}

std::vector<PostingCursor> Index::open_cursors(const std::string& query,
                                               SearchStats* stats) const {
  std::vector<std::string> terms = analyzer_.analyze(query);

  // Repeated query terms are collapsed; a term appearing twice would otherwise
  // be scored twice and silently double its own contribution.
  std::sort(terms.begin(), terms.end());
  terms.erase(std::unique(terms.begin(), terms.end()), terms.end());

  std::vector<PostingCursor> cursors;
  cursors.reserve(terms.size());
  for (const std::string& term : terms) {
    const TermRecord* record = find_term(term);
    if (record == nullptr) continue;
    cursors.emplace_back(blocks_for(*record), record->num_blocks, payload_for(*record),
                         bm25_idf(doc_count_, record->df), record->max_impact, stats);
  }
  return cursors;
}

std::vector<Result> Index::search(const std::string& query, uint32_t k,
                                  Algorithm algorithm, SearchStats* stats) const {
  SearchStats local;
  if (stats == nullptr) stats = &local;

  std::vector<PostingCursor> cursors = open_cursors(query, stats);
  if (cursors.empty()) return {};

  switch (algorithm) {
    case Algorithm::kDaatOr:
      return daat_or(cursors, k, stats);
    case Algorithm::kWand:
      return wand(cursors, k, stats);
    case Algorithm::kBlockMaxWand:
      return block_max_wand(cursors, k, stats);
  }
  return {};
}

// Exhaustive document-at-a-time disjunction. Correct by construction: every
// document containing any query term is fully scored. This is the oracle the
// other two are checked against.
std::vector<Result> Index::daat_or(std::vector<PostingCursor>& cursors, uint32_t k,
                                   SearchStats* stats) const {
  TopK top(k);
  while (true) {
    uint32_t current = kNoDocid;
    for (const PostingCursor& cursor : cursors) {
      if (!cursor.exhausted()) current = std::min(current, cursor.docid());
    }
    if (current == kNoDocid) break;

    float score = 0.0f;
    for (PostingCursor& cursor : cursors) {
      if (cursor.exhausted() || cursor.docid() != current) continue;
      score += bm25_term_score(cursor.idf(), cursor.freq(), doc_lengths_[current], avg_dl_,
                               params_);
      stats->postings_scored++;
      cursor.next();
    }
    stats->full_evaluations++;
    stats->candidates_seen++;
    top.push(current, score);
  }
  return top.drain();
}

// WAND (Broder et al. 2003). Cursors are kept sorted by docid; the pivot is the
// first document where the accumulated *term-wide* upper bounds could beat the
// heap threshold. Everything before the pivot is provably unable to make the
// top k, so it is skipped without being scored.
std::vector<Result> Index::wand(std::vector<PostingCursor>& cursors, uint32_t k,
                                SearchStats* stats) const {
  TopK top(k);
  std::vector<PostingCursor*> live;
  live.reserve(cursors.size());
  for (PostingCursor& cursor : cursors) live.push_back(&cursor);

  while (true) {
    live.erase(std::remove_if(live.begin(), live.end(),
                              [](const PostingCursor* c) { return c->exhausted(); }),
               live.end());
    if (live.empty()) break;
    sort_by_docid(&live);

    const float threshold = top.threshold();
    float bound = 0.0f;
    size_t pivot_index = live.size();
    for (size_t i = 0; i < live.size(); i++) {
      bound += live[i]->max_impact();
      // >=, not >: TopK::push() can let an exact score tie displace the
      // current worst when the new docid is smaller (see its comment), so a
      // bound that only ties threshold can still be worth evaluating.
      if (bound >= threshold) {
        pivot_index = i;
        break;
      }
    }
    // No suffix of the sorted cursors can reach the threshold: nothing left to find.
    if (pivot_index == live.size()) break;

    const uint32_t pivot_docid = live[pivot_index]->docid();
    stats->candidates_seen++;

    if (live[0]->docid() == pivot_docid) {
      float score = 0.0f;
      for (PostingCursor* cursor : live) {
        if (cursor->docid() != pivot_docid) break;
        score += bm25_term_score(cursor->idf(), cursor->freq(), doc_lengths_[pivot_docid],
                                 avg_dl_, params_);
        stats->postings_scored++;
      }
      stats->full_evaluations++;
      top.push(pivot_docid, score);
      for (PostingCursor* cursor : live) {
        if (cursor->docid() != pivot_docid) break;
        cursor->next();
      }
    } else {
      // Drag the laggard with the largest upper bound forward to the pivot.
      // Restricted to cursors strictly behind the pivot: live[0..pivot_index)
      // can contain a cursor already sitting at pivot_docid (a tie at the
      // pivot boundary, which is a normal DAAT occurrence whenever several
      // terms share a document). Picking that one would make next_geq a
      // no-op, and with no cursor advancing the outer loop never terminates.
      // live[0]->docid() < pivot_docid is guaranteed here (we're in this
      // branch precisely because it differs from pivot_docid, and sortedness
      // rules out it being greater), so a candidate always exists.
      PostingCursor* chosen = nullptr;
      for (size_t i = 0; i < pivot_index; i++) {
        if (live[i]->docid() >= pivot_docid) continue;
        if (chosen == nullptr || live[i]->max_impact() > chosen->max_impact()) chosen = live[i];
      }
      chosen->next_geq(pivot_docid);
    }
  }
  return top.drain();
}

// BlockMax-WAND (Ding & Suel 2011). Identical structure to WAND, with one extra
// test: once a pivot is chosen on term-wide bounds, re-check it against the sum
// of the *current blocks'* max impacts. Block bounds are far tighter, so most
// pivots fail this second test and an entire span of documents is skipped
// without decoding a single postings block.
std::vector<Result> Index::block_max_wand(std::vector<PostingCursor>& cursors, uint32_t k,
                                          SearchStats* stats) const {
  TopK top(k);
  std::vector<PostingCursor*> live;
  live.reserve(cursors.size());
  for (PostingCursor& cursor : cursors) live.push_back(&cursor);

  while (true) {
    live.erase(std::remove_if(live.begin(), live.end(),
                              [](const PostingCursor* c) { return c->exhausted(); }),
               live.end());
    if (live.empty()) break;
    sort_by_docid(&live);

    const float threshold = top.threshold();
    float bound = 0.0f;
    size_t pivot_index = live.size();
    for (size_t i = 0; i < live.size(); i++) {
      bound += live[i]->max_impact();
      // >=, not >: see the matching comment in wand() — TopK::push() can
      // let an exact tie in score displace the current worst.
      if (bound >= threshold) {
        pivot_index = i;
        break;
      }
    }
    if (pivot_index == live.size()) break;

    const uint32_t pivot_docid = live[pivot_index]->docid();
    stats->candidates_seen++;

    // Candidates are every cursor at or before pivot_docid — not just the
    // term-wide prefix up to pivot_index. pivot_index only guarantees the
    // *sum* of bounds through that point crosses the threshold; a cursor
    // tied with the pivot on docid can sit right after it (its own term
    // bound wasn't needed to cross the term-wide sum) and still contributes
    // to this document's real score, and matters for how far the hopeless
    // branch below may safely jump.
    size_t candidate_count = pivot_index + 1;
    while (candidate_count < live.size() && live[candidate_count]->docid() == pivot_docid) {
      candidate_count++;
    }

    // Snapshot before skip_to_block mutates docid(): it can reorder
    // candidates relative to cursors past them that are left untouched, at
    // which point an index into live no longer identifies the same cursors.
    const std::vector<PostingCursor*> candidates(live.begin(), live.begin() + candidate_count);

    // Position every candidate cursor's block over the pivot, then sum the
    // block bounds instead of the term bounds.
    float block_bound = 0.0f;
    for (PostingCursor* cursor : candidates) {
      cursor->skip_to_block(pivot_docid);
      if (cursor->exhausted()) {
        block_bound = -1.0f;
        break;
      }
      block_bound += cursor->block_max_impact();
    }
    if (block_bound < 0.0f) continue;  // a cursor ran out; re-filter next round

    if (block_bound >= threshold) {
      // Restore sortedness before trusting live[0]: the skip_to_block loop
      // above just moved every candidate forward independently, which can
      // leave live out of docid order.
      sort_by_docid(&live);

      if (live[0]->docid() == pivot_docid) {
        float score = 0.0f;
        for (PostingCursor* cursor : live) {
          if (cursor->docid() != pivot_docid) break;
          score += bm25_term_score(cursor->idf(), cursor->freq(),
                                   doc_lengths_[pivot_docid], avg_dl_, params_);
          stats->postings_scored++;
        }
        stats->full_evaluations++;
        top.push(pivot_docid, score);
        for (PostingCursor* cursor : live) {
          if (cursor->docid() != pivot_docid) break;
          cursor->next();
        }
      } else {
        // pivot_index indexed the pre-resort order, so it no longer marks
        // where docid == pivot_docid begins; filter on docid directly
        // instead. live is sorted ascending, so the first cursor at or past
        // the pivot ends the scan for everyone after it.
        PostingCursor* chosen = nullptr;
        for (PostingCursor* cursor : live) {
          if (cursor->docid() >= pivot_docid) break;
          if (chosen == nullptr || cursor->max_impact() > chosen->max_impact()) chosen = cursor;
        }
        chosen->next_geq(pivot_docid);
      }
    } else {
      // The whole block span is hopeless using only the candidates' bounds.
      // Jump past the earliest block end among them — but never past a live
      // cursor outside the candidate set. That cursor's docid sits inside
      // the span about to be skipped (candidates covers everything at or
      // before pivot_docid, and this one is the very next docid after), so
      // some document in the span could still combine its term with a
      // candidate's to cross the threshold; the span is only provably
      // hopeless up to that cursor's current position.
      uint32_t next_candidate = kNoDocid;
      for (PostingCursor* cursor : candidates) {
        next_candidate = std::min(next_candidate, cursor->block_max_docid());
      }
      if (next_candidate == kNoDocid) break;
      uint32_t target = next_candidate + 1;
      if (candidate_count < live.size()) {
        target = std::min(target, live[candidate_count]->docid());
      }
      for (PostingCursor* cursor : candidates) {
        if (cursor->docid() < target) cursor->next_geq(target);
      }
    }
  }
  return top.drain();
}

}  // namespace cascade
