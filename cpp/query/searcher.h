// Index reader and the three query evaluation strategies.
//
// All three return the same top-k. WAND and BlockMax-WAND are *safe*
// optimizations: they skip documents that provably cannot enter the heap, so a
// difference in results is a bug and never a tradeoff. What changes between them
// is how much work they do to get there, which is why SearchStats counts
// postings scored and full evaluations rather than just measuring time.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "analyzer.h"
#include "index_format.h"

namespace cascade {

enum class Algorithm { kDaatOr, kWand, kBlockMaxWand };

struct SearchStats {
  uint64_t postings_scored = 0;   // per-term contributions actually computed
  uint64_t full_evaluations = 0;  // documents scored across all their terms
  uint64_t blocks_decoded = 0;    // block payloads varint-decoded
  uint64_t candidates_seen = 0;   // docids considered as pivots
};

struct Result {
  uint32_t docid;
  float score;
};

// A cursor over one term's blocked postings list.
class PostingCursor {
 public:
  PostingCursor(const BlockMeta* blocks, uint32_t num_blocks, const uint8_t* payload,
                float idf, float max_impact, SearchStats* stats);

  uint32_t docid() const { return docid_; }
  uint32_t freq() const { return freq_; }
  bool exhausted() const { return exhausted_; }
  float max_impact() const { return max_impact_; }

  // Upper bound for the block currently positioned over. Tighter than
  // max_impact() and the reason BlockMax-WAND prunes more than WAND.
  float block_max_impact() const { return blocks_[block_index_].max_impact; }
  uint32_t block_max_docid() const { return blocks_[block_index_].max_docid; }

  void next();
  // Advance to the first docid >= target.
  void next_geq(uint32_t target);
  // Position the block pointer over target, skipping intermediate blocks on
  // their max_docid alone (BlockMax-WAND's pivot test wants block bounds
  // without paying to decode blocks it may reject). If this lands on a new
  // block, that one block is decoded so docid()/freq() stay accurate — the
  // laziness only applies to blocks skipped past, not the one landed on.
  void skip_to_block(uint32_t target);

 private:
  void decode_block();

  const BlockMeta* blocks_;
  uint32_t num_blocks_;
  const uint8_t* payload_;
  float idf_;
  float max_impact_;
  SearchStats* stats_;

  uint32_t block_index_ = 0;
  uint32_t position_ = 0;
  uint32_t docid_ = 0;
  uint32_t freq_ = 0;
  bool exhausted_ = false;

  uint32_t docids_[kBlockSize];
  uint32_t freqs_[kBlockSize];

 public:
  float idf() const { return idf_; }
};

class Index {
 public:
  explicit Index(const std::string& directory);
  ~Index();

  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;

  uint32_t doc_count() const { return doc_count_; }
  float avg_dl() const { return avg_dl_; }
  uint32_t doc_length(uint32_t docid) const { return doc_lengths_[docid]; }
  std::string external_id(uint32_t docid) const;
  uint32_t term_count() const { return term_count_; }

  // nullptr when the term is absent.
  const TermRecord* find_term(const std::string& term) const;
  const BlockMeta* blocks_for(const TermRecord& record) const;
  const uint8_t* payload_for(const TermRecord& record) const;

  std::vector<Result> search(const std::string& query, uint32_t k, Algorithm algorithm,
                             SearchStats* stats) const;

  const Analyzer& analyzer() const { return analyzer_; }

 private:
  std::vector<PostingCursor> open_cursors(const std::string& query,
                                          SearchStats* stats) const;
  std::vector<Result> daat_or(std::vector<PostingCursor>& cursors, uint32_t k,
                              SearchStats* stats) const;
  std::vector<Result> wand(std::vector<PostingCursor>& cursors, uint32_t k,
                           SearchStats* stats) const;
  std::vector<Result> block_max_wand(std::vector<PostingCursor>& cursors, uint32_t k,
                                     SearchStats* stats) const;

  void* map_file(const std::string& path, size_t* size);

  Analyzer analyzer_;
  BM25Params params_;

  uint32_t doc_count_ = 0;
  float avg_dl_ = 0;
  const uint32_t* doc_lengths_ = nullptr;

  uint32_t term_count_ = 0;
  const TermRecord* term_records_ = nullptr;
  const char* term_strings_ = nullptr;

  const uint8_t* postings_ = nullptr;

  const uint64_t* docid_offsets_ = nullptr;
  const char* docid_blob_ = nullptr;

  std::vector<std::pair<void*, size_t>> mappings_;
};

}  // namespace cascade
