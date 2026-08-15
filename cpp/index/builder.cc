// Index builder.
//
// The corpus is 8.8M passages and roughly 350M postings, which does not fit in
// memory alongside everything else on an 8GB machine. So the build is
// segment-based, the way Lucene does it: accumulate a bounded number of
// documents in memory, flush a self-contained sorted segment to disk, and
// k-way merge the segments at the end. Peak memory is one segment plus one
// term's postings, not the whole corpus.
//
// Docids are assigned sequentially, so segment i owns a contiguous docid range
// strictly below segment i+1's. That means the merge concatenates postings
// rather than interleaving them, and the result is already docid-ordered.
//
// Input is TSV (`docid \t text`), written by py/baselines/export.py from the
// same ir_datasets text that feeds the Lucene baseline.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "analyzer.h"
#include "index_format.h"

namespace fs = std::filesystem;
using namespace cascade;

namespace {

// Documents per in-memory segment. ~500k passages is roughly 20M postings,
// which is a few hundred MB with map overhead.
constexpr size_t kSegmentDocs = 500000;

struct Posting {
  uint32_t docid;
  uint32_t tf;
};

// One flushed segment: terms in sorted order, each with its postings.
struct SegmentReader {
  std::ifstream in;
  std::string term;
  std::vector<Posting> postings;
  bool exhausted = false;

  explicit SegmentReader(const fs::path& path) : in(path, std::ios::binary) { advance(); }

  void advance() {
    uint32_t term_len = 0;
    if (!in.read(reinterpret_cast<char*>(&term_len), 4)) {
      exhausted = true;
      return;
    }
    term.resize(term_len);
    in.read(term.data(), term_len);

    uint32_t df = 0;
    in.read(reinterpret_cast<char*>(&df), 4);
    postings.resize(df);
    in.read(reinterpret_cast<char*>(postings.data()), static_cast<std::streamsize>(df) * 8);
  }
};

void flush_segment(const std::map<std::string, std::vector<Posting>>& terms,
                   const fs::path& path) {
  std::ofstream out(path, std::ios::binary);
  for (const auto& [term, postings] : terms) {
    uint32_t term_len = static_cast<uint32_t>(term.size());
    uint32_t df = static_cast<uint32_t>(postings.size());
    out.write(reinterpret_cast<const char*>(&term_len), 4);
    out.write(term.data(), term_len);
    out.write(reinterpret_cast<const char*>(&df), 4);
    out.write(reinterpret_cast<const char*>(postings.data()),
              static_cast<std::streamsize>(df) * 8);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: build_index <corpus.tsv> <index_dir> [max_docs]\n";
    return 1;
  }
  const fs::path corpus_path = argv[1];
  const fs::path index_dir = argv[2];
  const size_t max_docs = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;

  fs::create_directories(index_dir);
  const fs::path temp_dir = index_dir / "segments";
  fs::create_directories(temp_dir);

  Analyzer analyzer;
  BM25Params params;

  // ---- Pass 1: analyze, accumulate segments, record document lengths --------
  std::vector<uint32_t> doc_lengths;
  std::string docids_blob;
  std::vector<uint64_t> docid_offsets{0};

  std::map<std::string, std::vector<Posting>> segment;
  std::vector<fs::path> segment_paths;
  size_t docs_in_segment = 0;

  std::ifstream corpus(corpus_path);
  if (!corpus) {
    std::cerr << "cannot open " << corpus_path << "\n";
    return 1;
  }

  std::string line;
  std::vector<std::string> terms;
  std::unordered_map<std::string, uint32_t> term_freqs;
  uint32_t docid = 0;

  while (std::getline(corpus, line)) {
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) continue;

    docids_blob.append(line, 0, tab);
    docid_offsets.push_back(docids_blob.size());

    terms.clear();
    analyzer.analyze(std::string_view(line).substr(tab + 1), &terms);
    doc_lengths.push_back(static_cast<uint32_t>(terms.size()));

    term_freqs.clear();
    for (const std::string& term : terms) term_freqs[term]++;
    for (const auto& [term, tf] : term_freqs) {
      segment[term].push_back({docid, tf});
    }

    docid++;
    docs_in_segment++;
    if (docs_in_segment >= kSegmentDocs) {
      fs::path path = temp_dir / ("seg_" + std::to_string(segment_paths.size()) + ".bin");
      flush_segment(segment, path);
      segment_paths.push_back(path);
      segment.clear();
      docs_in_segment = 0;
      std::cerr << "  flushed segment " << segment_paths.size() << " (" << docid
                << " docs)\n";
    }
    if (max_docs != 0 && docid >= max_docs) break;
  }
  if (!segment.empty()) {
    fs::path path = temp_dir / ("seg_" + std::to_string(segment_paths.size()) + ".bin");
    flush_segment(segment, path);
    segment_paths.push_back(path);
    segment.clear();
  }

  const uint32_t doc_count = docid;
  if (doc_count == 0) {
    std::cerr << "no documents read\n";
    return 1;
  }

  double total_len = 0;
  for (uint32_t len : doc_lengths) total_len += len;
  const float avg_dl = static_cast<float>(total_len / doc_count);
  std::cerr << "indexed " << doc_count << " docs, avg_dl=" << avg_dl << ", "
            << segment_paths.size() << " segments\n";

  {
    std::ofstream out(index_dir / "index.docs", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&doc_count), 4);
    out.write(reinterpret_cast<const char*>(&avg_dl), 4);
    out.write(reinterpret_cast<const char*>(doc_lengths.data()),
              static_cast<std::streamsize>(doc_count) * 4);
  }
  {
    std::ofstream out(index_dir / "index.docids", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&doc_count), 4);
    out.write(reinterpret_cast<const char*>(docid_offsets.data()),
              static_cast<std::streamsize>(docid_offsets.size()) * 8);
    out.write(docids_blob.data(), static_cast<std::streamsize>(docids_blob.size()));
  }

  // ---- Pass 2: k-way merge into the final blocked postings -----------------
  std::vector<std::unique_ptr<SegmentReader>> readers;
  for (const fs::path& path : segment_paths) {
    readers.push_back(std::make_unique<SegmentReader>(path));
  }

  std::ofstream post_out(index_dir / "index.post", std::ios::binary);
  std::vector<TermRecord> records;
  std::string strings_blob;
  uint64_t post_offset = 0;

  std::vector<Posting> merged;
  std::vector<BlockMeta> block_meta;
  std::vector<uint8_t> payload;

  while (true) {
    // Smallest term across all live segments.
    const std::string* smallest = nullptr;
    for (const auto& reader : readers) {
      if (reader->exhausted) continue;
      if (smallest == nullptr || reader->term < *smallest) smallest = &reader->term;
    }
    if (smallest == nullptr) break;
    const std::string term = *smallest;

    merged.clear();
    for (const auto& reader : readers) {
      if (reader->exhausted || reader->term != term) continue;
      merged.insert(merged.end(), reader->postings.begin(), reader->postings.end());
      reader->advance();
    }

    const uint32_t df = static_cast<uint32_t>(merged.size());
    const float idf = bm25_idf(doc_count, df);

    block_meta.clear();
    payload.clear();
    float term_max_impact = 0.0f;

    for (size_t start = 0; start < merged.size(); start += kBlockSize) {
      const size_t end = std::min(start + kBlockSize, merged.size());
      BlockMeta meta{};
      meta.count = static_cast<uint32_t>(end - start);
      meta.max_docid = merged[end - 1].docid;
      meta.payload_offset = static_cast<uint32_t>(payload.size());

      // Gaps within the block; the first docid of each block is absolute so a
      // skipped block never has to be decoded to resynchronize.
      uint32_t previous = 0;
      float block_max = 0.0f;
      for (size_t i = start; i < end; i++) {
        const uint32_t gap = (i == start) ? merged[i].docid : merged[i].docid - previous;
        put_varint(&payload, gap);
        previous = merged[i].docid;
        const float impact = bm25_term_score(idf, merged[i].tf,
                                             doc_lengths[merged[i].docid], avg_dl, params);
        block_max = std::max(block_max, impact);
      }
      for (size_t i = start; i < end; i++) put_varint(&payload, merged[i].tf);

      meta.max_impact = block_max;
      term_max_impact = std::max(term_max_impact, block_max);
      block_meta.push_back(meta);
    }

    TermRecord record{};
    record.str_offset = strings_blob.size();
    record.str_len = static_cast<uint32_t>(term.size());
    record.df = df;
    record.num_blocks = static_cast<uint32_t>(block_meta.size());
    record.max_impact = term_max_impact;
    record.post_offset = post_offset;
    strings_blob.append(term);
    records.push_back(record);

    post_out.write(reinterpret_cast<const char*>(block_meta.data()),
                   static_cast<std::streamsize>(block_meta.size()) * sizeof(BlockMeta));
    post_out.write(reinterpret_cast<const char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
    post_offset += block_meta.size() * sizeof(BlockMeta) + payload.size();

    if (records.size() % 500000 == 0) {
      std::cerr << "  merged " << records.size() << " terms\n";
    }
  }
  post_out.close();

  {
    std::ofstream out(index_dir / "index.terms", std::ios::binary);
    const uint32_t term_count = static_cast<uint32_t>(records.size());
    out.write(reinterpret_cast<const char*>(&term_count), 4);
    out.write(reinterpret_cast<const char*>(records.data()),
              static_cast<std::streamsize>(records.size()) * sizeof(TermRecord));
    out.write(strings_blob.data(), static_cast<std::streamsize>(strings_blob.size()));
  }

  for (const fs::path& path : segment_paths) fs::remove(path);
  fs::remove(temp_dir);

  std::cerr << "wrote " << records.size() << " terms, " << post_offset
            << " bytes of postings\n";
  return 0;
}
