// On-disk index layout and the shared BM25 scoring function.
//
//   index.docs    [u32 doc_count][f32 avg_dl][u32 doc_len]*
//   index.terms   [u32 term_count][TermRecord]*[term strings blob]
//   index.post    per term: [BlockMeta]* then the block payloads
//   index.docids  [u32 count][u64 offset]*(count+1)[external id blob]
//
// Postings are stored in fixed-size blocks. Each block carries its max docid
// (so a cursor can skip a whole block without decoding it) and its max BM25
// impact (which is what makes BlockMax-WAND possible: the pivot test can use a
// bound that is tight for the current block rather than for the whole term).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace cascade {

// 128 docs per block. Large enough that per-block metadata is a small fraction
// of the postings, small enough that a block max-impact is much tighter than a
// term-wide upper bound.
constexpr uint32_t kBlockSize = 128;

// Anserini's msmarco-passage defaults, matched so the Phase 0 baseline and this
// index differ only in implementation.
struct BM25Params {
  float k1 = 0.9f;
  float b = 0.4f;
};

#pragma pack(push, 1)
struct TermRecord {
  uint64_t str_offset;    // into the term strings blob
  uint64_t post_offset;   // into index.post
  uint32_t str_len;
  uint32_t df;
  uint32_t num_blocks;
  float max_impact;       // term-wide upper bound, for plain WAND
};

struct BlockMeta {
  uint32_t max_docid;
  uint32_t count;
  uint32_t payload_offset;  // relative to the term's postings start
  float max_impact;
};
#pragma pack(pop)

static_assert(sizeof(TermRecord) == 32, "TermRecord layout changed");
static_assert(sizeof(BlockMeta) == 16, "BlockMeta layout changed");

// BM25 as Lucene computes it, so a score difference is a length-normalization
// difference and nothing else.
//   idf = ln(1 + (N - df + 0.5) / (df + 0.5))
inline float bm25_idf(uint32_t doc_count, uint32_t df) {
  return std::log(1.0f + (static_cast<float>(doc_count) - static_cast<float>(df) + 0.5f) /
                             (static_cast<float>(df) + 0.5f));
}

inline float bm25_term_score(float idf, uint32_t tf, uint32_t doc_len, float avg_dl,
                             const BM25Params& p) {
  const float norm = p.k1 * (1.0f - p.b + p.b * static_cast<float>(doc_len) / avg_dl);
  return idf * (static_cast<float>(tf) * (p.k1 + 1.0f)) / (static_cast<float>(tf) + norm);
}

// LEB128. Docid gaps and term frequencies are both small and heavily skewed
// toward 1, so a byte-oriented varint beats a fixed width by roughly 3x here.
inline void put_varint(std::vector<uint8_t>* out, uint32_t value) {
  while (value >= 0x80) {
    out->push_back(static_cast<uint8_t>(value) | 0x80);
    value >>= 7;
  }
  out->push_back(static_cast<uint8_t>(value));
}

inline uint32_t get_varint(const uint8_t* data, size_t* pos) {
  uint32_t result = 0;
  int shift = 0;
  while (true) {
    uint8_t byte = data[(*pos)++];
    result |= static_cast<uint32_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) return result;
    shift += 7;
  }
}

}  // namespace cascade
