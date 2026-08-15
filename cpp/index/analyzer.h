// Text analysis, shared by indexing and querying.
//
// The plan calls this the training/serving-skew defense: there is exactly one
// implementation of tokenization, and both the index builder and the query path
// call it. The Python side reaches it through the pybind11 binding rather than
// reimplementing it, so the two can't drift.
//
// The pipeline mirrors Lucene's EnglishAnalyzer, which is Anserini's default and
// therefore what the Phase 0 baseline index was built with:
//
//     StandardTokenizer -> EnglishPossessiveFilter -> LowerCaseFilter
//       -> StopFilter(ENGLISH_STOP_WORDS_SET) -> PorterStemFilter
//
// The tokenizer is an ASCII approximation of UAX#29 rather than a full
// implementation. That is a known and deliberate divergence: it is the first
// place to look if our NDCG@10 misses Lucene's by more than the plan's 0.01.

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cascade {

class Analyzer {
 public:
  Analyzer();

  // Appends analyzed terms to `out`. Terms are lowercased, stopword-filtered,
  // and Porter-stemmed. Duplicate terms are kept: term frequency matters.
  void analyze(std::string_view text, std::vector<std::string>* out) const;

  std::vector<std::string> analyze(std::string_view text) const;

  bool is_stopword(const std::string& term) const;

  // Lucene's StandardTokenizer default; longer runs are split.
  static constexpr size_t kMaxTokenLength = 255;

 private:
  // Raw tokens, before lowercasing and filtering.
  static void tokenize(std::string_view text, std::vector<std::string>* out);
};

}  // namespace cascade
