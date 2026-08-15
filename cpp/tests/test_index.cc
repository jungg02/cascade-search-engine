// Correctness tests.
//
// The load-bearing one is algorithm equivalence: WAND and BlockMax-WAND are safe
// optimizations, so any difference from exhaustive DAAT-OR is a bug and never a
// tradeoff. The test builds a small index from synthetic documents, runs all
// three over generated queries, and requires the top-k to match exactly.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "analyzer.h"
#include "porter.h"
#include "searcher.h"

namespace fs = std::filesystem;
using namespace cascade;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
  if (condition) {
    std::cout << "  ok   " << what << "\n";
  } else {
    std::cout << "  FAIL " << what << "\n";
    failures++;
  }
  // Redirected stdout is fully buffered, not line-buffered; without this a
  // hang after this point looks like total silence instead of a partial run.
  std::cout.flush();
}

void test_porter_matches_reference_outputs() {
  std::cout << "porter stemmer\n";
  // Values from Porter's own published vocabulary/output pair.
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"caresses", "caress"}, {"ponies", "poni"},     {"ties", "ti"},
      {"caress", "caress"},   {"cats", "cat"},        {"feed", "feed"},
      {"agreed", "agre"},     {"plastered", "plaster"}, {"motoring", "motor"},
      {"sing", "sing"},       {"conflated", "conflat"}, {"troubling", "troubl"},
      {"sized", "size"},      {"hopping", "hop"},     {"falling", "fall"},
      {"hissing", "hiss"},    {"fizzed", "fizz"},     {"failing", "fail"},
      {"filing", "file"},     {"happy", "happi"},     {"sky", "sky"},
      {"relational", "relat"}, {"conditional", "condit"}, {"rational", "ration"},
      {"valenci", "valenc"},  {"hesitanci", "hesit"}, {"digitizer", "digit"},
      {"conformabli", "conform"}, {"radicalli", "radic"}, {"differentli", "differ"},
      {"vileli", "vile"},     {"analogousli", "analog"}, {"vietnamization", "vietnam"},
      {"predication", "predic"}, {"operator", "oper"}, {"feudalism", "feudal"},
      {"decisiveness", "decis"}, {"hopefulness", "hope"}, {"callousness", "callous"},
      {"formaliti", "formal"}, {"sensitiviti", "sensit"}, {"sensibiliti", "sensibl"},
      {"triplicate", "triplic"}, {"formative", "form"}, {"formalize", "formal"},
      {"electriciti", "electr"}, {"electrical", "electr"}, {"hopeful", "hope"},
      {"goodness", "good"},   {"revival", "reviv"},   {"allowance", "allow"},
      {"inference", "infer"}, {"airliner", "airlin"}, {"gyroscopic", "gyroscop"},
      {"adjustable", "adjust"}, {"defensible", "defens"}, {"irritant", "irrit"},
      {"replacement", "replac"}, {"adjustment", "adjust"}, {"dependent", "depend"},
      {"adoption", "adopt"},  {"homologou", "homolog"}, {"communism", "commun"},
      {"activate", "activ"},  {"angulariti", "angular"}, {"homologous", "homolog"},
      {"effective", "effect"}, {"bowdlerize", "bowdler"}, {"probate", "probat"},
      {"rate", "rate"},       {"cease", "ceas"},      {"controll", "control"},
      {"roll", "roll"},
  };
  int wrong = 0;
  for (const auto& [input, expected] : cases) {
    const std::string got = PorterStemmer::stem(input);
    if (got != expected) {
      std::cout << "       " << input << " -> " << got << " (want " << expected << ")\n";
      wrong++;
    }
  }
  check(wrong == 0, "all " + std::to_string(cases.size()) + " reference stems match");
}

void test_analyzer_pipeline() {
  std::cout << "analyzer\n";
  Analyzer analyzer;

  auto terms = analyzer.analyze("The quick brown foxes are running");
  // "the" and "are" are stopwords; the rest are stemmed.
  check(terms == std::vector<std::string>({"quick", "brown", "fox", "run"}),
        "stopwords removed and remaining terms stemmed");

  check(analyzer.analyze("John's cars") == std::vector<std::string>({"john", "car"}),
        "possessive stripped before stemming");

  check(analyzer.analyze("pi is 3.14 and e") == std::vector<std::string>({"pi", "3.14", "e"}),
        "decimal numbers survive as one token");

  auto repeated = analyzer.analyze("cat cat dog");
  check(repeated.size() == 3, "repeats are kept, since term frequency matters");
}

// Builds a corpus with Zipfian term reuse so that some terms are very common and
// some are rare — otherwise every pruning strategy looks equally good.
void write_corpus(const fs::path& path, size_t docs, unsigned seed) {
  std::mt19937 rng(seed);
  std::ofstream out(path);
  const std::vector<std::string> vocabulary = [] {
    std::vector<std::string> words;
    for (int i = 0; i < 400; i++) words.push_back("term" + std::to_string(i));
    return words;
  }();

  for (size_t d = 0; d < docs; d++) {
    out << "doc" << d << "\t";
    const size_t length = 10 + rng() % 60;
    for (size_t i = 0; i < length; i++) {
      // Zipf-ish: small indices are far more likely.
      const size_t rank = static_cast<size_t>(
          std::pow(static_cast<double>(rng() % 10000) / 10000.0, 3.0) * vocabulary.size());
      out << vocabulary[std::min(rank, vocabulary.size() - 1)] << " ";
    }
    out << "\n";
  }
}

bool same_results(const std::vector<Result>& a, const std::vector<Result>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i].docid != b[i].docid) return false;
    if (std::abs(a[i].score - b[i].score) > 1e-4f) return false;
  }
  return true;
}

// A stale pivot re-scoring a document it already emitted is a live failure
// mode for BMW specifically; same_results compares positionally and would
// just report it as a generic mismatch rather than naming the cause.
bool has_duplicate_docid(const std::vector<Result>& results) {
  std::vector<uint32_t> docids;
  docids.reserve(results.size());
  for (const Result& r : results) docids.push_back(r.docid);
  std::sort(docids.begin(), docids.end());
  return std::adjacent_find(docids.begin(), docids.end()) != docids.end();
}

void test_algorithms_agree(const fs::path& index_dir) {
  std::cout << "algorithm equivalence\n";
  Index index(index_dir.string());

  std::mt19937 rng(99);
  int mismatched_wand = 0;
  int mismatched_bmw = 0;
  int duplicated_wand = 0;
  int duplicated_bmw = 0;
  uint64_t exhaustive_postings = 0;
  uint64_t wand_postings = 0;
  uint64_t bmw_postings = 0;

  for (int q = 0; q < 300; q++) {
    std::string query;
    const int terms = 1 + rng() % 5;
    for (int t = 0; t < terms; t++) {
      const size_t rank = static_cast<size_t>(
          std::pow(static_cast<double>(rng() % 10000) / 10000.0, 3.0) * 400);
      query += "term" + std::to_string(std::min(rank, size_t{399})) + " ";
    }

    SearchStats s1, s2, s3;
    const auto exhaustive = index.search(query, 10, Algorithm::kDaatOr, &s1);
    const auto wand = index.search(query, 10, Algorithm::kWand, &s2);
    const auto bmw = index.search(query, 10, Algorithm::kBlockMaxWand, &s3);

    if (!same_results(exhaustive, wand)) mismatched_wand++;
    if (!same_results(exhaustive, bmw)) mismatched_bmw++;
    if (has_duplicate_docid(wand)) duplicated_wand++;
    if (has_duplicate_docid(bmw)) duplicated_bmw++;
    exhaustive_postings += s1.postings_scored;
    wand_postings += s2.postings_scored;
    bmw_postings += s3.postings_scored;
  }

  check(mismatched_wand == 0, "WAND top-10 identical to exhaustive DAAT-OR (300 queries)");
  check(mismatched_bmw == 0,
        "BlockMax-WAND top-10 identical to exhaustive DAAT-OR (300 queries)");
  check(duplicated_wand == 0, "WAND top-10 has no duplicate docids");
  check(duplicated_bmw == 0, "BlockMax-WAND top-10 has no duplicate docids");
  check(bmw_postings <= wand_postings && wand_postings <= exhaustive_postings,
        "postings scored: BMW (" + std::to_string(bmw_postings) + ") <= WAND (" +
            std::to_string(wand_postings) + ") <= exhaustive (" +
            std::to_string(exhaustive_postings) + ")");
  check(bmw_postings < exhaustive_postings,
        "BlockMax-WAND scores fewer postings (" + std::to_string(bmw_postings) + " vs " +
            std::to_string(exhaustive_postings) + ")");
}

void test_index_roundtrip(const fs::path& index_dir, size_t docs) {
  std::cout << "index structure\n";
  Index index(index_dir.string());
  check(index.doc_count() == docs, "doc count round-trips");
  check(index.external_id(0) == "doc0", "external ids round-trip");
  check(index.external_id(docs - 1) == "doc" + std::to_string(docs - 1),
        "last external id round-trips");
  check(index.find_term("term0") != nullptr, "common term is present");
  check(index.find_term("nonexistentterm") == nullptr, "absent term returns null");
}

}  // namespace

int main(int argc, char** argv) {
  test_porter_matches_reference_outputs();
  test_analyzer_pipeline();

  const fs::path temp = fs::temp_directory_path() / "cascade_test_index";
  fs::remove_all(temp);
  fs::create_directories(temp);
  const fs::path corpus = temp / "corpus.tsv";
  const size_t docs = 20000;
  write_corpus(corpus, docs, 7);

  const std::string builder = argc > 1 ? argv[1] : "build/build_index";
  const std::string command = builder + " " + corpus.string() + " " +
                              (temp / "index").string() + " 2>/dev/null";
  if (std::system(command.c_str()) != 0) {
    std::cerr << "index build failed: " << command << "\n";
    return 1;
  }

  test_index_roundtrip(temp / "index", docs);
  test_algorithms_agree(temp / "index");

  fs::remove_all(temp);
  std::cout << (failures == 0 ? "\nall checks passed\n"
                              : "\n" + std::to_string(failures) + " check(s) failed\n");
  return failures == 0 ? 0 : 1;
}
