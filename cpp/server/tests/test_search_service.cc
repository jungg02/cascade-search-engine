// Correctness tests for SearchServiceImpl's request handling: cache wiring
// (miss then hit, query normalization makes near-identical queries hit the
// same entry) over a tiny synthetic index. Concurrency and load-shedding
// under real timing are covered by the Python integration test against the
// actual running server (py/server/tests/test_server_integration.py) — an
// in-process call here has no network latency to make two calls reliably
// overlap, so it is not the right place to test RESOURCE_EXHAUSTED.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "search_service.h"
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
  std::cout.flush();
}

fs::path build_tiny_index(const std::string& builder) {
  const fs::path temp = fs::temp_directory_path() / "cascade_server_test_index";
  fs::remove_all(temp);
  fs::create_directories(temp);
  const fs::path corpus = temp / "corpus.tsv";
  {
    std::ofstream out(corpus);
    out << "doc0\tthe quick brown fox\n";
    out << "doc1\tthe lazy dog sleeps\n";
    out << "doc2\tfox and dog play\n";
  }
  const fs::path index_dir = temp / "index";
  const std::string command =
      builder + " " + corpus.string() + " " + index_dir.string() + " 2>/dev/null";
  if (std::system(command.c_str()) != 0) {
    std::cerr << "index build failed: " << command << "\n";
    std::exit(1);
  }
  return index_dir;
}

SearchRequest make_request(const std::string& query, uint32_t k = 10) {
  SearchRequest req;
  req.set_query(query);
  req.set_k(k);
  return req;
}

void test_cache_miss_then_hit(const Index& index) {
  std::cout << "cache miss then hit\n";
  SearchServiceImpl service(&index, Algorithm::kWand, /*num_workers=*/2,
                            /*queue_depth=*/4, /*cache_capacity=*/8);

  SearchResponse first;
  auto req = make_request("fox");
  grpc::Status status = service.Query(nullptr, &req, &first);
  check(status.ok(), "first call succeeds");
  check(!first.cache_hit(), "first call is a cache miss");
  check(first.results_size() > 0, "first call returns results for a term that appears (fox)");

  SearchResponse second;
  status = service.Query(nullptr, &req, &second);
  check(status.ok(), "second call succeeds");
  check(second.cache_hit(), "second identical call is a cache hit");
  check(second.results_size() == first.results_size(), "cached results have the same count");
  for (int i = 0; i < first.results_size(); i++) {
    check(second.results(i).docid() == first.results(i).docid() &&
              second.results(i).score() == first.results(i).score(),
          "cached result " + std::to_string(i) + " matches the original");
  }
}

void test_normalization_shares_cache_entry(const Index& index) {
  std::cout << "normalization\n";
  SearchServiceImpl service(&index, Algorithm::kWand, /*num_workers=*/2,
                            /*queue_depth=*/4, /*cache_capacity=*/8);

  SearchResponse warm;
  auto warm_req = make_request("Fox Dog");
  check(service.Query(nullptr, &warm_req, &warm).ok(), "warm-up call succeeds");
  check(!warm.cache_hit(), "warm-up call is a miss");

  SearchResponse variant;
  auto variant_req = make_request("  fox   dog  ");
  check(service.Query(nullptr, &variant_req, &variant).ok(), "differently-spaced call succeeds");
  check(variant.cache_hit(),
        "differently-cased, differently-spaced query normalizes to the same cache key");
}

void test_different_queries_miss_independently(const Index& index) {
  std::cout << "distinct queries\n";
  SearchServiceImpl service(&index, Algorithm::kWand, /*num_workers=*/2,
                            /*queue_depth=*/4, /*cache_capacity=*/8);
  SearchResponse fox_resp, dog_resp;
  auto fox_req = make_request("fox");
  auto dog_req = make_request("dog");
  check(service.Query(nullptr, &fox_req, &fox_resp).ok(), "fox call succeeds");
  check(service.Query(nullptr, &dog_req, &dog_resp).ok(), "dog call succeeds");
  check(!dog_resp.cache_hit(), "a genuinely different query is not a cache hit off fox's entry");
}

void test_different_k_misses_independently(const Index& index) {
  std::cout << "distinct k\n";
  SearchServiceImpl service(&index, Algorithm::kWand, /*num_workers=*/2,
                            /*queue_depth=*/4, /*cache_capacity=*/8);

  SearchResponse k10_resp;
  auto k10_req = make_request("fox", /*k=*/10);
  check(service.Query(nullptr, &k10_req, &k10_resp).ok(), "k=10 call succeeds");
  check(!k10_resp.cache_hit(), "k=10 call is a miss");

  SearchResponse k1_resp;
  auto k1_req = make_request("fox", /*k=*/1);
  check(service.Query(nullptr, &k1_req, &k1_resp).ok(), "k=1 call succeeds");
  check(!k1_resp.cache_hit(),
        "same query text with a different k is not a cache hit off the k=10 entry");
  check(k1_resp.results_size() == 1,
        "k=1 call returns exactly 1 result, not the k=10 call's result count");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string builder = argc > 1 ? argv[1] : "build/build_index";
  const fs::path index_dir = build_tiny_index(builder);
  Index index(index_dir.string());

  test_cache_miss_then_hit(index);
  test_normalization_shares_cache_entry(index);
  test_different_queries_miss_independently(index);
  test_different_k_misses_independently(index);

  fs::remove_all(index_dir.parent_path());
  std::cout << (failures == 0 ? "\nall checks passed\n"
                              : "\n" + std::to_string(failures) + " check(s) failed\n");
  return failures == 0 ? 0 : 1;
}
