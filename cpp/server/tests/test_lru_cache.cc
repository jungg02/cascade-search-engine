// Correctness tests for LruCache: miss on empty, hit returns the value,
// eviction is by recency of *use* (not insertion), and get() itself counts
// as a use.

#include <iostream>
#include <string>

#include "lru_cache.h"

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

void test_miss_on_empty() {
  std::cout << "miss\n";
  LruCache<int> cache(2);
  int out = 0;
  check(!cache.get("a", &out), "miss on empty cache");
}

void test_put_then_get() {
  std::cout << "put/get\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  int out = 0;
  check(cache.get("a", &out) && out == 1, "get returns the value put in");
  check(cache.size() == 1, "size reflects one entry");
}

void test_overwrite_existing_key() {
  std::cout << "overwrite\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  cache.put("a", 2);
  int out = 0;
  check(cache.get("a", &out) && out == 2, "put on existing key overwrites the value");
  check(cache.size() == 1, "overwriting does not grow size");
}

void test_evicts_least_recently_used() {
  std::cout << "eviction order\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  cache.put("b", 2);
  // "a" is now the least recently used of the two.
  cache.put("c", 3);  // should evict "a", not "b"
  int out = 0;
  check(!cache.get("a", &out), "least-recently-used entry (a) was evicted");
  check(cache.get("b", &out) && out == 2, "b survives eviction");
  check(cache.get("c", &out) && out == 3, "c (just inserted) survives eviction");
}

void test_get_counts_as_use() {
  std::cout << "get refreshes recency\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  cache.put("b", 2);
  int out = 0;
  check(cache.get("a", &out), "touch a via get, making b the least recently used");
  cache.put("c", 3);  // should evict "b", since "a" was just used via get()
  check(!cache.get("b", &out), "b (not touched since a's get) was evicted");
  check(cache.get("a", &out) && out == 1, "a survives because get() refreshed it");
  check(cache.get("c", &out) && out == 3, "c (just inserted) survives");
}

}  // namespace

int main() {
  test_miss_on_empty();
  test_put_then_get();
  test_overwrite_existing_key();
  test_evicts_least_recently_used();
  test_get_counts_as_use();
  std::cout << (failures == 0 ? "\nall checks passed\n"
                              : "\n" + std::to_string(failures) + " check(s) failed\n");
  return failures == 0 ? 0 : 1;
}
