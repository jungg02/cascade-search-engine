// Ties BoundedQueue/WorkerPool, LruCache, and cascade::Index together behind
// the generated Search::Service RPC interface. A cache hit skips Index::search
// entirely; a miss runs it on a worker thread pulled from the fixed pool, with
// RESOURCE_EXHAUSTED returned immediately (not queued) when the pool's queue
// is already at its configured depth.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lru_cache.h"
#include "search.grpc.pb.h"
#include "searcher.h"
#include "worker_pool.h"

namespace cascade {

class SearchServiceImpl final : public Search::Service {
 public:
  // Does not take ownership of index; the caller (main.cc) must keep it
  // alive for the lifetime of this service. index_->search() is const and
  // touches no shared mutable state (verified by reading searcher.cc), so
  // concurrent calls from multiple worker threads are safe without extra
  // locking here.
  SearchServiceImpl(const Index* index, Algorithm algorithm, size_t num_workers,
                     size_t queue_depth, size_t cache_capacity);

  grpc::Status Query(grpc::ServerContext* context, const SearchRequest* request,
                      SearchResponse* response) override;

 private:
  // The actual query logic, run on a worker thread. Populates response
  // (results + cache_hit) directly; does not touch queue_wait_us, which
  // Query() fills in from timestamps only it has.
  void handle_query(const SearchRequest& request, SearchResponse* response);

  static std::string normalize(const std::string& query);

  const Index* index_;
  Algorithm algorithm_;
  LruCache<std::vector<Result>> cache_;
  WorkerPool pool_;
};

}  // namespace cascade
