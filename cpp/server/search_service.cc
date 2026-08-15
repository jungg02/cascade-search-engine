#include "search_service.h"

#include <cctype>
#include <chrono>
#include <future>
#include <memory>

namespace cascade {

namespace {
uint32_t microseconds_between(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end) {
  return static_cast<uint32_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}
}  // namespace

SearchServiceImpl::SearchServiceImpl(const Index* index, Algorithm algorithm,
                                      size_t num_workers, size_t queue_depth,
                                      size_t cache_capacity)
    : index_(index),
      algorithm_(algorithm),
      cache_(cache_capacity),
      pool_(num_workers, queue_depth) {}

std::string SearchServiceImpl::normalize(const std::string& query) {
  std::string out;
  out.reserve(query.size());
  bool last_was_space = true;  // collapses leading whitespace too
  for (char c : query) {
    char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (std::isspace(static_cast<unsigned char>(lower))) {
      if (!last_was_space) out.push_back(' ');
      last_was_space = true;
    } else {
      out.push_back(lower);
      last_was_space = false;
    }
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

void SearchServiceImpl::handle_query(const SearchRequest& request, SearchResponse* response) {
  const std::string key = normalize(request.query()) + "|" + std::to_string(request.k());

  std::vector<Result> cached;
  if (cache_.get(key, &cached)) {
    response->set_cache_hit(true);
    for (const Result& r : cached) {
      SearchResult* out = response->add_results();
      out->set_docid(r.docid);
      out->set_score(r.score);
    }
    return;
  }

  SearchStats stats;
  std::vector<Result> results = index_->search(request.query(), request.k(), algorithm_, &stats);
  response->set_cache_hit(false);
  for (const Result& r : results) {
    SearchResult* out = response->add_results();
    out->set_docid(r.docid);
    out->set_score(r.score);
  }
  cache_.put(key, results);
}

grpc::Status SearchServiceImpl::Query(grpc::ServerContext*, const SearchRequest* request,
                                       SearchResponse* response) {
  auto promise = std::make_shared<std::promise<void>>();
  std::future<void> future = promise->get_future();
  const auto queued_at = std::chrono::steady_clock::now();

  const bool accepted = pool_.try_submit([this, request, response, promise, queued_at] {
    response->set_queue_wait_us(
        microseconds_between(queued_at, std::chrono::steady_clock::now()));
    handle_query(*request, response);
    promise->set_value();
  });

  if (!accepted) {
    return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "queue full");
  }
  future.wait();
  return grpc::Status::OK;
}

}  // namespace cascade
