// Phase 2 sub-project A server binary: loads the cascade index, wires up
// SearchServiceImpl (worker pool + bounded queue + LRU cache), and serves
// gRPC Search/Query on the given port until killed.
//
// Usage: server_bin <index_dir> [--port=P] [--workers=N] [--queue-depth=D]
//                    [--cache-capacity=C] [--algorithm=wand|blockmax-wand|daat-or]

#include <grpcpp/grpcpp.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "search_service.h"
#include "searcher.h"

namespace {

using cascade::Algorithm;

struct Flags {
  std::string index_dir;
  int port = 50051;
  size_t workers = 4;
  size_t queue_depth = 64;
  size_t cache_capacity = 200;
  Algorithm algorithm = Algorithm::kWand;
};

Algorithm parse_algorithm(const std::string& value) {
  if (value == "wand") return Algorithm::kWand;
  if (value == "blockmax-wand") return Algorithm::kBlockMaxWand;
  if (value == "daat-or") return Algorithm::kDaatOr;
  std::cerr << "unknown --algorithm=" << value << " (want wand|blockmax-wand|daat-or)\n";
  std::exit(1);
}

// Parses "--name=value" out of argv, or returns default_value if absent.
std::string flag_value(int argc, char** argv, const std::string& name,
                        const std::string& default_value) {
  const std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  }
  return default_value;
}

Flags parse_flags(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]).rfind("--", 0) == 0) {
    std::cerr << "usage: server_bin <index_dir> [--port=P] [--workers=N] "
                 "[--queue-depth=D] [--cache-capacity=C] "
                 "[--algorithm=wand|blockmax-wand|daat-or]\n";
    std::exit(1);
  }
  Flags flags;
  flags.index_dir = argv[1];
  flags.port = std::stoi(flag_value(argc, argv, "port", "50051"));
  flags.workers = std::stoul(flag_value(argc, argv, "workers", "4"));
  flags.queue_depth = std::stoul(flag_value(argc, argv, "queue-depth", "64"));
  flags.cache_capacity = std::stoul(flag_value(argc, argv, "cache-capacity", "200"));
  flags.algorithm = parse_algorithm(flag_value(argc, argv, "algorithm", "wand"));
  return flags;
}

}  // namespace

int main(int argc, char** argv) {
  Flags flags = parse_flags(argc, argv);

  std::cerr << "loading index from " << flags.index_dir << "...\n";
  cascade::Index index(flags.index_dir);
  std::cerr << "loaded " << index.doc_count() << " docs\n";

  cascade::SearchServiceImpl service(&index, flags.algorithm, flags.workers,
                                      flags.queue_depth, flags.cache_capacity);

  const std::string address = "127.0.0.1:" + std::to_string(flags.port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Bounds the gRPC layer's own dispatch-thread pool so it can't grow
  // unbounded under load: every request that isn't shed either occupies a
  // worker or sits in the queue, so workers + queue_depth gRPC threads
  // blocked on a future covers that worst case. This is not the concurrency
  // control the plan asks for — WorkerPool/BoundedQueue is — it just keeps
  // gRPC's own bookkeeping bounded too.
  //
  // +8 headroom, not workers + queue_depth exactly: with no headroom, gRPC's
  // own admission control exactly coincides with BoundedQueue's capacity, so
  // gRPC always rejects the (N+1)th concurrent call with its own
  // RESOURCE_EXHAUSTED ("Server Threadpool Exhausted") before that call can
  // ever reach try_submit() — BoundedQueue's own "queue full" rejection path
  // is then dead code from the network's perspective, for any workers/
  // queue_depth split, since a caller can only ever call try_submit() while
  // holding one of these threads. Measured directly: 150 concurrent requests
  // over 150 separate channels against workers=1/queue_depth=1 (quota=2)
  // produced 14 RESOURCE_EXHAUSTED responses, 0 of 14 with detail "queue
  // full" — all from gRPC's own quota. +8 gives gRPC enough slack that
  // requests beyond BoundedQueue's own capacity can still get a dispatch
  // thread and reach try_submit(), making its real rejection observable,
  // while remaining a small fixed buffer rather than defeating the
  // fixed-size intent of the pool.
  grpc::ResourceQuota quota;
  quota.SetMaxThreads(static_cast<int>(flags.workers + flags.queue_depth + 8));
  builder.SetResourceQuota(quota);

  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  std::cerr << "listening on " << address << " (workers=" << flags.workers
            << " queue_depth=" << flags.queue_depth
            << " cache_capacity=" << flags.cache_capacity << ")\n";
  server->Wait();
  return 0;
}
