// pybind11 binding.
//
// The Python side never reimplements tokenization or scoring; it calls into the
// same objects the C++ benchmark uses. That is the point of the binding: one
// analyzer, one BM25, no offline/online skew.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "analyzer.h"
#include "searcher.h"

namespace py = pybind11;
using namespace cascade;

PYBIND11_MODULE(cascade_index, m) {
  m.doc() = "Cascade C++ inverted index with BlockMax-WAND";

  py::enum_<Algorithm>(m, "Algorithm")
      .value("DAAT_OR", Algorithm::kDaatOr)
      .value("WAND", Algorithm::kWand)
      .value("BLOCKMAX_WAND", Algorithm::kBlockMaxWand);

  py::class_<SearchStats>(m, "SearchStats")
      .def(py::init<>())
      .def_readonly("postings_scored", &SearchStats::postings_scored)
      .def_readonly("full_evaluations", &SearchStats::full_evaluations)
      .def_readonly("blocks_decoded", &SearchStats::blocks_decoded)
      .def_readonly("candidates_seen", &SearchStats::candidates_seen);

  py::class_<Analyzer>(m, "Analyzer")
      .def(py::init<>())
      .def("analyze",
           [](const Analyzer& self, const std::string& text) { return self.analyze(text); },
           py::arg("text"),
           "Tokenize, lowercase, drop stopwords, and Porter-stem.")
      .def("is_stopword", &Analyzer::is_stopword);

  py::class_<Index>(m, "Index")
      .def(py::init<const std::string&>(), py::arg("directory"))
      .def_property_readonly("doc_count", &Index::doc_count)
      .def_property_readonly("term_count", &Index::term_count)
      .def_property_readonly("avg_dl", &Index::avg_dl)
      .def("external_id", &Index::external_id)
      .def(
          "search",
          [](const Index& self, const std::string& query, uint32_t k, Algorithm algorithm) {
            SearchStats stats;
            std::vector<Result> results;
            {
              // The GIL is not needed while searching, and releasing it is what
              // lets the Phase 2 load generator drive real concurrency.
              py::gil_scoped_release release;
              results = self.search(query, k, algorithm, &stats);
            }
            std::vector<std::pair<std::string, float>> out;
            out.reserve(results.size());
            for (const Result& result : results) {
              out.emplace_back(self.external_id(result.docid), result.score);
            }
            return py::make_tuple(out, stats);
          },
          py::arg("query"), py::arg("k") = 1000,
          py::arg("algorithm") = Algorithm::kBlockMaxWand,
          "Returns ((external_doc_id, score) list, SearchStats).")
      .def(
          "search_docids",
          [](const Index& self, const std::string& query, uint32_t k, Algorithm algorithm) {
            SearchStats stats;
            std::vector<Result> results;
            {
              py::gil_scoped_release release;
              results = self.search(query, k, algorithm, &stats);
            }
            std::vector<std::pair<uint32_t, float>> out;
            out.reserve(results.size());
            for (const Result& result : results) out.emplace_back(result.docid, result.score);
            return py::make_tuple(out, stats);
          },
          py::arg("query"), py::arg("k") = 1000,
          py::arg("algorithm") = Algorithm::kBlockMaxWand,
          "Internal docids, for the algorithm-equivalence check.");
}
