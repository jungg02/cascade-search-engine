// Porter stemmer (Porter 1980), the algorithm behind Lucene's PorterStemFilter.
//
// Lucene's EnglishAnalyzer — Anserini's default, and therefore what our Phase 0
// baseline was indexed with — applies this stemmer. Matching it is a prerequisite
// for the plan's "within ~0.01 NDCG@10 of Lucene" check: a stemmer that diverges
// on common suffixes changes df, idf, and the postings themselves.
//
// This is deliberately the original Porter algorithm, not Porter2/Snowball. They
// differ on a nontrivial fraction of words, and Lucene uses the former.

#pragma once

#include <string>

namespace cascade {

class PorterStemmer {
 public:
  // Stems in place. Returns the new length.
  static std::string stem(const std::string& word);

 private:
  static bool is_consonant(const std::string& s, int i);
  // Measure: the number of vowel-consonant sequences in s[0..j].
  static int measure(const std::string& s, int j);
  static bool has_vowel(const std::string& s, int j);
  static bool double_consonant(const std::string& s, int j);
  // CVC where the final consonant is not w, x or y.
  static bool cvc(const std::string& s, int i);
};

}  // namespace cascade
