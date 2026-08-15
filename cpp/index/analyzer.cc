#include "analyzer.h"

#include <algorithm>
#include <cctype>
#include <unordered_set>

#include "porter.h"

namespace cascade {
namespace {

// Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, verbatim. Anserini indexes
// with this set, so our postings lists must be missing exactly these terms.
const std::unordered_set<std::string>& stopwords() {
  static const std::unordered_set<std::string>* kSet = new std::unordered_set<std::string>{
      "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
      "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that",
      "the", "their", "then", "there", "these", "they", "this", "to", "was",
      "will", "with"};
  return *kSet;
}

inline bool is_alnum(unsigned char c) { return std::isalnum(c) != 0; }
inline bool is_alpha(unsigned char c) { return std::isalpha(c) != 0; }
inline bool is_digit(unsigned char c) { return std::isdigit(c) != 0; }

// EnglishPossessiveFilter: strip a trailing "'s". Lucene applies this before
// lowercasing and accepts both straight and curly apostrophes.
void strip_possessive(std::string* token) {
  size_t n = token->size();
  if (n < 3) return;
  char last = (*token)[n - 1];
  char prev = (*token)[n - 2];
  if ((last == 's' || last == 'S') && (prev == '\'')) {
    token->resize(n - 2);
  }
}

}  // namespace

Analyzer::Analyzer() = default;

// Approximates UAX#29 word segmentation for English text: maximal runs of
// alphanumerics, allowing an apostrophe between letters ("don't") and a period
// or comma between digits ("3.14", "1,000"). Bytes >= 0x80 are treated as
// letters so that UTF-8 words survive as single tokens rather than fragmenting.
void Analyzer::tokenize(std::string_view text, std::vector<std::string>* out) {
  const size_t n = text.size();
  size_t i = 0;
  while (i < n) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if (!is_alnum(c) && c < 0x80) {
      i++;
      continue;
    }
    size_t start = i;
    while (i < n) {
      unsigned char ch = static_cast<unsigned char>(text[i]);
      if (is_alnum(ch) || ch >= 0x80) {
        i++;
        continue;
      }
      // An interior separator only continues the token if both neighbours match.
      if (i + 1 < n) {
        unsigned char next = static_cast<unsigned char>(text[i + 1]);
        unsigned char prev = static_cast<unsigned char>(text[i - 1]);
        if (ch == '\'' && is_alpha(prev) && is_alpha(next)) {
          i++;
          continue;
        }
        if ((ch == '.' || ch == ',') && is_digit(prev) && is_digit(next)) {
          i++;
          continue;
        }
      }
      break;
    }
    if (i > start) {
      size_t len = std::min(i - start, kMaxTokenLength);
      out->emplace_back(text.substr(start, len));
    }
  }
}

void Analyzer::analyze(std::string_view text, std::vector<std::string>* out) const {
  std::vector<std::string> raw;
  raw.reserve(64);
  tokenize(text, &raw);

  for (std::string& token : raw) {
    strip_possessive(&token);
    if (token.empty()) continue;
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (stopwords().count(token) != 0) continue;
    out->push_back(PorterStemmer::stem(token));
  }
}

std::vector<std::string> Analyzer::analyze(std::string_view text) const {
  std::vector<std::string> out;
  analyze(text, &out);
  return out;
}

bool Analyzer::is_stopword(const std::string& term) const {
  return stopwords().count(term) != 0;
}

}  // namespace cascade
