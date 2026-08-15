#include "porter.h"

#include <cstring>

namespace cascade {
namespace {

// Working buffer shared by the step functions. `k` is the index of the last
// character; `j` is a scratch offset used by the suffix rules.
struct Stemmer {
  std::string b;
  int k = 0;
  int j = 0;

  bool consonant(int i) const {
    switch (b[i]) {
      case 'a': case 'e': case 'i': case 'o': case 'u':
        return false;
      case 'y':
        return i == 0 ? true : !consonant(i - 1);
      default:
        return true;
    }
  }

  // Number of vowel-consonant sequences between 0 and j.
  int measure() const {
    int n = 0;
    int i = 0;
    while (true) {
      if (i > j) return n;
      if (!consonant(i)) break;
      i++;
    }
    i++;
    while (true) {
      while (true) {
        if (i > j) return n;
        if (consonant(i)) break;
        i++;
      }
      i++;
      n++;
      while (true) {
        if (i > j) return n;
        if (!consonant(i)) break;
        i++;
      }
      i++;
    }
  }

  bool vowel_in_stem() const {
    for (int i = 0; i <= j; i++) {
      if (!consonant(i)) return true;
    }
    return false;
  }

  bool double_consonant(int i) const {
    if (i < 1) return false;
    if (b[i] != b[i - 1]) return false;
    return consonant(i);
  }

  // consonant-vowel-consonant, where the last is not w, x or y. Used to decide
  // whether a short word needs an -e restored.
  bool cvc(int i) const {
    if (i < 2 || !consonant(i) || consonant(i - 1) || !consonant(i - 2)) return false;
    char ch = b[i];
    return !(ch == 'w' || ch == 'x' || ch == 'y');
  }

  bool ends(const char* s) {
    int len = static_cast<int>(std::strlen(s));
    if (len > k + 1) return false;
    if (b.compare(k - len + 1, len, s) != 0) return false;
    j = k - len;
    return true;
  }

  void setto(const char* s) {
    int len = static_cast<int>(std::strlen(s));
    b.replace(j + 1, b.size() - (j + 1), s, len);
    k = j + len;
  }

  void r(const char* s) {
    if (measure() > 0) setto(s);
  }

  // Plurals and -ed / -ing.
  void step1ab() {
    if (b[k] == 's') {
      if (ends("sses")) {
        k -= 2;
      } else if (ends("ies")) {
        setto("i");
      } else if (b[k - 1] != 's') {
        k--;
      }
    }
    if (ends("eed")) {
      if (measure() > 0) k--;
    } else if ((ends("ed") || ends("ing")) && vowel_in_stem()) {
      k = j;
      if (ends("at")) {
        setto("ate");
      } else if (ends("bl")) {
        setto("ble");
      } else if (ends("iz")) {
        setto("ize");
      } else if (double_consonant(k)) {
        k--;
        char ch = b[k];
        if (ch == 'l' || ch == 's' || ch == 'z') k++;
      } else if (measure() == 1 && cvc(k)) {
        setto("e");
      }
    }
  }

  // Terminal y to i when there is another vowel in the stem.
  void step1c() {
    if (ends("y") && vowel_in_stem()) b[k] = 'i';
  }

  // Double suffixes to single ones.
  void step2() {
    if (k == 0) return;
    switch (b[k - 1]) {
      case 'a':
        if (ends("ational")) { r("ate"); break; }
        if (ends("tional")) { r("tion"); break; }
        break;
      case 'c':
        if (ends("enci")) { r("ence"); break; }
        if (ends("anci")) { r("ance"); break; }
        break;
      case 'e':
        if (ends("izer")) { r("ize"); break; }
        break;
      case 'l':
        if (ends("bli")) { r("ble"); break; }
        if (ends("alli")) { r("al"); break; }
        if (ends("entli")) { r("ent"); break; }
        if (ends("eli")) { r("e"); break; }
        if (ends("ousli")) { r("ous"); break; }
        break;
      case 'o':
        if (ends("ization")) { r("ize"); break; }
        if (ends("ation")) { r("ate"); break; }
        if (ends("ator")) { r("ate"); break; }
        break;
      case 's':
        if (ends("alism")) { r("al"); break; }
        if (ends("iveness")) { r("ive"); break; }
        if (ends("fulness")) { r("ful"); break; }
        if (ends("ousness")) { r("ous"); break; }
        break;
      case 't':
        if (ends("aliti")) { r("al"); break; }
        if (ends("iviti")) { r("ive"); break; }
        if (ends("biliti")) { r("ble"); break; }
        break;
      case 'g':
        if (ends("logi")) { r("log"); break; }
        break;
      default:
        break;
    }
  }

  void step3() {
    switch (b[k]) {
      case 'e':
        if (ends("icate")) { r("ic"); break; }
        if (ends("ative")) { r(""); break; }
        if (ends("alize")) { r("al"); break; }
        break;
      case 'i':
        if (ends("iciti")) { r("ic"); break; }
        break;
      case 'l':
        if (ends("ical")) { r("ic"); break; }
        if (ends("ful")) { r(""); break; }
        break;
      case 's':
        if (ends("ness")) { r(""); break; }
        break;
      default:
        break;
    }
  }

  // Remaining suffixes when the measure is greater than 1.
  void step4() {
    if (k == 0) return;
    switch (b[k - 1]) {
      case 'a': if (ends("al")) break; return;
      case 'c': if (ends("ance")) break; if (ends("ence")) break; return;
      case 'e': if (ends("er")) break; return;
      case 'i': if (ends("ic")) break; return;
      case 'l': if (ends("able")) break; if (ends("ible")) break; return;
      case 'n':
        if (ends("ant")) break;
        if (ends("ement")) break;
        if (ends("ment")) break;
        if (ends("ent")) break;
        return;
      case 'o':
        if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
        if (ends("ou")) break;
        return;
      case 's': if (ends("ism")) break; return;
      case 't': if (ends("ate")) break; if (ends("iti")) break; return;
      case 'u': if (ends("ous")) break; return;
      case 'v': if (ends("ive")) break; return;
      case 'z': if (ends("ize")) break; return;
      default: return;
    }
    if (measure() > 1) k = j;
  }

  // Terminal -e, and -ll to -l.
  void step5() {
    j = k;
    if (b[k] == 'e') {
      int a = measure();
      if (a > 1 || (a == 1 && !cvc(k - 1))) k--;
    }
    if (b[k] == 'l' && double_consonant(k) && measure() > 1) k--;
  }
};

}  // namespace

std::string PorterStemmer::stem(const std::string& word) {
  // Words of two characters or fewer are returned unchanged, as in Porter's
  // reference implementation.
  if (word.size() <= 2) return word;

  Stemmer s;
  s.b = word;
  s.k = static_cast<int>(word.size()) - 1;

  s.step1ab();
  s.step1c();
  s.step2();
  s.step3();
  s.step4();
  s.step5();

  s.b.resize(s.k + 1);
  return s.b;
}

}  // namespace cascade
