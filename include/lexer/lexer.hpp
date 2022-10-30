#pragma once

#include <istream>
#include "commons.hpp"


class Lexer {
public:
  Lexer(std::istream& is): hasError(false), curChar(' '), is(is) {}

  Token getNextToken(SemanticValue& sval);
private:
  bool hasError;
  char curChar;
  std::istream& is;

  std::string scanIdentifierOrKeyword();
  float scanFloat();
  void handleLineComment();

  int cur_line_number = 0;
  int cur_col = 0;
};
