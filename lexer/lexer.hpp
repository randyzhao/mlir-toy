#pragma once

#include <istream>
#include "../lib/commons.hpp"


class Lexer {
public:
  Lexer(): hasError(false), curChar(' ') {}

  Token getNextToken(SemanticValue& sval, std::istream& is);
private:
  char curChar;
  bool hasError;
  std::string scanIdentifierOrKeyword(std::istream& is);
  float scanFloat(std::istream& is);
  void handleLineComment(std::istream& is);
};
