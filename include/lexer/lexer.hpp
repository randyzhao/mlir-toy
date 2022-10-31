#pragma once

#include <istream>
#include "commons.hpp"

class Lexer {
public:
  Lexer(std::istream& is, std::string filename): 
    hasError(false), 
    curChar(' '), 
    is(is),
    loc({std::make_shared<std::string>(std::move(filename), 0, 0)}) {}

  Token getNextToken(SemanticValue& sval);

  Location getLocation() { return loc; }

private:
  bool hasError;
  char curChar;
  std::istream& is;

  Location loc;

  std::string scanIdentifierOrKeyword();
  float scanFloat();
  void handleLineComment();

  void getNextChar();
};
