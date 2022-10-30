#pragma once

#include <string>

enum Token {
  Nil = 100,
  Identifier = 1010,
  Return = 1020,
  FloatConst = 1030,
  Def = 1040,
  Var = 1050,
  SingleChar = 1060,
  Eof = 3000,
  Error = 4000,
};

struct SemanticValue {
  std::string identifierValue;
  float floatConstValue;
  char singleCharValue;
};


