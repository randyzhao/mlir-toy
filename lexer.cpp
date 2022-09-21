#include <istream>
#include <cctype>
#include <string>
#include <cassert>

#include "lexer.hpp"
#include "commons.hpp"

Token Lexer::getNextToken(SemanticValue& sval, std::istream& is) {
  while (is.good() && (curChar == ' ' || curChar == '\n' || curChar == '\t')) {
    is.get(curChar);
  };
  if (!is.good()) return Token::Eof;

  if (isalpha(curChar) || curChar == '_') { // identifier or keyword
    std::string alphaNum = scanIdentifierOrKeyword(is);
    if (alphaNum == "def") return Token::Def;
    if (alphaNum == "var") return Token::Var;

    // identifier
    sval.identifierValue = alphaNum;
    return Token::Identifier;
  } else if (isdigit(curChar)) { // float const
    sval.floatConstValue = scanFloat(is);
    return hasError ? Token::Error : Token::FloatConst;
  } else if (curChar == '#') {
    handleLineComment(is);
  } else { // single char
    sval.singleCharValue = curChar;
    is.get(curChar);
    return Token::SingleChar;
  }
}

std::string Lexer::scanIdentifierOrKeyword(std::istream& is) {
  std::string ret(1, curChar);
  is.get(curChar);
  while (isalnum(curChar) || curChar == '_') {
    ret += curChar;
    is.get(curChar);
  }
  return ret;
}

float Lexer::scanFloat(std::istream& is) {
  assert(isdigit(curChar));

  bool hasDot = false;
  std::string raw(1, curChar);
  is.get(curChar);
  while (isdigit(curChar) || curChar == '.') {
    if (curChar == '.') {
      if (hasDot) {
        hasError = true;
        return -1;
      }
      hasDot = true;
    }
    raw += curChar;
  }
  return std::stof(raw);
}

void Lexer::handleLineComment(std::istream& is) {
  is.get(curChar);
  while (is.good() && curChar != '\n') is.get(curChar);
}
