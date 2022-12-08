#include <istream>
#include <cctype>
#include <string>
#include <cassert>

#include "lexer/lexer.hpp"
#include "commons.hpp"


Token Lexer::getNextToken(SemanticValue& sval) {
  while (is.good() && (curChar == ' ' || curChar == '\n')) {
    if (curChar == '\n') loc.lineNumber++;
    getNextChar();
  };
  if (!is.good()) return Token::Eof;

  if (isalpha(curChar) || curChar == '_') { // identifier or keyword
    std::string alphaNum = scanIdentifierOrKeyword();
    if (alphaNum == "def") return Token::Def;
    if (alphaNum == "var") return Token::Var;
    if (alphaNum == "return") return Token::Return;

    // identifier
    sval.identifierValue = alphaNum;
    return Token::Identifier;
  } else if (isdigit(curChar)) { // float const
    sval.floatConstValue = scanFloat();
    return hasError ? Token::Error : Token::FloatConst;
  } else if (curChar == '#') {
    handleLineComment();
  } else { // single char
    sval.singleCharValue = curChar;
    getNextChar();
    return Token::SingleChar;
  }

  return Token::Error;
}

std::string Lexer::scanIdentifierOrKeyword() {
  std::string ret(1, curChar);
  getNextChar();
  while (isalnum(curChar) || curChar == '_') {
    ret += curChar;
    getNextChar();
  }
  return ret;
}

float Lexer::scanFloat() {
  assert(isdigit(curChar));

  bool hasDot = false;
  std::string raw(1, curChar);
  getNextChar();
  while (isdigit(curChar) || curChar == '.') {
    if (curChar == '.') {
      if (hasDot) {
        hasError = true;
        return -1;
      }
      hasDot = true;
    }
    raw += curChar;
    getNextChar();
  }
  return std::stof(raw);
}

void Lexer::handleLineComment() {
  getNextChar();
  while (is.good() && curChar != '\n') getNextChar();
}

void Lexer::getNextChar() {
  loc.col++;
  is.get(curChar);
}
