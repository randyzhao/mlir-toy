#include <memory>

#include "parser.hpp"

using std::unique_ptr;

unique_ptr<AST::Module> Parser::parse() {
  unique_ptr<AST::Module> ret = nullptr;
  getCurrentTok();
  if (curTok == Token::Error || curTok == Token::Eof) {
    errorMessage = "no input";
    return ret;
  }

  auto module = std::make_unique<AST::Module>();
  while (getCurrentTok() != Token::Eof) {
    auto func = parseFunction();
    if (func) {
      module->functions.push_back(std::move(func));
    } else {
      return ret;
    }
  }
  return module;
}

unique_ptr<AST::Function> Parser::parseFunction() {
  auto ret = std::make_unique<AST::Function>();
  if (getCurrentTok() != Token::Def) return nullptr;
  consume();
  if (getCurrentTok() != Token::Identifier) return nullptr;
  ret->name = sval.identifierValue;
  consume();
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '(') {
    return nullptr;
  }
  consume();
  // TODO: Parse params
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != ')') {
    return nullptr;
  }
  consume();
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '{') {
    return nullptr;
  }
  consume();
  // TODO: Parse statements
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '}') {
    return nullptr;
  }
  consume();
  return ret;
}

Token Parser::getCurrentTok() {
  if (curTok == Token::Nil) {
    curTok = lexer.getNextToken(sval, is);
  }
  return curTok;
}

Token Parser::getNextTok() {
  if (nextTok == Token::Nil) {
    nextTok = lexer.getNextToken(nextSVal, is);
  }
  return nextTok;
}

void Parser::consume() {
  getNextTok();
  curTok = nextTok;
  sval = nextSVal;

  nextTok = Token::Nil;
  getNextTok();
}
