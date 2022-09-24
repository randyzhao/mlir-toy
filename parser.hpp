#pragma once

#include <memory>
#include <istream>

#include "lexer.hpp"
#include "ast.hpp"
#include "commons.hpp"

using std::unique_ptr;

class Parser {
public:
  Parser(Lexer& lexer, std::istream& is): lexer(lexer), is(is) {}
  unique_ptr<AST::Module> parse();

private:
  Lexer& lexer;
  std::istream& is;
  Token getCurrentTok();
  Token getNextTok();
  void consume();

  unique_ptr<AST::Function> parseFunction();
  vector<string> parseFormals();

  Token curTok = Token::Nil;
  Token nextTok = Token::Nil;
  SemanticValue sval;
  SemanticValue nextSVal;

  string errorMessage;
};
