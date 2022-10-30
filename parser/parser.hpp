#pragma once

#include <memory>
#include <istream>

#include "lexer.hpp"
#include "../lib/ast.hpp"
#include "../lib/commons.hpp"

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
  vector<string> parseFormalsOrArgs();
  vector<unique_ptr<AST::Expression> > parseExpressions();
  unique_ptr<AST::Expression> parseExpression();
  unique_ptr<AST::VarDeclExpression> parseVarDeclExpression();
  unique_ptr<AST::NestedListExpression> parseNestedListExpression();
  unique_ptr<AST::DispatchExpression> parseDispatchExpression();
  vector<int> parseIntegerList();
  vector<float> parseFloatList();
  unique_ptr<AST::NestedList> parseNestedList();


  Token curTok = Token::Nil;
  Token nextTok = Token::Nil;
  SemanticValue sval;
  SemanticValue nextSVal;

  string errorMessage;
};
