#include <memory>

#include "parser/parser.hpp"

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
  ret->formals = parseFormalsOrArgs();
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != ')') {
    return nullptr;
  }
  consume();
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '{') {
    return nullptr;
  }
  consume();

  while (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '}') {
    auto expr = parseExpression();
    if (expr) ret->expressions.push_back(std::move(expr));
    consume(); // ;
  }

  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '}') {
    return nullptr;
  }
  consume();
  return ret;
}

vector<string> Parser::parseFormalsOrArgs() {
  vector<string> ret;
  while (getCurrentTok() == Token::Identifier) {
    ret.push_back(sval.identifierValue);
    consume(); // consume identifier
    // TODO: Handle syntax error

    if (sval.singleCharValue == ',') {
      consume();
    } else {
      break;
    }
  }
  return ret;
}

vector<unique_ptr<AST::Expression> > Parser::parseExpressions() {
  vector<unique_ptr<AST::Expression> > ret;
  while (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '}') {
    auto expr = parseExpression();
    if (expr) ret.push_back(std::move(expr));
    // TODO: Handle error
  }
  return ret;
}

unique_ptr<AST::Expression> Parser::parseExpression() {
  if (getCurrentTok() == Token::Return) {
    consume();
    return std::make_unique<AST::ReturnExpression>(parseExpression());
  } else if (getCurrentTok() == Token::Var) {
    return parseVarDeclExpression();
  } else if (getCurrentTok() == Token::SingleChar && sval.singleCharValue == '[') {
    return parseNestedListExpression();
  } else if (getCurrentTok() == Token::Identifier &&
             getNextTok() == Token::SingleChar &&
             nextSVal.singleCharValue == '(') {
    return parseDispatchExpression();
  }

  return nullptr;
}

unique_ptr<AST::VarDeclExpression> Parser::parseVarDeclExpression() {
  // TODO: Handle error

  consume(); // var
  string name = sval.identifierValue;
  consume(); // name
  vector<int> shape;
  if (getCurrentTok() == Token::SingleChar && sval.singleCharValue == '<') {
    consume(); // <
    shape = std::move(parseIntegerList());
    consume(); // >
  }
  unique_ptr<AST::Expression> init = nullptr;
  if (getCurrentTok() == Token::SingleChar && sval.singleCharValue == '=') {
    consume(); // =
    init = std::move(parseExpression());
  }
  return std::make_unique<AST::VarDeclExpression>(name, shape, std::move(init));
}

unique_ptr<AST::NestedListExpression> Parser::parseNestedListExpression() {
  return std::make_unique<AST::NestedListExpression>(parseNestedList());
}

unique_ptr<AST::DispatchExpression> Parser::parseDispatchExpression() {
  string name = std::move(sval.identifierValue);
  consume(); // identifier
  consume(); // (
  vector<string> args = parseFormalsOrArgs();
  consume(); // )
  return std::make_unique<AST::DispatchExpression>(name, args);
}

vector<int> Parser::parseIntegerList() {
  vector<int> ret;
  while (getCurrentTok() == Token::FloatConst) {
    ret.push_back(int(sval.floatConstValue));
    consume();

    if (getCurrentTok() == Token::SingleChar && sval.singleCharValue == ',') {
      consume();
    } else {
      break;
    }
  }
  return ret;
}

vector<float> Parser::parseFloatList() {
  vector<float> ret;
  while (getCurrentTok() == Token::FloatConst) {
    ret.push_back(sval.floatConstValue);
    consume();

    if (getCurrentTok() == Token::SingleChar && sval.singleCharValue == ',') {
      consume();
    } else {
      break;
    }
  }
  return ret;
}

unique_ptr<AST::NestedList> Parser::parseNestedList() {
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '[') {
    return nullptr;
  }

  unique_ptr<AST::NestedList> ret = nullptr;

  consume(); // [
  if (getCurrentTok() == Token::SingleChar) {
    ret = std::make_unique<AST::NestedList>(*parseNestedList());
  } else {
    ret = std::make_unique<AST::NestedList>(parseFloatList());
  }
  consume(); // ]
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
