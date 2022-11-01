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

  vector<unique_ptr<AST::Function> > functions;

  while (getCurrentTok() != Token::Eof) {
    auto func = parseFunction();
    if (func) {
      functions.push_back(std::move(func));
    } else {
      return ret;
    }
  }
  return std::make_unique<AST::Module>(lexer.getLocation(), std::move(functions));
}

unique_ptr<AST::Function> Parser::parseFunction() {
  string name;
  vector<string> formals;
  vector<unique_ptr<AST::Expression> > expressions;

  if (getCurrentTok() != Token::Def) return nullptr;
  consume();
  if (getCurrentTok() != Token::Identifier) return nullptr;
  name = sval.identifierValue;
  consume();
  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '(') {
    return nullptr;
  }
  consume();
  formals = parseFormals();
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
    if (expr) expressions.push_back(std::move(expr));
    consume(); // ;
  }

  if (getCurrentTok() != Token::SingleChar || sval.singleCharValue != '}') {
    return nullptr;
  }
  consume();
  return std::make_unique<AST::Function>(
    lexer.getLocation(), 
    name, 
    std::move(formals), 
    std::move(expressions)
  );
}

vector<string> Parser::parseFormals() {
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

vector<unique_ptr<AST::Expression> > Parser::parseArgs() {
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
    return std::make_unique<AST::ReturnExpression>(
      lexer.getLocation(),
      parseExpression()
    );
  } else if (getCurrentTok() == Token::Var) {
    return parseVarDeclExpression();
  } else if (getCurrentTok() == Token::SingleChar && sval.singleCharValue == '[') {
    return parseNestedListExpression();
  } else if (getCurrentTok() == Token::Identifier) {
    if (getNextTok() == Token::SingleChar &&
        nextSVal.singleCharValue == '(') {
      return parseDispatchExpression();
    } else {
      auto objExp = std::make_unique<AST::ObjectExpression>(
        lexer.getLocation(),
        std::move(sval.identifierValue)
      );
      consume(); // identifier
      return objExp;
    }
  }

  std::cout << "parse nullptr current token " << getCurrentTok() << std::endl;
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
  return std::make_unique<AST::VarDeclExpression>(lexer.getLocation(), name, shape, std::move(init));
}

unique_ptr<AST::NestedListExpression> Parser::parseNestedListExpression() {
  return std::make_unique<AST::NestedListExpression>(
    lexer.getLocation(),
    parseNestedList()
  );
}

unique_ptr<AST::DispatchExpression> Parser::parseDispatchExpression() {
  string name = std::move(sval.identifierValue);
  vector<unique_ptr<AST::Expression>> args;
  consume(); // identifier
  consume(); // (
  while (!isCurTokSingleChar(')')) {
    args.push_back(parseExpression());
    if (isCurTokSingleChar(',')) consume(); // ,
  }
  consume(); // )
  return std::make_unique<AST::DispatchExpression>(
    lexer.getLocation(),
    name, 
    std::move(args)
  );
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
    curTok = lexer.getNextToken(sval);
  }
  return curTok;
}

Token Parser::getNextTok() {
  if (nextTok == Token::Nil) {
    nextTok = lexer.getNextToken(nextSVal);
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
