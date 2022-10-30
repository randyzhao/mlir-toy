#include <iostream>
#include <fstream>
#include <string>

#include "lexer/lexer.hpp"
#include "commons.hpp"

int main(int argc, char **argv) {
  std::string filename = argc < 2 ? "test.toy" : argv[1];
  std::ifstream fin(filename, std::ifstream::in);
  Token tok;
  Lexer lexer(fin);
  SemanticValue sval;

  std::cout << "Start scanning" << std::endl;

  while ((tok = lexer.getNextToken(sval)) != Token::Eof) {
    switch (tok) {
    case Token::Def:
      std::cout << "def" << std::endl;
      break;
    case Token::Var:
      std::cout << "var" << std::endl;
      break;
    case Token::Identifier:
      std::cout << "Identifier: " << sval.identifierValue << std::endl;
      break;
    case Token::FloatConst:
      std::cout << "FloatConst: " << sval.floatConstValue << std::endl;
      break;
    case Token::SingleChar:
      std::cout << "SingleChar: " << sval.singleCharValue << std::endl;
      break;
    case Token::Error:
      std::cout << "error" << std::endl;
      break;
    default:
      std::cout << "Unexpected " << std::endl;
      break;
    }
  }

  return 0;
}
