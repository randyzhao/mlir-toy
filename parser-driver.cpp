#include <fstream>
#include <iostream>

#include "ast.hpp"
#include "parser/parser.hpp"
#include "lexer/lexer.hpp"

int main(int argc, char **argv) {
  std::string filename = argc < 2 ? "test.toy" : argv[1];
  std::ifstream fin(filename, std::ifstream::in);

  Lexer lexer(fin, filename);
  Parser parser(lexer);

  auto module = parser.parse();
  if (module) {
    AST::ASTDumper dumper(std::cout);
    dumper.visit(*module);
  }
  else std::cout << "Parse error" << std::endl;

  return 0;
}
