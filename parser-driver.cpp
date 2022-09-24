#include <fstream>
#include <iostream>

#include "ast.hpp"
#include "parser.hpp"
#include "lexer.hpp"

int main(int argc, char **argv) {
  std::string filename = argc < 2 ? "test.toy" : argv[1];
  std::ifstream fin(filename, std::ifstream::in);

  Lexer lexer;
  Parser parser(lexer, fin);

  auto module = parser.parse();
  if (module) std::cout << "Parse OK" << std::endl;
  else std::cout << "Parse error" << std::endl;

  return 0;
}
