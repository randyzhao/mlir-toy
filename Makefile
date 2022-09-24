lexer-driver: commons.hpp lexer.hpp lexer.cpp lexer-driver.cpp
	g++ --std=c++17 -O0 -g  -o lexer lexer-driver.cpp lexer.cpp

parser-driver: commons.hpp lexer.hpp lexer.cpp parser.hpp parser.cpp parser-driver.cpp
	g++ --std=c++17 -O0 -g parser.cpp lexer.cpp parser-driver.cpp -o parser
