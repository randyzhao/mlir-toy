lexer: commons.hpp lexer.hpp lexer.cpp

lexer-driver: lexer lexer-driver.cpp
	g++ -O0 -g lexer-driver.cpp lexer.cpp -o lexer

parser: lexer parser.cpp parser.hpp

parser-driver: parser parser-driver.cpp
	g++ -std=c++11 -O0 -g parser.cpp lexer.cpp parser-driver.cpp -o parser
