lexer: commons.hpp lexer.hpp lexer.cpp

lexer-driver: lexer lexer-driver.cpp
	g++ -O0 -g lexer-driver.cpp lexer.cpp -o lexer
