cmake_minimum_required(VERSION 3.13.4)

project(mlir-toy LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(lexer)
add_subdirectory(parser)
add_subdirectory(lib)

add_executable(lexer-driver
  lexer-driver.cpp)

target_link_libraries(lexer-driver lexer)

add_executable(parser-driver parser-driver.cpp)
target_link_libraries(parser-driver parser)


