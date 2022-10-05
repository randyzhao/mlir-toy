add_library(lexer "")
target_sources(lexer
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/lexer.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/lexer.hpp
  )
target_include_directories(lexer
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    lib)

