add_library(parser "")
target_sources(parser
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/parser.cpp
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include/parser/parser.hpp
  )

target_link_libraries(parser lexer)
