add_library(parser "")
target_sources(parser
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/parser.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/parser.hpp
  )

target_link_libraries(parser lexer)

target_include_directories(lexer
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    lib)

