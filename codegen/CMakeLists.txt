add_library(codegen "")
target_sources(codegen
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/ToyIRGen.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/ToyIRGen.hpp
  )
target_include_directories(codegen
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    lib)
