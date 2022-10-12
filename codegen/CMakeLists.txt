add_llvm_library(codegen ToyIRGen.cpp )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${dialect_libs}
  MLIROptLib
  )

#target_sources(codegen
#  PRIVATE
#    ${CMAKE_CURRENT_LIST_DIR}/ToyIRGen.cpp
#  PUBLIC
#    ${CMAKE_CURRENT_LIST_DIR}/ToyIRGen.hpp
#  )

target_include_directories(codegen
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    lib)

target_link_libraries(codegen PRIVATE ${LIBS})

mlir_check_all_link_libraries(codegen)
