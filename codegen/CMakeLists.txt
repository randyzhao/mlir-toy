add_mlir_library(codegen 
  ToyIRGen.cpp
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRToy
  MLIROptLib
  )

target_link_libraries(codegen PRIVATE ${LIBS})

mlir_check_all_link_libraries(codegen)
