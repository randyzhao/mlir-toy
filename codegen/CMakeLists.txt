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

#target_sources(codegen
#  PRIVATE
#    ${CMAKE_CURRENT_LIST_DIR}/ToyIRGen.cpp
#  PUBLIC
#    ${CMAKE_CURRENT_LIST_DIR}/ToyIRGen.hpp
#  )

# target_include_directories(codegen
#   PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     lib
#   INTERFACE
#     ${CMAKE_CURRENT_LIST_DIR}/ToyIRGen.hpp
#   PRIVATE
#     ${PROJECT_SOURCE_DIR}/include/Toy
#   )

target_link_libraries(codegen PRIVATE ${LIBS})

mlir_check_all_link_libraries(codegen)
