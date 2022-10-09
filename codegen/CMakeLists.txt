add_library(codegen "")

set(LIBS
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIROptLib
  )

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

target_link_libraries(codegen PRIVATE ${LIBS})
