set(LLVM_TARGET_DEFINITIONS ToyOps.td)

add_mlir_dialect_library(MLIRToy
  ToyDialect.cpp
  ToyOps.cpp
  ToyCombine.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Toy

  DEPENDS
  MLIRToyOpsIncGen

  LINK_LIBS PRIVATE
  MLIRIR
)

