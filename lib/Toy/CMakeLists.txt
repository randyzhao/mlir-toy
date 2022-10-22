add_mlir_dialect(ToyOps toy)

set(LLVM_TARGET_DEFINITIONS ToyOps.td)

add_mlir_dialect_library(MLIRToy
  ToyDialect.cpp
  ToyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Toy

  DEPENDS
  MLIRToyOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

