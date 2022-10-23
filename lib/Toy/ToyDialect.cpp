#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

#include "Toy/ToyOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::toy;



void ToyDialect::Initialize() {
  addOperation<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
  >();
}
