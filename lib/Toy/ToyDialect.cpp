#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "ToyDialect.h"
#include "ToyOps.h"

#include "Toy/ToyOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::toy;



void ToyDialect::Initialize() {
  addOperation<
#define GET_OP_LIST
#include "dialect/ToyOps.cpp.inc"
  >();
}
