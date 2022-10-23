#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

#include "Toy/ToyOpsDialect.cpp.inc"

using namespace mlir;
using namespace toy;



void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
  >();
}
