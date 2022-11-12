#include "Toy/ToyOps.h"
#include "Toy/ToyDialect.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"
