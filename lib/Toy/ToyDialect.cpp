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

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}
