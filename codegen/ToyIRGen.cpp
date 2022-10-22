#include "ToyIRGen.hpp"

void ToyIRGen::visit(AST::Module& module) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (auto &func : module.functions) func->accept(*this);
}

void ToyIRGen::visit(AST::Function& function) {
  llvm::SmallVector<mlir::Type, 4> argTypes(
    function.formals.size(),
    builder.getF64Type()
  );
  auto funcType = builder.getFunctionType(argTypes, llvm::None);
  return builder.create<mlir::toy::FuncOp>(
    builder.getUnknownLoc(),
    function.name,
    funcType
  );
}
