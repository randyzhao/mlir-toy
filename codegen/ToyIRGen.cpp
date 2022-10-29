#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

#include "ToyIRGen.hpp"

#include <iostream>

using namespace toy;

void ToyIRGen::visit(AST::Module& module) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (auto &func : module.functions) func->accept(*this);
}

void ToyIRGen::visit(AST::Function& function) {
  llvm::SmallVector<mlir::Type, 4> argTypes(
    function.formals.size(),
    builder.getF64Type()
  );
  
  builder.setInsertionPointToEnd(theModule.getBody());

  auto funcType = builder.getFunctionType(argTypes, llvm::None);
  builder.create<toy::FuncOp>(
    builder.getUnknownLoc(),
    function.name,
    funcType
  );
}
