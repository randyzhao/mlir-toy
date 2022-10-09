

#include "ToyIRGen.hpp"

void ToyIRGen::visit(AST::Module& module) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (auto &func : module.functions) func->accept(*this);
}

void ToyIRGen::visit(AST::Function& function) {

}
