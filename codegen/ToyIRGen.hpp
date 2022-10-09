#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "../lib/ast.hpp"

class ToyIRGen: AST::Visitor {
public:
  void visit(AST::Module& module) override;
  void visit(AST::Function& function) override;

  ToyIRGen(mlir::MLIRContext &context): builder(&context) {}

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
};
