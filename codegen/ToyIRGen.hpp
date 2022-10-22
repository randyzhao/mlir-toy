#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "../lib/ast.hpp"
#include "Toy/ToyDialect.h"

class ToyIRGen: AST::Visitor {
public:
  void visit(AST::Module& module) override;
  void visit(AST::Function& function) override;

  ToyIRGen(): builder(&context) {
    context.getOrLoadDialect<toy::ToyDialect>();
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  mlir::MLIRContext context;
};
