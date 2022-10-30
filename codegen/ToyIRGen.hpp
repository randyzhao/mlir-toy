#pragma once

#include <vector>
#include <memory>

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "../lib/ast.hpp"
#include "Toy/ToyDialect.h"

using std::vector;
using std::unique_ptr;

class ToyIRGen: AST::Visitor {
public:
  void visit(AST::Module& module) override;
  void visit(AST::Function& function) override;
  void visit(AST::VarDeclExpression& expr) override;

  ToyIRGen(): builder(&context) {
    context.getOrLoadDialect<toy::ToyDialect>();
  }

  void dump() { theModule.dump(); }
private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  mlir::MLIRContext context;

  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symTab;

  void codeGenFunctionBody(vector<unique_ptr<AST::Expression> >& expressions);

  mlir::Value exprVal;
};
