#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"

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
  toy::FuncOp funcOp = builder.create<toy::FuncOp>(
    builder.getUnknownLoc(),
    function.name,
    funcType
  );

  mlir::Block &entryBlock = funcOp.front();

  builder.setInsertionPointToStart(&entryBlock);

  codeGenFunctionBody(function.expressions);
}

void ToyIRGen::codeGenFunctionBody(vector<unique_ptr<AST::Expression> >& expressions) {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symTab);
  for (auto& expr: expressions) {
    expr->accept(*this);
  }
}

void ToyIRGen::visit(AST::VarDeclExpression& expr) {
  expr.init->accept(*this);
  mlir::Value initValue = exprVal;

  if (symTab.count(expr.name)) {
    mlir::emitError(builder.getUnknownLoc(), "var declared already");
    return;
  }

  symTab.insert(expr.name, initValue);  
}
