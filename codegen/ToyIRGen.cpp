#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

#include "codegen/ToyIRGen.hpp"

#include <iostream>

using namespace toy;

namespace {
  mlir::Type getTypeFromShape(mlir::OpBuilder& builder, llvm::ArrayRef<int64_t> shape) {
    if (shape.empty()) {
      return mlir::UnrankedTensorType::get(builder.getF32Type());
    }

    return mlir::RankedTensorType::get(shape, builder.getF32Type());
  }

  mlir::Location toMLIRLocaction(mlir::OpBuilder& builder, const Location& loc) {
    return mlir::FileLineColLoc::get(
      builder.getStringAttr(*loc.filename), 
      loc.lineNumber, 
      loc.col
    );
  }
}

void ToyIRGen::visit(AST::Module& module) {
  theModule = mlir::ModuleOp::create(toMLIRLocaction(builder, module.loc));

  for (auto &func : module.functions) func->accept(*this);
}

void ToyIRGen::visit(AST::Function& function) {
  llvm::SmallVector<mlir::Type, 4> argTypes(
    function.formals.size(),
    builder.getF32Type()
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

void ToyIRGen::visit(AST::NestedListExpression& expr) {
  std::vector<float> data;
  expr.nestedList->flattenTo(data);

  mlir::Type elementType = builder.getF32Type();
  vector<int64_t> shape;
  expr.nestedList->getShape(shape);

  llvm::ArrayRef<int64_t> dims(shape);

  auto dataType = mlir::RankedTensorType::get(dims, elementType);

  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

  exprVal = builder.create<toy::ConstantOp>(
    builder.getUnknownLoc(),
    getTypeFromShape(builder, shape),
    dataAttribute
  );
}
