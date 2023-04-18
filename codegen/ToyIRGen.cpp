#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"
#include "Toy/Passes.h"

#include "codegen/ToyIRGen.hpp"

#include <iostream>

using namespace toy;

namespace {
  mlir::Type getTypeFromShape(mlir::OpBuilder& builder, llvm::ArrayRef<int64_t> shape) {
    if (shape.empty()) {
      return mlir::UnrankedTensorType::get(builder.getF64Type());
    }

    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  mlir::Location toMLIRLocaction(mlir::OpBuilder& builder, const Location& loc) {
    return mlir::FileLineColLoc::get(
      builder.getStringAttr(*loc.filename), 
      loc.lineNumber, 
      loc.col
    );
  }
}

ToyIRGen::ToyIRGen(): builder(&context) {
  context.getOrLoadDialect<toy::ToyDialect>();
}

void ToyIRGen::visit(AST::Module& module) {
  theModule = mlir::ModuleOp::create(toMLIRLocaction(builder, module.loc));
  for (auto &func : module.functions) func->accept(*this);

  mlir::PassManager pm(&context);
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<toy::FuncOp>(mlir::createCanonicalizerPass());

  pm.addPass(toy::createLowerToAffinePass());
  
  if (mlir::failed(pm.run(theModule))) { 
    // TODO:
  }
}

void ToyIRGen::visit(AST::Function& function) {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symTab);

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

  // TODO:
  // if (function.name != "main") funcOp.setPrivate();

  mlir::Block &entryBlock = funcOp.front();

   // declare all function arguments
  for (const auto nameValue: llvm::zip(function.formals, entryBlock.getArguments())) {
    symTab.insert(std::get<0>(nameValue), std::get<1>(nameValue));
  }

  builder.setInsertionPointToStart(&entryBlock);

  codeGenFunctionBody(function.expressions);
}

void ToyIRGen::codeGenFunctionBody(vector<unique_ptr<AST::Expression> >& expressions) {
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
  std::vector<double> data;
  expr.nestedList->flattenTo(data);

  mlir::Type elementType = builder.getF64Type();
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

void ToyIRGen::visit(AST::DispatchExpression& expr) {
  llvm::StringRef callee = expr.name;
  auto loc = toMLIRLocaction(builder, expr.loc);

  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto& expr: expr.args) {
    expr->accept(*this);
    operands.push_back(exprVal);
  }

  if (callee == "transpose") {
    exprVal = builder.create<toy::TransposeOp>(loc, operands[0]);
    return;
  }

  exprVal = builder.create<toy::GenericDispatchOp>(loc, callee, operands);
}

void ToyIRGen::visit(AST::ObjectExpression& expr) {
  // TODO: Test if the object has been declared.

  exprVal = symTab.lookup(expr.name);
}

void ToyIRGen::visit(AST::ReturnExpression& expr) {
  auto loc = toMLIRLocaction(builder, expr.loc);

  mlir::Value returnVal = nullptr;
  if (expr.expr) {
    expr.expr->accept(*this);
    returnVal = exprVal;
  }
  builder.create<toy::ReturnOp>(
    loc,
    returnVal ? llvm::makeArrayRef(returnVal) : llvm::ArrayRef<mlir::Value>()
  );
}
