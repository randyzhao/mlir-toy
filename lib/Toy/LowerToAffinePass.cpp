#include <memory>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Toy/Passes.h"
#include "Toy/ToyDialect.h"

namespace {

struct ToyToAffineLoweringPass: public mlir::PassWrapper<ToyToAffineLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final {
      mlir::ConversionTarget target(mlir::Pass::getContext());

    target.addLegalDialect<mlir::AffineDialect, mlir::BuiltinDialect, mlir::arith::ArithmeticDialect,
      mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
    
    target.addIllegalDialect<toy::ToyDialect>();

    // TODO:
  }
};

}

std::unique_ptr<mlir::Pass> toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
