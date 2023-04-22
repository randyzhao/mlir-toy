#include <memory>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "Toy/Passes.h"
#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

namespace {

struct FuncOpLowering: public mlir::OpRewritePattern<toy::FuncOp> {
  using OpRewritePattern<toy::FuncOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(toy::FuncOp op, mlir::PatternRewriter &rewriter) const override {
    // Create a new mlir::FuncOp with the same name and type.
    auto funcOp = mlir::func::FuncOp::create(op.getLoc(), op.getName(), op.getFunctionType());
    
    // Clone the body region of the original FuncOp to the new FuncOp.
    funcOp.getBody().takeBody(op.getBody());

    // Replace the original FuncOp with the new FuncOp.
    rewriter.replaceOp(op, funcOp.getOperation()->getResults());

    return mlir::success(); 
  }
};

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

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpLowering>(&getContext());

    // if (failed(
    //       applyPartialConversion(getOperation(), target, std::move(patterns))))
    // signalPassFailure();

    // TODO:
  }
};

}

std::unique_ptr<mlir::Pass> toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
