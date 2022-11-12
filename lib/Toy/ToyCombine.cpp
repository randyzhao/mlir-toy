#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

using namespace mlir;
using namespace toy;

namespace {
  struct SimplifyRedundantTranspose: public mlir::OpRewritePattern<TransposeOp> {
    SimplifyRedundantTranspose(mlir::MLIRContext *context): OpRewritePattern<TransposeOp>(context, 1) {}

    mlir::LogicalResult matchAndRewrite(TransposeOp op, mlir::PatternRewriter &rewriter) const override {
      mlir::Value transposeInput = op.getOperand();
      TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

      if (!transposeInputOp) return failure();

      rewriter.replaceOp(op, {transposeInputOp.getOperand()});
      return success();
    }
  };
}

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet & results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet & results, MLIRContext *context) {
  // TODO:
}
