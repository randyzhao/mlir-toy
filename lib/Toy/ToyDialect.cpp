#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

#include "Toy/ToyOpsDialect.cpp.inc"

using namespace mlir;
using namespace toy;

struct ToyInlinerInterface: public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Operation *, bool) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, BlockAndValueMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, BlockAndValueMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const final {
    auto returnOp = cast<ReturnOp>(op);

    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it: llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }

  // Operation *materializeCallConversion(OpBuilder &builder, Value input, Type resultType, Location conversionLoc) const final {
  //   return builder.create<CastOp>(conversionLoc, resultType, input);
  // }
};

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
  >();

  addInterfaces<ToyInlinerInterface>();
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(p, *this, false);
}

mlir::Region *FuncOp::getCallableRegion() { return &getBody(); }

llvm::ArrayRef<mlir::Type> FuncOp::getCallableResults() {
  return getFunctionType().getResults();
}

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
mlir::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = getValue().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

// mlir::LogicalResult TransposeOp::verify() {
//   auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
//   auto resultType = getType().dyn_cast<RankedTensorType>();
//   if (!inputType || !resultType)
//     return mlir::success();

//   auto inputShape = inputType.getShape();
//   if (!std::equal(inputShape.begin(), inputShape.end(),
//                   resultType.getShape().rbegin())) {
//     return emitError()
//            << "expected result shape to be a transpose of the input";
//   }
//   return mlir::success();
// }

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void GenericDispatchOp::build(
  mlir::OpBuilder &builder, 
  mlir::OperationState &state,
  StringRef callee, llvm::ArrayRef<mlir::Value> arguments)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

CallInterfaceCallable GenericDispatchOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

Operation::operand_range GenericDispatchOp::getArgOperands() { return getInputs(); }
