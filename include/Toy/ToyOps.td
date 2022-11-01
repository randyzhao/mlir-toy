#ifndef _TOY_OPS_TD
#define _TOY_OPS_TD

include "ToyDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def ConstantOp: ToyOp<"constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Constant op turns a literal into a SSA value.
  }];

  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "::mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "double":$value)>
  ];

  let hasVerifier = 1;
}

def FuncOp: ToyOp<"func", [
  FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
  let summary = "user defined function op";
  let description = "user defined function op";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "::llvm::StringRef":$name, 
    "::mlir::FunctionType":$type, 
    CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let skipDefaultBuilders = 1;
}

def GenericDispatchOp: ToyOp<"generic_dispatch"> {
  let summary = "generic dispatch operation";
  let description = [{
    Dispatch operation
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F32Tensor>:$inputs);

  let results = (outs F32Tensor);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$callee, "llvm::ArrayRef<mlir::Value>":$arguments)>
  ];
}

def TransposeOp: ToyOp<"transpose"> {
  let summary = "transpose operation";

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$input)>
  ];
}

#endif
