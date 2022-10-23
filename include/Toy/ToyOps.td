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

def TransposeOp: ToyOp<"transpose"> {
  let summary = "transpose op";
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input)>
  ];

  let hasVerifier = 1;
}

def FuncOp: ToyOp<"func", [
  FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
  let summary = "user defined function op";
  let description = "user defined function op";

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "::llvm::StringRef":$name, CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)
  >];
}

def CallOp: ToyOp<"call"> {
  let summary = "call op";
  let description = "call op";

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$input);

  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$callee, "::llvm::ArrayRef<::mlir::Value>":$arguments)>
  ];
}

#endif
