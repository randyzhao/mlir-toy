#ifndef _TOY_OPS_TD
#define _TOY_OPS_TD

include "ToyDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
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
  DeclareOpInterfaceMethods<CallableOpInterface>,
  FunctionOpInterface, 
  IsolatedFromAbove, 
  Symbol
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
  let hasCustomAssemblyFormat = 1;
}

def GenericDispatchOp: ToyOp<"generic_dispatch", [
    DeclareOpInterfaceMethods<CallOpInterface>
  ]>
{
  let summary = "generic dispatch operation";
  let description = [{
    Dispatch operation
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$callee, "llvm::ArrayRef<mlir::Value>":$arguments)>
  ];
}

def TransposeOp: ToyOp<"transpose", [NoSideEffect]> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$input)>
  ];

  let hasCanonicalizer = 1;
}

def ReturnOp: ToyOp<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";

  let arguments = (ins Variadic<F64Tensor>:$input);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}

def ReshapeOp : ToyOp<"reshape"> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
}

def AddOp: ToyOp<"add", [NoSideEffect]> {
  let summary = "element-wise add";
  let description = [{
    Element-wise add
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def MulOp: ToyOp<"add", [NoSideEffect]> {
  let summary = "element-wise mul";
  let description = [{
    Element-wise mul
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

// def CastOp : ToyOp<"cast", [
//      DeclareOpInterfaceMethods<CastOpInterface>,
//      DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//      NoSideEffect,
//      SameOperandsAndResultShape
//   ]> {
//   let summary = "shape cast operation";
//   let description = [{
//     The "cast" operation converts a tensor from one type to an equivalent type
//     without changing any data elements. The source and destination types must
//     both be tensor types with the same element type. If both are ranked, then
//     shape is required to match. The operation is invalid if converting to a
//     mismatching constant dimension.
//   }];

//   let arguments = (ins F64Tensor:$input);
//   let results = (outs F64Tensor:$output);

//   let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
// }

#endif
