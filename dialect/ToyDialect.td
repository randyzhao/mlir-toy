#ifndef _TOY_DIALECT
#define _TOY_DIALECT

include "mlir/IR/OpBase.td"

def ToyDialect: Dialect {
  let name = "toy";
  let summary = "A toy MLIR dialect";
  let description = [{
    This dialect is a minimal example.
  }];

  let cppNamespace = "::toy";
}

class ToyOp<string mnemonic, list<Trait> traits = []>:
  Op<ToyDialect, mnemonic, traits>;

#endif
