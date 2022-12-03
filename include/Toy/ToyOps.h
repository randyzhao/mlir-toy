#ifndef _TOY_OPS_H
#define _TOY_OPS_H


#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "Toy/ToyOps.h.inc"

#endif
