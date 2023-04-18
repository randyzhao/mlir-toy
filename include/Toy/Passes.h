#ifndef _PASSES_H_
#define _PASSES_H_

#include <memory>
#include "mlir/Pass/Pass.h"

namespace toy {

std::unique_ptr<mlir::Pass> createLowerToAffinePass();

}

#endif
