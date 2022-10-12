#include <iostream>
#include <fstream>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "ToyIRGen.hpp"
#include "lexer.hpp"
#include "parser.hpp"

int main(int argc, char **argv) {
  std::string filename = argc < 2 ? "test.toy" : argv[1];
  std::ifstream fin(filename, std::ifstream::in);

  Lexer lexer;
  Parser parser(lexer, fin);

  auto module = parser.parse();
  if (module) {
    std::cout << "Parse succeeded" << std::endl;
  } else {
    std::cout << "Parse error" << std::endl;
    return 1;
  }

  ToyIRGen gen;
  gen.visit(*module);

  return 0;
}
