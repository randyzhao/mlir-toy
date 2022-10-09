#pragma once

#include "../lib/ast.hpp"

class ToyIRGen: AST::Visitor {
  void visit(AST::Module& module) override;
};
