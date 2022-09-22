#pragma once

#include <vector>
#include <memory>
#include <string>

using std::vector;
using std::unique_ptr;
using std::string;

struct ASTNode;
struct Module;
struct Function;
struct Statement;
struct ReturnStatement;
struct Expression;
struct MulExpression;

struct Visitor {
  virtual void visit(Module& module) {}
  virtual void visit(Function& function) {}
  virtual void visit(ReturnStatement& stat) {}
  virtual void visit(MulExpression& expr) {}
};

struct ASTNode {
  virtual void accept(Visitor& visitor) = 0;
};

struct Module: ASTNode {
  vector<unique_ptr<Function> > functions;
  void accept(Visitor& visitor) { visitor.visit(*this); }
};

struct Function: ASTNode {
  string name;
  vector<string> params;
  vector<unique_ptr<Statement> > statements;

  void accept(Visitor& visitor) { visitor.visit(*this); }
};

struct Statement: ASTNode {
  virtual void accept(Visitor& visitor) = 0;
};

struct ReturnStatement: Statement {
  unique_ptr<Expression> expr;

  void accept(Visitor& visitor) { visitor.visit(*this); }
};

struct Expression: ASTNode {
  virtual void accept(Visitor& visitor) = 0;
};

struct MulExpression: Expression {
  void accept(Visitor& visitor) { visitor.visit(*this); }
};



