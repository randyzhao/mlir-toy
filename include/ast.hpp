#pragma once

#include <vector>
#include <memory>
#include <string>
#include <ostream>
#include <iostream>
#include <variant>

#include "commons.hpp"

using std::vector;
using std::unique_ptr;
using std::string;

namespace AST {

struct ASTNode;
struct Module;
struct Function;
struct ReturnExpression;
struct Expression;
struct BinOpExpression;
struct DispatchExpression;
struct NestedListExpression;
struct VarDeclExpression;
struct DispatchExpression;
struct ObjectExpression;
struct Error;

struct NestedList;

struct Visitor {
  virtual void visit(Module& module) {}
  virtual void visit(Function& function) {}
  virtual void visit(ReturnExpression& expr) {}
  virtual void visit(BinOpExpression& expr) {}
  virtual void visit(VarDeclExpression& expr) {}
  virtual void visit(DispatchExpression& expr) {}
  virtual void visit(NestedListExpression& expr) {}
  virtual void visit(ObjectExpression& expr) {}
  virtual void visit(Error& err) {}
};

struct NestedList {
  using ListOfNestedLists = std::vector<NestedList>;
  using FloatList = std::vector<float>;
  using ListElement = std::variant<ListOfNestedLists, FloatList>;

  ListElement element;

  explicit NestedList(ListElement element): element(std::move(element)) {}

  void flattenTo(vector<float>& data) {
    if (std::holds_alternative<ListOfNestedLists>(element)) {
      for (auto& nestedList: std::get<ListOfNestedLists>(element)) {
        nestedList.flattenTo(data);
      }
    } else {
      for (float f: std::get<FloatList>(element)) {
        data.push_back(f);
      }
    }
  }

  void getShape(vector<int64_t>& shape) {
    if (std::holds_alternative<ListOfNestedLists>(element)) {
      auto& lists = std::get<ListOfNestedLists>(element);
      shape.push_back(lists.size());
      lists[0].getShape(shape);
    } else {
      auto& floats = std::get<FloatList>(element);
      shape.push_back(floats.size());
    }
  }
};



struct ASTNode {
  virtual void accept(Visitor& visitor) = 0;
  virtual ~ASTNode() {}

  Location loc;

  ASTNode(Location loc): loc(std::move(loc)) { }
};

struct Error: ASTNode {
  string errorMsg;
  void accept(Visitor& visitor) override { visitor.visit(*this); }

  Error(Location loc, string errorMsg): ASTNode(std::move(loc)), errorMsg(errorMsg) {}
};

struct Module: ASTNode {
  vector<unique_ptr<Function> > functions;
  void accept(Visitor& visitor) override { visitor.visit(*this); }

  Module(Location loc, vector<unique_ptr<Function> > functions):
    ASTNode(std::move(loc)), functions(std::move(functions)) { }
};

struct Function: ASTNode {
  string name;
  vector<string> formals;
  vector<unique_ptr<Expression> > expressions;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  Function(Location loc, string name, vector<string> formals, vector<unique_ptr<Expression> > expressions):
    ASTNode(std::move(loc)), name(std::move(name)), formals(std::move(formals)),
    expressions(std::move(expressions)) { }
};

struct Expression: ASTNode {
  virtual void accept(Visitor& visitor) override = 0;

  Expression(Location loc): ASTNode(std::move(loc)) { }
};

struct ReturnExpression: Expression {
  unique_ptr<Expression> expr;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  ReturnExpression(Location loc, unique_ptr<Expression> expr): 
    Expression(std::move(loc)),
    expr(std::move(expr)) {}
};

struct VarDeclExpression: Expression {
  string name;
  vector<int> shape;
  unique_ptr<Expression> init;
  void accept(Visitor& visitor) override { visitor.visit(*this); }

  VarDeclExpression(Location loc, string name, vector<int>& shape, unique_ptr<Expression> init):
    Expression(std::move(loc)),
    name(name), shape(std::move(shape)), init(std::move(init)) {}
};

struct BinOpExpression: Expression {
  char op;
  unique_ptr<Expression> lhs;
  unique_ptr<Expression> rhs;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  BinOpExpression(Location loc, char op, unique_ptr<Expression> lhs, unique_ptr<Expression> rhs):
    Expression(std::move(loc)),
    op(op),
    lhs(std::move(lhs)),
    rhs(std::move(rhs)) { }
};

struct DispatchExpression: Expression {
  string name;
  vector<unique_ptr<Expression>> args;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  DispatchExpression(Location loc, string name, vector<unique_ptr<Expression>> args):
    Expression(std::move(loc)),
    name(std::move(name)), args(std::move(args)) {}
};

struct ObjectExpression: Expression {
  string name;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  ObjectExpression(Location loc, string name):
    Expression(std::move(loc)),
    name(std::move(name)) { }
};

struct NestedListExpression: Expression {
  unique_ptr<NestedList> nestedList;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  NestedListExpression(Location loc, unique_ptr<NestedList> nestedList):
    Expression(std::move(loc)),
    nestedList(std::move(nestedList)) {}
};

class ASTDumper: Visitor {
public:
  ASTDumper(std::ostream& os): os(os) {}

  void visit(Module& module) override {
    curLevel++;

    os << pad() << "Module:" << std::endl;

    for (auto& func : module.functions) {
      func->accept(*this);
    }

    curLevel--;
  }

  void visit(Function& function) override {
    curLevel++;

    os << pad() << "Function" << std::endl;

    curLevel++;

    os << pad() << "Proto '" << function.name << "'" << std::endl;
    os << pad() << "Params: [" << printFormalsOrArgs(function.formals) << "]" << std::endl;


    os << pad() << "Block {" << std::endl;
    for (auto& expr : function.expressions) {
      expr->accept(*this);
    }
    os << pad() << "}" << std::endl;

    curLevel--;

    curLevel--;
  }

  void visit(ReturnExpression& expr) override {
    curLevel++;

    os << pad() << "Return " << std::endl;
    if (expr.expr) {
      expr.expr->accept(*this);
    }
    os << std::endl;

    curLevel--;
  }

  void visit(BinOpExpression& expr) override {
    curLevel++;

    os << pad() << "BinaryExpr: " << std::endl;

    curLevel++;
    os << pad() << "op: " << expr.op << std::endl;
    os << pad() << "lhs: " << std::endl;
    expr.lhs->accept(*this);
    os << pad() << "rhs: " << std::endl;
    expr.rhs->accept(*this);
    curLevel--;

    curLevel--;
  }

  void visit(VarDeclExpression& expr) override {
    curLevel++;

    os << pad() << "VarDecl " << expr.name << "<";
    for (auto& n : expr.shape) {
      os << n << ",";
    }
    os << ">" << std::endl;
    if (expr.init) expr.init->accept(*this);

    curLevel--;
  }

  void visit(NestedListExpression& expr) override {
    curLevel++;
    printNestedList(*expr.nestedList);
    curLevel--;
  }

  void visit(DispatchExpression& expr) override {
    curLevel++;
    os << pad() << "Call '" << expr.name << "'" << std::endl;
    curLevel++;
    for (auto& arg : expr.args) {
      os << pad() << "var: ";
      arg->accept(*this);
      os << std::endl;
    }
    curLevel--;
    curLevel--;
  }

  void visit(ObjectExpression& expr) override {
    curLevel++;
    os << pad() << "object: " << expr.name << "'" << std::endl;
    curLevel--;
  }

  void visit(Error& err) override {
    curLevel++;

    os << pad() << "Error" << std::endl;

    curLevel--;
  }

private:
  int curLevel = -1;
  std::ostream& os;

  string pad() {
    string ret(curLevel * 2, ' ');
    return ret;
  }

  string printFormalsOrArgs(vector<string>& formalsOrArgs) {
    string ret;
    for (size_t i = 0; i < formalsOrArgs.size(); i++) {
      ret += formalsOrArgs[i];
      if (i != formalsOrArgs.size() - 1) {
        ret += ", ";
      }
    }
    return ret;
  }

  void printNestedList(NestedList& nl) {
    os << pad() << "Literal: [";
    if (std::holds_alternative<std::vector<float>>(nl.element)) {
      std::vector<float>& nums = std::get<std::vector<float>>(nl.element);
      for (size_t i = 0; i < nums.size(); i++) {
        os << nums[i];
        if (i != nums.size() - 1) os << ",";
      }
    } else {
      std::vector<NestedList> lists = std::get<std::vector<NestedList>>(nl.element);
      for (size_t i = 0; i < lists.size(); i++) {
        printNestedList(lists[i]);
        if (i != lists.size() - 1) os << ",";
      }
    }
    os << "]" << std::endl;
  }
};

}
