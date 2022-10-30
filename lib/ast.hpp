#pragma once

#include <vector>
#include <memory>
#include <string>
#include <ostream>
#include <variant>

using std::vector;
using std::unique_ptr;
using std::string;

namespace AST {

struct ASTNode;
struct Module;
struct Function;
struct ReturnExpression;
struct Expression;
struct MulExpression;
struct DispatchExpression;
struct NestedListExpression;
struct VarDeclExpression;
struct DispatchExpression;
struct Error;

struct NestedList;

struct Visitor {
  virtual void visit(Module& module) {}
  virtual void visit(Function& function) {}
  virtual void visit(ReturnExpression& expr) {}
  virtual void visit(MulExpression& expr) {}
  virtual void visit(VarDeclExpression& expr) {}
  virtual void visit(DispatchExpression& expr) {}
  virtual void visit(NestedListExpression& expr) {}
  virtual void visit(Error& err) {}
};

struct NestedList {
  using ListOfNestedLists = std::vector<NestedList>;
  using FloatList = std::vector<float>;
  using ListElement = std::variant<ListOfNestedLists, FloatList>;

  ListElement element;

  explicit NestedList(ListElement element): element(std::move(element)) {}
};

struct ASTNode {
  virtual void accept(Visitor& visitor) = 0;
  virtual ~ASTNode() {}
};

struct Error: ASTNode {
  string errorMsg;
  void accept(Visitor& visitor) override { visitor.visit(*this); }

  Error(string errorMsg): errorMsg(errorMsg) {}
};

struct Module: ASTNode {
  vector<unique_ptr<Function> > functions;
  void accept(Visitor& visitor) override { visitor.visit(*this); }
};

struct Function: ASTNode {
  string name;
  vector<string> formals;
  vector<unique_ptr<Expression> > expressions;

  void accept(Visitor& visitor) override { visitor.visit(*this); }
};

struct Expression: ASTNode {
  virtual void accept(Visitor& visitor) override = 0;
};

struct ReturnExpression: Expression {
  unique_ptr<Expression> expr;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  ReturnExpression(unique_ptr<Expression> expr): expr(std::move(expr)) {}
};

struct VarDeclExpression: Expression {
  string name;
  vector<int> shape;
  unique_ptr<Expression> init;
  void accept(Visitor& visitor) override { visitor.visit(*this); }

  VarDeclExpression(string name, vector<int>& shape, unique_ptr<Expression> init):
    name(name), shape(std::move(shape)), init(std::move(init)) {}
};

struct MulExpression: Expression {
  void accept(Visitor& visitor) override { visitor.visit(*this); }
};

struct DispatchExpression: Expression {
  string name;
  vector<string> args;

  void accept(Visitor& visitor) override { visitor.visit(*this); }

  DispatchExpression(string name, vector<string> args):
    name(name), args(std::move(args)) {}
};

struct NestedListExpression: Expression {
  unique_ptr<NestedList> nestedList;

  NestedListExpression(unique_ptr<NestedList> nestedList):
    nestedList(std::move(nestedList)) {}

  void accept(Visitor& visitor) override { visitor.visit(*this); }
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

    os << pad() << "Return" << std::endl;

    curLevel--;
  }
  void visit(MulExpression& expr) override {
    curLevel++;

    os << pad() << "Mul" << std::endl;

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
    for (auto& name : expr.args) {
      os << pad() << "var: " << name << std::endl;
    }
    curLevel--;
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
