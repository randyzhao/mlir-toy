#pragma once

#include <vector>
#include <memory>
#include <string>
#include <ostream>

using std::vector;
using std::unique_ptr;
using std::string;

namespace AST {

struct ASTNode;
struct Module;
struct Function;
struct Statement;
struct ReturnStatement;
struct Expression;
struct MulExpression;
struct Error;

struct Visitor {
  virtual void visit(Module& module) {}
  virtual void visit(Function& function) {}
  virtual void visit(ReturnStatement& stat) {}
  virtual void visit(MulExpression& expr) {}
  virtual void visit(Error& err) {}
};

struct ASTNode {
  virtual void accept(Visitor& visitor) = 0;
  virtual ~ASTNode() {}
};

struct Error: ASTNode {
  string errorMsg;
  void accept(Visitor& visitor) { visitor.visit(*this); }

  Error(string errorMsg): errorMsg(errorMsg) {}
};

struct Module: ASTNode {
  vector<unique_ptr<Function> > functions;
  void accept(Visitor& visitor) { visitor.visit(*this); }
};

struct Function: ASTNode {
  string name;
  vector<string> formals;
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

class ASTDumper: Visitor {
public:
  ASTDumper(std::ostream& os): os(os) {}

  void visit(Module& module) {
    curLevel++;

    os << pad() << "Module:" << std::endl;

    for (auto& func : module.functions) {
      func->accept(*this);
    }

    curLevel--;
  }

  void visit(Function& function) {
    curLevel++;

    os << pad() << "Function" << std::endl;

    curLevel++;

    os << pad() << "Proto '" << function.name << "'" << std::endl;
    os << pad() << "Params: [" << printFormalsOrArgs(function.formals) << "]" << std::endl;


    os << pad() << "Block {" << std::endl;
    curLevel++;
    for (auto& stat : function.statements) {
      stat->accept(*this);
    }
    curLevel--;
    os << pad() << "}" << std::endl;

    curLevel--;

    curLevel--;
  }

  void visit(ReturnStatement& stat) {
    curLevel++;

    os << pad() << "Return" << std::endl;

    curLevel--;
  }
  void visit(MulExpression& expr) {
    curLevel++;

    os << pad() << "Mul" << std::endl;

    curLevel--;
  }
  void visit(Error& err) {
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
    for (int i = 0; i < formalsOrArgs.size(); i++) {
      ret += formalsOrArgs[i];
      if (i != formalsOrArgs.size() - 1) {
        ret += ", ";
      }
    }
    return ret;
  }
};

}
