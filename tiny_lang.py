#!/usr/bin/python3

# import stdlib

import string
from typing import *
from dataclasses import dataclass

@dataclass(frozen=True)
class FatalError(Exception):
  msg: str


TT_STRING   = "TT_STRING"
TT_NUMBER   = "TT_NUMBER"
TT_SYMBOL   = "TT_SYMBOL"
TT_ARROW    = "TT_ARROW"
TT_WALRUS   = "TT_WALRUS"
TT_LBRACKET = "TT_LBRACKET"
TT_RBRACKET = "TT_RBRACKET"
TT_EOF      = "TT_EOF"

TOKEN_SINGLE: Any = {
  "[": TT_LBRACKET,
  "]": TT_RBRACKET,
}

SYMBOL_PUNCTUATION: str = "+-*/%&|^~<>=!@?._"


@dataclass
class Token(object):
  typ: str
  val: Any = None

  def __repr__(self) -> str:
    postfix: str = f":{self.val}" if self.val else ""
    return f"{self.typ}{postfix}"


class Lexer:
  def __init__(self, source: str) -> None:
    self.source = source or " "

    self.current_index: int = 0
    self.record_index: int = 0

    self.current: str = self.source[self.current_index]

    TOKEN_TYPES: Any = {
      TT_STRING:   self.make_string,
      TT_NUMBER:   self.make_number,
      TT_SYMBOL:   self.make_symbol,
      TT_ARROW:    self.make_arrow,
      TT_WALRUS:   self.make_walrus,
      TT_LBRACKET: lambda: self.atom(TT_LBRACKET),  
      TT_RBRACKET: lambda: self.atom(TT_RBRACKET),  
      TT_EOF:      lambda: self.atom(TT_EOF),
    }

    self.make_token = lambda typ: TOKEN_TYPES[typ]()

  def atom(self, typ) -> Token:
    self.increment()
    return Token(typ)

  def increment(self) -> str:
    self.current_index += 1

    if self.current_index < len(self.source):
      self.current = self.source[self.current_index]
    else:
      self.current = ""

    return self.current

  def record(self) -> None:
    self.record_index = self.current_index

  def replay(self) -> None:
    self.current_index = self.record_index - 1
    self.increment()

  def next(self) -> Token:
    while self.current:
      while self.current and self.current in string.whitespace:
        self.increment()

      if self.current == "":
        break

      if self.current == "\"":
        return self.make_token(TT_STRING)
      elif self.current.isdigit() or self.current == ".":
        return self.make_token(TT_NUMBER)
      elif self.current == "=":
        return self.make_token(TT_ARROW)
      elif self.current == ":":
        return self.make_token(TT_WALRUS)
      elif self.current in TOKEN_SINGLE:
        return self.make_token(TOKEN_SINGLE[self.current])
      elif self.current.isalpha() or self.current in SYMBOL_PUNCTUATION:
        return self.make_token(TT_SYMBOL)
      else:
        raise FatalError(f"unkown token; `{self.current}`")

    return self.make_token(TT_EOF)

  def peek(self) -> Token:
    self.record()
    token: Token = self.next()
    self.replay()

    return token

  def make_string(self) -> Token:
    self.increment()
    buffer: str = ""

    while self.current != "\"":
      if self.current in ("\n", ""):
        raise FatalError("string-constant not terminated")
      buffer += self.current
      self.increment()

    self.increment()
    return Token(TT_STRING, buffer)

  def make_number(self) -> Token:
    buffer: str = ""
    
    while self.current.isdigit() or self.current in (".", "'"):
      buffer += self.current
      self.increment()

    if buffer.count(".") > 1:
      raise FatalError("malformed number; `%s`" % buffer)

    buffer = buffer.replace("'", "")
    return Token(TT_NUMBER, float(buffer) if "." in buffer else int(buffer))

  def make_symbol(self) -> Token:
    buffer: str = ""

    while ((self.current.isalpha() or self.current in SYMBOL_PUNCTUATION)
           and self.current != ""):
      buffer += self.current
      self.increment()

    return Token(TT_SYMBOL, buffer)

  def make_arrow(self) -> Token:
    self.increment()

    if self.current != ">":
      return Token(TT_SYMBOL, "=")

    self.increment ()
    return Token(TT_ARROW)

  def make_walrus(self) -> Token:
    self.increment()

    if self.current != "=":
      raise FatalError(f"malformed walrus-operator; `:{self.current}`")

    self.increment ()
    return Token(TT_WALRUS)



@dataclass(frozen=True)
class ProgramNode:
  nodes: List[Any]


@dataclass(frozen=True)
class FunctionDefinitionNode:
  args: List[Token]
  body: ProgramNode


@dataclass(frozen=True)
class FunctionInvocationNode:
  func: Any
  args: List[Any]


@dataclass(frozen=True)
class SymbolBindNode:
  name: Token
  val: Any


@dataclass(frozen=True)
class LiteralNode:
  val: Any


def print_node(node: Any, indent: int = 0, prefix = "") -> None:
  print((" " * indent) + prefix, end="")

  if isinstance(node, ProgramNode):
    print(f"{type(node).__name__}:")
    for child in node.nodes:
      print_node(child, indent + 2)
  
  elif isinstance(node, FunctionDefinitionNode):
    print(f"{type(node).__name__}:")

    for index, child in enumerate(node.args):
      print_node(child, indent + 2, prefix = f"{index} = ")
    print_node(node.body, indent + 2)

  elif isinstance(node, FunctionInvocationNode):
    print(f"{type(node).__name__}:")

    print_node(node.func, indent=indent + 2)

    for index, child in enumerate(node.args):
      print_node(child, indent + 2, prefix = f"{index} = ")

  elif isinstance(node, SymbolBindNode):
    print(f"{type(node).__name__}:")
    print_node(node.name, indent=indent + 2)
    print_node(node.val, indent=indent + 2)

  elif isinstance(node, LiteralNode):
    print(f"{node.val}")
  
  else:
    print(f"{node}")


class Parser:
  def __init__(self, lexer: Lexer) -> None:
    self.lexer: Lexer = lexer
    self.current: Token = self.increment()

  def increment(self) -> Token:
    self.current = lexer.next()
    return self.current

  def parse_program(self) -> Any:
    program: List[Any] = []
    while self.current.typ not in [TT_EOF, TT_RBRACKET]:
      program.append(self.parse_expression())

    return ProgramNode(program)

  def parse_expression(self) -> Any:
    if self.current.typ == TT_LBRACKET:
      return self.parse_function()
    return self.parse_atom()

  def parse_function(self) -> Any:
    self.lexer.record()
    self.increment()

    brackets: int = 1
    
    """
      0: function-invocation
      1: function-definition
      2: symbol-bind
    """
    typ: int = 0

    while self.current.typ != TT_EOF:
      if self.current.typ == TT_LBRACKET:
        brackets = brackets + 1
      elif self.current.typ == TT_RBRACKET:
        brackets = brackets - 1

      if brackets == 0:
        break
      elif brackets == 1:
        if self.current.typ == TT_ARROW:
          typ = 1
          break
        elif self.current.typ == TT_WALRUS:
          typ = 2
          break

      self.increment()

    if self.current.typ == TT_EOF and typ == 1:
      raise FatalError("expected `]`")

    self.lexer.replay()
    self.current = self.lexer.peek()

    if typ == 0:
      return self.parse_function_invocation()
    elif typ == 1:
      return self.parse_function_definition()
    elif typ == 2:
      return self.parse_symbol_bind()

  def parse_function_definition(self) -> Any:
    self.increment()

    args: List[Any] = []

    while self.current.typ not in [TT_ARROW, TT_EOF]:
      if self.current.typ != TT_SYMBOL:
        raise FatalError("expected symbols")

      if self.current in args:
        raise FatalError("argument `%s` already exists" % (self.current.val))

      args.append(self.current)
      self.increment()

    assert self.current.typ == TT_ARROW, "expected TT_ARROW"
    self.increment()

    body: Any = self.parse_program()

    if self.current.typ != TT_RBRACKET:
      raise FatalError("expected `]`")

    self.increment()
    return FunctionDefinitionNode(args, body)

  def parse_function_invocation(self) -> Any:
    self.increment()

    func: Any = self.parse_expression()
    args: List[Any] = []

    while self.current.typ not in [TT_RBRACKET, TT_EOF]:
      args.append(self.parse_expression())

    if self.current.typ != TT_RBRACKET:
      raise FatalError("expected `]`")

    self.increment()
    return FunctionInvocationNode(func, args)

  def parse_symbol_bind(self) -> Any:
    self.increment()

    symbol: Token = self.current

    self.increment()
    if self.current.typ != TT_WALRUS:
      raise FatalError("expected `:=`")
    self.increment()

    value: Any = self.parse_expression()
    if self.current.typ != TT_RBRACKET:
      raise FatalError("expected `]`")

    self.increment()
    return SymbolBindNode(symbol, value)

  def parse_atom(self) -> Any:
    if self.current.typ not in (TT_NUMBER, TT_STRING, TT_SYMBOL):
      raise FatalError("expected symbol or literal")

    node: LiteralNode = LiteralNode(self.current)
    self.increment()
    return node

  def parse(self) -> Any:
    program: ProgramNode = self.parse_program()

    if self.current.typ != TT_EOF:
      raise FatalError("invalid syntax")

    return program


class SymbolTable:
  def __init__(self, parent: Any = None) -> None:
    self.symbols: Dict[str, Any] = {}
    self.parent: Any = parent

  def get(self, name: str) -> Any:
    value: Any = self.symbols.get(name, None)
    if not value and self.parent:
      return self.parent.get(name)
    return value

  def bind(self, name: str, value: Any) -> None:
    self.symbols[name] = value


@dataclass(frozen=True)
class NumberValue:
  val: int | float

  @staticmethod
  def __repr__() -> str:
    return f"__number__"


@dataclass(frozen=True)
class StringValue:
  val: str

  @staticmethod
  def __repr__() -> str:
    return f"__string__"


@dataclass(frozen=True)
class FunctionValue:
  val: FunctionDefinitionNode | Callable
  st: SymbolTable

  def execute(self, args: List[Any]) -> Any:
    if callable(self.val):
      return self.val(args, self.st)

    if len(self.val.args) != len(args):
      raise FatalError("amount arguments doesn't match")

    for index, value in enumerate(args):
      # print(self.val.args[index].val, value)
      self.st.bind(self.val.args[index].val, value)

    return Interpreter.translate(self.val.body, self.st)

  @staticmethod
  def __repr__() -> str:
    return f"__function__"


@dataclass(frozen=True)
class VoidValue:
  @staticmethod
  def __repr__() -> str:
    return f"__void__"


class Interpreter:
  @staticmethod
  def LiteralNode(node: Any, st: SymbolTable) -> Any:
    if node.val.typ == TT_SYMBOL:
      value: Any = st.get(node.val.val)
      if not value:
        raise FatalError(f"unbound symbol; `{node.val.val}`")
      return value or node.val
    elif node.val.typ == TT_STRING:
      return StringValue(node.val.val)
    return NumberValue(node.val.val)

  @staticmethod
  def SymbolBindNode(node: Any, st: SymbolTable) -> Any:
    name: str = node.name.val
    value: Any = Interpreter.translate(node.val, st)
    st.bind(name, value)
    return value

  @staticmethod
  def FunctionDefinitionNode(node: Any, st: SymbolTable) -> Any:
    child: SymbolTable = SymbolTable(parent=SymbolTable(st.symbols.copy()))
    return FunctionValue(node, child)

  @staticmethod
  def FunctionInvocationNode(node: Any, st: SymbolTable) -> Any:
    # func: Any
    # args: List[Any]
    # child_st = SymbolTable(parent=st)

    func: Any = Interpreter.translate(node.func, st)
    args: List[Any] = [Interpreter.translate(arg, st) for arg in node.args]

    result = func.execute(args)
    return result[-1]

  @staticmethod
  def ProgramNode(node: Any, st: SymbolTable) -> Any:
    return [Interpreter.translate(child, st) for child in node.nodes]

  @staticmethod
  def translate(node: Any, st: SymbolTable) -> Any:
    method_name = f'{type(node).__name__}'
    method = getattr(Interpreter, method_name, Interpreter.undefined)
    return method(node, st)
  
  @staticmethod
  def undefined(node: Any, st: SymbolTable):
    raise Exception(f"undefined method; {type(node).__name__}")

class stdlib:
  @staticmethod
  def add(args: List[Any], st: SymbolTable) -> List[Any]:
    if len(args) == 0:
      raise FatalError("insufficient arguments")

    if len(types := {type(item) for item in args}) > 1:
      message: str = ", ".join([item.__repr__() for item in types])
      raise FatalError(f"unsupported operands between types; {message}")

    result: Any = args[0].val
    for arg in args[1:]:
      result += arg.val

    if isinstance(result, int) or isinstance(result, float):
      return [NumberValue(result)]
    elif isinstance(result, str):
      return [StringValue(result)]
    return [VoidValue()]

  @staticmethod
  def sub(args: List[Any], st: SymbolTable) -> List[Any]:
    if len(args) == 0:
      raise FatalError("insufficient arguments")

    if len(types := {type(item) for item in args}) > 1:
      message: str = ", ".join([item.__repr__() for item in types])
      raise FatalError(f"unsupported operands between types; {message}")

    result: int | float = args[0].val
    for arg in args[1:]:
      result -= arg.val

    return [NumberValue(result)]

  @staticmethod
  def mul(args: List[Any], st: SymbolTable) -> List[Any]:
    if len(args) == 0:
      raise FatalError("insufficient arguments")

    if len(args) == 2:
      if isinstance(args[0], StringValue) and isinstance(args[1], NumberValue):
        return [StringValue(args[0].val * int(args[1].val))]

    if len(types := {type(item) for item in args}) > 1:
      message: str = ", ".join([item.__repr__() for item in types])
      raise FatalError(f"unsupported operands between types; {message}")

    result: int | float = args[0].val
    for arg in args[1:]:
      result *= arg.val

    return [NumberValue(result)]

  @staticmethod
  def len(args: List[Any], st: SymbolTable) -> List[Any]:
    if len(args) == 0:
      raise FatalError("insufficient arguments")

    if not isinstance(args[0], StringValue):
      raise FatalError(f"expected {StringValue.__repr__()}")

    return [NumberValue(len(args[0].val))]

  @staticmethod
  def print(args: List[Any], st: SymbolTable) -> List[Any]:
    if len(args) == 0:
      print()

    for arg in args:
      if isinstance(arg, NumberValue):
        print("%g" % arg.val)
      elif isinstance(arg, StringValue):
        print("%s" % arg.val)
      else:
        raise FatalError(f"cannot print type; {arg}")

    return [VoidValue()]

  @staticmethod
  def number(args: List[Any], st: SymbolTable) -> List[Any]:
    if len(args) == 0:
      raise FatalError("insufficient arguments")


if __name__ == "__main__":
  global_st = SymbolTable()
  global_st.bind("+", FunctionValue(stdlib.add, global_st))
  global_st.bind("-", FunctionValue(stdlib.sub, global_st))
  global_st.bind("*", FunctionValue(stdlib.mul, global_st))
  global_st.bind("len", FunctionValue(stdlib.len, global_st))
  global_st.bind("print", FunctionValue(stdlib.print, global_st))

  try:
    lexer = Lexer(open("example.txt", "r").read())

    parser = Parser(lexer)
    ast = parser.parse()

    # print_node(ast)

    result = Interpreter.translate(ast, global_st)
  except FatalError as e:
    print("fatal-error:", e)

