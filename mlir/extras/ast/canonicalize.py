import ast
import difflib
import enum
import inspect
import logging
import types
from abc import ABC, abstractmethod
from dis import findlinestarts
from opcode import opmap
from types import CodeType
from typing import List, Union, Callable

import astunparse
from bytecode import ConcreteBytecode

from ..ast.util import get_function_ast, copy_func

logger = logging.getLogger(__name__)


class Transformer(ast.NodeTransformer):
    def __init__(self, context, first_lineno):
        super().__init__()
        self.context = context
        self.first_lineno = first_lineno


class StrictTransformer(Transformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node


def transform_func(f, *transformer_ctors: type(Transformer)):
    ast_module = get_function_ast(f)
    context = types.SimpleNamespace()
    for transformer_ctor in transformer_ctors:
        orig_code = astunparse.unparse(ast_module)
        func_node = ast_module.body[0]
        replace = transformer_ctor(
            context=context, first_lineno=f.__code__.co_firstlineno - 1
        )
        logger.debug("[transformer] %s", replace.__class__.__name__)
        func_node = replace.generic_visit(func_node)
        new_code = astunparse.unparse(func_node)

        diff = list(
            difflib.unified_diff(
                orig_code.splitlines(),  # to this
                new_code.splitlines(),  # delta from this
                lineterm="",
            )
        )
        logger.debug("[transformed code diff]\n%s", "\n" + "\n".join(diff))
        logger.debug("[transformed code]\n%s", new_code)
        ast_module.body[0] = func_node

    logger.debug("[final transformed code]\n\n%s", new_code)

    return ast_module


def transform_ast(
    f, transformers: List[Union[type(Transformer), type(StrictTransformer)]] = None
):
    if transformers is None:
        return f

    module = transform_func(f, *transformers)
    module = ast.fix_missing_locations(module)
    module = ast.increment_lineno(module, f.__code__.co_firstlineno - 1)
    module_code_o = compile(module, f.__code__.co_filename, "exec")
    new_f_code_o = next(
        c
        for c in module_code_o.co_consts
        if type(c) is CodeType and c.co_name == f.__name__
    )
    n_lines = len(inspect.getsource(f).splitlines())
    line_starts = list(findlinestarts(new_f_code_o))
    assert (
        max([l for _, l in line_starts]) - min([l for _, l in line_starts]) + 1
        <= n_lines
    ), f"something went wrong with the line numbers for the rewritten/canonicalized function"
    return copy_func(f, new_f_code_o)


# this is like this because i couldn't figure out how to subclass
# Enum and simultaneously pass in opmap
OpCode = enum.Enum("OpCode", opmap)


def to_int(self: OpCode):
    return self.value


def to_str(self: OpCode):
    return self.name


setattr(OpCode, "__int__", to_int)
setattr(OpCode, "__str__", to_str)


class BytecodePatcher(ABC):
    def __init__(self, context=None):
        self.context = context

    @abstractmethod
    def patch_bytecode(self, code: ConcreteBytecode, original_f) -> ConcreteBytecode:
        pass


def patch_bytecode(f, patchers: List[type(BytecodePatcher)] = None):
    if patchers is None:
        return f

    bytecode = ConcreteBytecode.from_code(f.__code__)
    context = types.SimpleNamespace()
    for patcher in patchers:
        bytecode = patcher(context).patch_bytecode(bytecode, f)

    return copy_func(f, bytecode.to_code())


class Canonicalizer(ABC):
    @property
    @abstractmethod
    def ast_transformers(self) -> List[StrictTransformer]:
        pass

    @property
    @abstractmethod
    def bytecode_patchers(self) -> List[BytecodePatcher]:
        pass


def canonicalize(*, using: Canonicalizer):
    """
    A decorator to transform function AST and patch Python byte code
    """
    def wrapper(f : Callable):
        f = transform_ast(f, using.ast_transformers)
        f = patch_bytecode(f, using.bytecode_patchers)
        return f

    return wrapper
