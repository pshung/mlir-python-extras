import inspect
import sys
from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.context import ContextManager
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.func import toMLIRFunc

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_emit(ctx: MLIRContext):
    @toMLIRFunc
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()
    correct = dedent(
        """\
    module {
      func.func @demo_fun1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, ctx.module)


def test_declare_byte_rep(ctx: MLIRContext):
    def demo_fun1():
        ...

    if sys.version_info.minor == 12:
        assert demo_fun1.__code__.co_code == b"\x97\x00y\x00"
    elif sys.version_info.minor == 11:
        assert demo_fun1.__code__.co_code == b"\x97\x00d\x00S\x00"
    elif sys.version_info.minor in {8, 9, 10}:
        assert demo_fun1.__code__.co_code == b"d\x00S\x00"
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")


def test_declare(ctx: MLIRContext):
    @toMLIRFunc
    def demo_fun1() -> T.i32():
        ...

    @toMLIRFunc
    def demo_fun2() -> (T.i32(), T.i32()):
        ...

    @toMLIRFunc
    def demo_fun3(x: T.i32()) -> (T.i32(), T.i32()):
        ...

    @toMLIRFunc
    def demo_fun4(x: T.i32(), y: T.i32()) -> (T.i32(), T.i32()):
        ...

    demo_fun1()
    demo_fun2()
    one = constant(1)
    demo_fun3(one)
    demo_fun4(one, one)

    ctx.module.operation.verify()
    correct = dedent(
        """\
    module {
      func.func private @demo_fun1() -> i32
      func.func private @demo_fun2() -> (i32, i32)
      func.func private @demo_fun3(i32) -> (i32, i32)
      func.func private @demo_fun4(i32, i32) -> (i32, i32)
      %0 = func.call @demo_fun1() : () -> i32
      %1:2 = func.call @demo_fun2() : () -> (i32, i32)
      %c1_i32 = arith.constant 1 : i32
      %2:2 = func.call @demo_fun3(%c1_i32) : (i32) -> (i32, i32)
      %3:2 = func.call @demo_fun4(%c1_i32, %c1_i32) : (i32, i32) -> (i32, i32)
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_base_meta(ctx: MLIRContext):
    @toMLIRFunc
    def foo1():
        one = constant(1)
        return one

    foo1.emit()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
    }
    """
    )
    filecheck(correct, ctx.module)

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_base_meta2(ctx: MLIRContext):
    @toMLIRFunc
    def foo1():
        one = constant(1)
        return one

    foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, ctx.module)


def test_func_no_context():
    @toMLIRFunc
    def foo1():
        one = constant(1)
        return one

    with ContextManager() as mod_ctx:
        foo1()
    correct = dedent(
        """\
    module {
      func.func @foo1() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        return %c1_i32 : i32
      }
      %0 = func.call @foo1() : () -> i32
    }
    """
    )
    filecheck(correct, mod_ctx.module)
