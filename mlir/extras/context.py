import contextlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

from .. import ir


@dataclass
class MLIRContext:
    context: ir.Context
    module: ir.Module

    def __str__(self):
        return str(self.module)


@contextmanager
def ContextManager(
    loc: ir.Location = None,
    allow_unregistered_dialects=False,
) -> MLIRContext:
    """
    A context manger for MLIR's Context, Location, Module and InsertionPoint.
    This acts as a syntax surgar to the following "with" statement.
    with ir.Context() as ctx:
        module = ir.Module.create()
        with ir.InsertionPoint(module.body), ir.Location.unknown()):
            ...
            ...
            ...
    This returns a MLIRContext object so that users can call __enter__ and __exit__ at their disposal.
    usage:
    ctx = ContextManager()
    ctx.__enter__()
    ...MLIR IR builder
    ctx.__exit__()
    """
    context = ir.Context()
    if allow_unregistered_dialects:
        context.allow_unregistered_dialects = True

    with ExitStack() as stack:
        stack.enter_context(context)
        if loc is None:
            loc = ir.Location.unknown()
        stack.enter_context(loc)
        module = ir.Module.create()
        ip = ir.InsertionPoint(module.body)
        stack.enter_context(ip)
        yield MLIRContext(context, module)

    context._clear_live_operations()


@contextlib.contextmanager
def enable_multithreading(context=None):
    from ..ir import Context

    if context is None:
        context = Context.current
    context.enable_multithreading(True)
    yield
    context.enable_multithreading(False)


@contextlib.contextmanager
def disable_multithreading(context=None):
    from ..ir import Context

    if context is None:
        context = Context.current

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)
