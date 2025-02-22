import difflib
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from subprocess import Popen, PIPE
from textwrap import dedent

import pytest

from ..context import MLIRContext, ContextManager
from .generate_test_checks import main
from ..runtime.refbackend import LLVMJITBackend


def filecheck(correct: str, module):
    filecheck_name = "FileCheck"
    if platform.system() == "Windows":
        filecheck_name += ".exe"

    # try from mlir-native-tools
    filecheck_path = Path(sys.prefix) / "bin" / filecheck_name
    # try to find using which
    if not filecheck_path.exists():
        filecheck_path = shutil.which(filecheck_name)
    assert (
        filecheck_path is not None and Path(filecheck_path).exists() is not None
    ), "couldn't find FileCheck"

    correct = "\n".join(filter(None, correct.splitlines()))
    correct = dedent(correct)
    correct_with_checks = main(correct).replace("CHECK:", "CHECK-NEXT:")

    op = str(module).strip()
    op = "\n".join(filter(None, op.splitlines()))
    op = dedent(op)
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(correct_with_checks.encode())
        tmp.flush()
        p = Popen([filecheck_path, tmp.name], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out, err = map(lambda o: o.decode(), p.communicate(input=op.encode()))
        if p.returncode:
            diff = list(
                difflib.unified_diff(
                    op.splitlines(),  # to this
                    correct.splitlines(),  # delta from this
                    lineterm="",
                )
            )
            diff.insert(1, "delta from module to correct")
            print("lit report:", err, file=sys.stderr)
            raise ValueError("\n" + "\n".join(diff))


@pytest.fixture
def mlir_ctx() -> MLIRContext:
    with ContextManager(allow_unregistered_dialects=True) as ctx:
        yield ctx


@pytest.fixture
def backend() -> LLVMJITBackend:
    return LLVMJITBackend()
