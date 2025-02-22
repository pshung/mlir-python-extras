import numpy as np

# you need this to register the memref value caster
# noinspection PyUnresolvedReferences
import mlir.extras.dialects.ext.memref
import mlir.extras.types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import MLIRContext, ContextManager
from mlir.extras.dialects.ext.arith import constant
from mlir.extras.dialects.ext.func import toMLIRFunc
from mlir.extras.dialects.ext.scf import canonicalizer as scf_canonicalizer, range_ as range
from mlir.extras.runtime.passes import Pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend


def setting_memref(ctx: MLIRContext, backend: LLVMJITBackend):
    K = 10
    memref_i64 = T.memref(K, K, T.i64())

    @toMLIRFunc
    @canonicalize(using=scf_canonicalizer)
    def memfoo(A: memref_i64, B: memref_i64, C: memref_i64):
        one = constant(1)
        two = constant(2)
        if one > two:
            three = constant(3)
        else:
            for i in range(0, K):
                for j in range(0, K):
                    C[i, j] = A[i, j] * B[i, j]

    memfoo.emit()
    # run_pipeline(ctx.module, Pipeline().cse())
    print(ctx.module)

    module = backend.compile(
        ctx.module,
        kernel_name=memfoo.__name__,
        pipeline=Pipeline().bufferize().lower_to_llvm(),
    )

    # windows defaults to int32
    A = np.random.randint(0, 10, (K, K)).astype(np.int64)
    B = np.random.randint(0, 10, (K, K)).astype(np.int64)
    C = np.zeros((K, K), dtype=np.int64)

    backend.load(module).memfoo(A, B, C)
    assert np.array_equal(A * B, C)


with MLIRContext() as ctx:
    setting_memref(ctx, LLVMJITBackend())
