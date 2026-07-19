### Kernel Practice, simple Pallas kernels building up to the fused MoE kernel to learn from the Pallas Docs
import jax
import functools
import jax.numpy as jnp
from jax.experimental import pallas as pl


# Simple Kernel's that handle operation for 1 element
def iota_kernel(o_ref):
    pid = pl.program_id(0)
    o_ref[pid] = pid


def iota(size):
    return pl.pallas_call(
        kernel=iota_kernel,
        out_shape=jax.ShapeDtypeStruct((size,), dtype=jnp.int32),
        grid=(size,),
        interpret=True,
    )()


def copy_kernel(x_ref, o_ref):
    pid = pl.program_id(0)
    o_ref[pid] = x_ref[pid]


def copy_pallas(x):
    return pl.pallas_call(
        kernel=copy_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, dtype=x.dtype),
        grid=(x.shape[0],),
        interpret=True,
    )(x)


# x = jnp.array([1,2,4,5,7,10])
# print(copy_pallas(x))
# Handling operations for block or slice


def masked_block_add_one_kernel(x_ref, o_ref, *, n, BLOCK):
    pid = pl.program_id(0)
    offsets = pid * BLOCK + jnp.arange(BLOCK)  # Array of memory addresses
    mask = offsets < n
    x = pl.load(
        x_ref, offsets, mask=mask, other=0.0
    )  # Load Values at those memory addr with mask

    y = x * 2.0
    pl.store(o_ref, offsets, y, mask=mask)


def block_add_one_pallas(x, BLOCK=4):
    n = x.shape[0]
    return pl.pallas_call(
        kernel=functools.partial(masked_block_add_one_kernel, n=n, BLOCK=BLOCK),
        out_shape=jax.ShapeDtypeStruct(x.shape, dtype=x.dtype),
        grid=(pl.cdiv(n, BLOCK),),
        interpret=True,
    )(x)


# x = jnp.arange(10, dtype=jnp.float32)
# print(block_add_one_pallas(x, BLOCK=4))
