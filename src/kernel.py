import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


### Kernel for fusing together the W_up and W_down projections
def fused_mlp_one_expert(
    x_ref, w_up_ref, w_down_ref, group_offests_ref, y_ref, *, M, D, F, BM, BF
):
    expert_id = pl.program_id(0)
    pid_m = pl.program_id(1)  # Axis 0

    expert_start = pl.load(group_offests_ref, (expert_id,))
    expert_end = pl.load(group_offests_ref, (expert_id + 1,))

    m_start = expert_start + pid_m * BM

    @pl.when(m_start < expert_end)
    def compute_block():
        m_local = jnp.arange(BM)
        m_offsets = m_start + m_local
        m_mask = m_offsets < expert_end

        d_offsets = jnp.arange(D)

        # Load input X matrix
        # global shape (M, D), loading (BM, D)
        x = pl.load(
            x_ref,
            (m_offsets[:, None], d_offsets[None, :]),
            mask=m_mask[:, None],
            other=0.0,
        ).astype(jnp.float32)

        # Accumulator
        acc = jnp.zeros((BM, D), dtype=jnp.float32)

        # streaming chunks
        def streaming_hidden_block(f_block, acc):
            # Stream though hidden chunks
            f_offsets = f_block * BF + jnp.arange(BF)

            # w_up global shape (E, D, F)
            w_up = pl.load(
                w_up_ref,
                (
                    expert_id,
                    d_offsets[:, None],
                    f_offsets[None, :],
                ),
            ).astype(jnp.float32)

            # w_down global shape (E, F, D)
            w_down = pl.load(
                w_down_ref,
                (
                    expert_id,
                    f_offsets[:, None],
                    d_offsets[None, :],
                ),
            ).astype(jnp.float32)

            # Compute ops
            # (BM, D) @ (D, BF) -> (BM, BF)
            h = jax.nn.gelu(pl.dot(x, w_up))

            # (BM, BF) @ (BF, D) -> (BM, D)
            acc += pl.dot(h, w_down)
            return acc

        # Jax for loop
        acc = jax.lax.fori_loop(0, F // BF, streaming_hidden_block, acc)

        # Store results
        pl.store(
            y_ref,
            idx=(m_offsets[:, None], d_offsets[None, :]),
            val=acc,
            mask=m_mask[:, None],
        )


def fused_ragged_dot_caller(
    x,
    w_up,
    w_down,
    group_sizes,
    *,
    BM: int = 16,
    BF: int = 64,
    max_group_size: int,
    interpret: bool = False,
):
    group_offsets = group_sizes_to_offsets(group_sizes)
    M, D = x.shape
    E, D2, F = w_up.shape
    assert D == D2
    assert w_down.shape == (E, F, D)
    assert M % BM == 0
    assert F % BF == 0

    grid = (
        E,
        pl.cdiv(max_group_size, BM),
    )

    kernel = functools.partial(
        fused_mlp_one_expert,
        M=M,
        D=D,
        F=F,
        BM=BM,
        BF=BF,
    )

    return pl.pallas_call(
        kernel=kernel,
        out_shape=jax.ShapeDtypeStruct((M, D), dtype=jnp.float32),
        grid=grid,
        interpret=interpret,
    )(x, w_up, w_down, group_offsets)


def group_sizes_to_offsets(group_sizes):
    return jnp.concatenate(
        [
            jnp.zeros((1,), dtype=group_sizes.dtype),
            jnp.cumsum(group_sizes),
        ]
    )


# Reference implementations (two ragged_dot calls with the intermediate
# (M, F) activation materialized in HBM)
def ref_ragged_mlp_one_line(x, w_up, w_down, group_sizes):
    return jax.lax.ragged_dot(
        jax.nn.gelu(
            jax.lax.ragged_dot(
                x,
                w_up,
                group_sizes,
            ),
            approximate=True,
        ),
        w_down,
        group_sizes,
    )


def ref_ragged_mlp(
    x,
    w_up,
    w_down,
    group_sizes,
):
    h = jax.lax.ragged_dot(
        x,
        w_up,
        group_sizes,
        preferred_element_type=jnp.float32,
    )

    h = jax.nn.gelu(
        h,
    )
    y = jax.lax.ragged_dot(
        h,
        w_down,
        group_sizes,
        preferred_element_type=jnp.float32,
    )

    return y
