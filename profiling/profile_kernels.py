from src.kernel import fused_ragged_dot_caller, ref_ragged_mlp_one_line

import os
import json
import time
import shutil
from pathlib import Path
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl


print("JAX version:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())


# Fixed configuration for profiling
M = 16_384
D = 128
F = 1024
E = 4

BM = 128
BF = 16

DTYPE = jnp.float32
SEED = 0


# Balanced group sizes
group_sizes_host = np.full(
    E,
    M // E,
    dtype=np.int32,
)

group_sizes_host[: M % E] += 1

assert len(group_sizes_host) == E
assert int(group_sizes_host.sum()) == M

group_sizes = jnp.asarray(
    group_sizes_host,
    dtype=jnp.int32,
)

max_group_size = int(group_sizes_host.max())

print("Group sizes:", group_sizes_host)
print("Maximum group size:", max_group_size)


# Create input and weight tensors
key = jax.random.PRNGKey(SEED)
key_x, key_up, key_down = jax.random.split(key, 3)

x = jax.random.normal(
    key_x,
    shape=(M, D),
    dtype=DTYPE,
)

w_up = jax.random.normal(
    key_up,
    shape=(E, D, F),
    dtype=DTYPE,
) / jnp.sqrt(D)

w_down = jax.random.normal(
    key_down,
    shape=(E, F, D),
    dtype=DTYPE,
) / jnp.sqrt(F)

# Force all initial allocations and random-number generation to finish.
jax.block_until_ready((x, w_up, w_down, group_sizes))

print("x shape:", x.shape)
print("w_up shape:", w_up.shape)
print("w_down shape:", w_down.shape)


# Fused implementation
fused_jit = jax.jit(
    lambda x, w_up, w_down, group_sizes: fused_ragged_dot_caller(
        x,
        w_up,
        w_down,
        group_sizes,
        BM=BM,
        BF=BF,
        max_group_size=max_group_size,
        interpret=False,
    )
)


# Reference implementation
reference_jit = jax.jit(ref_ragged_mlp_one_line)


@partial(
    jax.profiler.annotate_function,
    name="fused_ragged_mlp",
)
def run_fused():
    output = fused_jit(
        x,
        w_up,
        w_down,
        group_sizes,
    )

    return output.block_until_ready()


@partial(
    jax.profiler.annotate_function,
    name="reference_ragged_mlp",
)
def run_reference():
    output = reference_jit(
        x,
        w_up,
        w_down,
        group_sizes,
    )

    return output.block_until_ready()


PROFILE_ROOT = Path(os.environ.get("PROFILE_ROOT", "jax_profiles"))
FUSED_TRACE_DIR = PROFILE_ROOT / "fused"
REFERENCE_TRACE_DIR = PROFILE_ROOT / "reference"

shutil.rmtree(
    FUSED_TRACE_DIR,
    ignore_errors=True,
)

FUSED_TRACE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with jax.profiler.trace(
    str(FUSED_TRACE_DIR),
    create_perfetto_trace=True,
):
    for iteration in range(5):
        with jax.profiler.TraceAnnotation(
            "fused_iteration",
            iteration=iteration,
        ):
            run_fused()

print("Fused trace saved to:")
print(FUSED_TRACE_DIR)


shutil.rmtree(
    REFERENCE_TRACE_DIR,
    ignore_errors=True,
)

REFERENCE_TRACE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with jax.profiler.trace(
    str(REFERENCE_TRACE_DIR),
    create_perfetto_trace=True,
):
    for iteration in range(5):
        with jax.profiler.TraceAnnotation(
            "reference_iteration",
            iteration=iteration,
        ):
            run_reference()

print("Reference trace saved to:")
print(REFERENCE_TRACE_DIR)
