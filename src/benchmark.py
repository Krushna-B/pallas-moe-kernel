import time
import functools
import pandas as pd
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

import numpy as np

from src.kernel import (
    fused_ragged_dot_caller,
    ref_ragged_mlp_one_line,
    ref_ragged_mlp,
)


def make_ref_jit():
    return jax.jit(ref_ragged_mlp_one_line)


def make_fused_jit(
    BM,
    BF,
    max_group_size,
    interpret,
):
    return jax.jit(
        lambda x, w_up, w_down, group_offsets: fused_ragged_dot_caller(
            x,
            w_up,
            w_down,
            group_offsets,
            max_group_size=max_group_size,
            BM=BM,
            BF=BF,
            interpret=interpret,
        )
    )


# Make group sizes
def make_group_sizes(
    M,
    E,
    group_fractions=None,
):
    if M < E:
        raise ValueError(
            f"M={M} must be at least E={E} so every expert "
            "can receive at least one token."
        )

    if group_fractions is None:
        # Moderately uneven default distribution
        weights = np.linspace(
            0.85,
            1.15,
            E,
            dtype=np.float64,
        )
    else:
        weights = np.asarray(
            group_fractions,
            dtype=np.float64,
        )

        if weights.shape != (E,):
            raise ValueError(
                f"group_fractions must have shape ({E},), but received {weights.shape}."
            )

        if np.any(weights <= 0):
            raise ValueError("Every group fraction must be positive.")

    weights = weights / weights.sum()

    exact_sizes = M * weights
    group_sizes = np.floor(exact_sizes).astype(np.int32)

    # Give remaining tokens to experts with the largest
    # fractional remainders
    remainder = M - int(group_sizes.sum())

    fractional_parts = exact_sizes - group_sizes
    remainder_order = np.argsort(-fractional_parts)

    group_sizes[remainder_order[:remainder]] += 1

    assert int(group_sizes.sum()) == M

    return group_sizes


def benchmark_fn(fn, *args, warmup=2, iters=4):
    # Compile and intial run
    start = time.perf_counter()
    output = fn(*args)
    output.block_until_ready()
    first_call_ms = (time.perf_counter() - start) * 1000.0

    # Warmup the function Note: Warmup is faster since it loads into caches
    for _ in range(warmup):
        output = fn(*args)
        output.block_until_ready()

    print(f"Warmup done")
    # Time Runs
    times_ms = []
    for _ in range(iters):
        start = time.perf_counter()

        output = fn(*args)
        output.block_until_ready()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        times_ms.append(elapsed_ms)

    times_ms = np.asarray(
        times_ms,
        dtype=np.float64,
    )

    return {
        "first_call_ms": first_call_ms,
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "times_ms": times_ms,
        "output": output,
    }


def run_sweep(
    M_values,
    *,
    E=4,
    D=512,
    F=2048,
    BM=16,
    BF=64,
    dtype=jnp.float32,
    warmup=2,
    iters=4,
    interpret=False,
    group_fractions=None,
    seed=0,
):
    if F % BF != 0:
        raise ValueError(f"F={F} must be divisible by BF={BF}.")
    key = jax.random.PRNGKey(0)
    rows = []

    # Make the two functions jit
    ref_jit = make_ref_jit()

    # All Token values
    for M in M_values:
        # Make group sizes
        group_sizes_host = make_group_sizes(
            M=M,
            E=E,
            group_fractions=group_fractions,
        )
        max_group_size = int(group_sizes_host.max())

        group_sizes = jnp.asarray(
            group_sizes_host,
            dtype=jnp.int32,
        )
        fused_jit = make_fused_jit(
            BM=BM, BF=BF, max_group_size=max_group_size, interpret=interpret
        )

        key, k1, k2, k3 = jax.random.split(
            key,
            4,
        )

        # Create x and weight matrices
        x = jax.random.normal(
            k1,
            (M, D),
            dtype=dtype,
        )
        w_up = jax.random.normal(
            k2,
            (E, D, F),
            dtype=dtype,
        ) / jnp.sqrt(D)

        w_down = jax.random.normal(
            k3,
            (E, F, D),
            dtype=dtype,
        ) / jnp.sqrt(F)

        print(f"\nM={M}: compiling/benchmarking ragged_dot...")
        ref_stats = benchmark_fn(
            ref_jit,
            x,
            w_up,
            w_down,
            group_sizes,
            warmup=warmup,
            iters=iters,
        )
        print(f"M={M}: compiling/benchmarking fused Pallas...")

        fused_stats = benchmark_fn(
            fused_jit,
            x,
            w_up,
            w_down,
            group_sizes,
            warmup=warmup,
            iters=iters,
        )

        y_ref = ref_stats["output"]
        y_fused = fused_stats["output"]

        # Correctness
        difference = y_fused - y_ref
        absolute_error = jnp.abs(difference)

        max_absolute_error = float(jnp.max(absolute_error))

        mean_absolute_error = float(jnp.mean(absolute_error))

        reference_norm = jnp.linalg.norm(y_ref)
        difference_norm = jnp.linalg.norm(difference)

        l2_relative_error = float(difference_norm / jnp.maximum(reference_norm, 1e-12))

        normalized_max_error = float(
            jnp.max(absolute_error)
            / jnp.maximum(
                jnp.max(jnp.abs(y_ref)),
                1e-12,
            )
        )

        fused_is_finite = bool(jnp.all(jnp.isfinite(y_fused)))

        reference_is_finite = bool(jnp.all(jnp.isfinite(y_ref)))

        # A global numerical acceptance criterion
        numerically_acceptable = bool(
            fused_is_finite
            and reference_is_finite
            and l2_relative_error < 1e-4
            and normalized_max_error < 1e-3
        )

        #### Performace
        ref_ms = ref_stats["median_ms"]
        fused_ms = fused_stats["median_ms"]

        speedup = ref_ms / fused_ms

        ref_tokens_per_second = M / (ref_ms / 1000.0)

        fused_tokens_per_second = M / (fused_ms / 1000.0)

        # Two matrix multiplications:
        #
        # up:   2 * M * D * F
        # down: 2 * M * F * D
        #
        # total = 4 * M * D * F
        total_flops = 4 * M * D * F

        ref_tflops = total_flops / (ref_ms / 1000.0) / 1e12

        fused_tflops = total_flops / (fused_ms / 1000.0) / 1e12

        hidden_intermediate_mib = M * F * jnp.dtype(dtype).itemsize / (1024**2)

        rows.append(
            {
                "M": M,
                "E": E,
                "D": D,
                "F": F,
                "BM": BM,
                "BF": BF,
                "group_sizes": tuple(int(v) for v in group_sizes_host),
                "max_group_size": max_group_size,
                "grid_expert_blocks": (max_group_size + BM - 1) // BM,
                "fused_is_finite": fused_is_finite,
                "reference_is_finite": reference_is_finite,
                "numerically_acceptable": numerically_acceptable,
                "max_absolute_error": max_absolute_error,
                "mean_absolute_error": mean_absolute_error,
                "normalized_max_error": normalized_max_error,
                "l2_relative_error": l2_relative_error,
                "ref_compile_plus_first_ms": ref_stats["first_call_ms"],
                "fused_compile_plus_first_ms": fused_stats["first_call_ms"],
                "ref_mean_ms": ref_stats["mean_ms"],
                "ref_median_ms": ref_stats["median_ms"],
                "ref_min_ms": ref_stats["min_ms"],
                "ref_max_ms": ref_stats["max_ms"],
                "ref_std_ms": ref_stats["std_ms"],
                "fused_mean_ms": fused_stats["mean_ms"],
                "fused_median_ms": fused_stats["median_ms"],
                "fused_min_ms": fused_stats["min_ms"],
                "fused_max_ms": fused_stats["max_ms"],
                "fused_std_ms": fused_stats["std_ms"],
                "speedup_ref_over_fused": speedup,
                "ref_tokens_per_second": ref_tokens_per_second,
                "fused_tokens_per_second": fused_tokens_per_second,
                "ref_tflops": ref_tflops,
                "fused_tflops": fused_tflops,
                # This is the global-memory intermediate that the
                # fused kernel is intended to avoid materializing.
                "hidden_intermediate_mib": hidden_intermediate_mib,
                "ref_run_times_ms": ref_stats["times_ms"].tolist(),
                "fused_run_times_ms": fused_stats["times_ms"].tolist(),
            }
        )

        print(
            f"M={M} finished | "
            f"groups={group_sizes_host.tolist()} | "
            f"reference={ref_ms:.4f} ms | "
            f"fused={fused_ms:.4f} ms | "
            f"speedup={speedup:.3f}x | "
            f"L2 error={l2_relative_error:.3e} | "
            f"valid={numerically_acceptable}"
        )

    return pd.DataFrame(rows)
