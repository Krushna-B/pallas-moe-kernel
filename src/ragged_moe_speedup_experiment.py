"""
Reproducible performance sweep for a fused ragged MoE MLP Pallas kernel

The experiment compares:

    H = GELU(ragged_dot(X, W_up, group_sizes))
    Y = ragged_dot(H, W_down, group_sizes)

Outputs
-------
<output_dir>/
    config.json
    environment.json
    results.csv
    failures.csv
    best_configs.csv
    summary_by_M.csv
    summary_by_D.csv
    summary_by_F.csv
    summary_by_E.csv
    summary_by_BM.csv
    summary_by_BF.csv
    plots/
        measured_vs_ideal_speedup.png
        reference_vs_fused_latency.png
        speedup_vs_M_summary.png
        speedup_vs_hidden_intermediate.png
        speedup_vs_padding_overhead.png
        prediction_error_vs_M.png
        median_speedup_BM_BF_heatmap.png
        speedup_by_expert_count.png
        correctness_l2_error.png
        by_shape/*.png


"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import functools
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

try:
    import jaxlib
except ImportError:  # pragma: no cover
    jaxlib = None


# Compatibility layer for Pallas API changes JAX <=0.8 exposes generic
# pl.load/pl.store. JAX >=0.9 moved masked loads/stores to backend modules
_HAS_LEGACY_PALLAS_MEMORY_OPS = hasattr(pl, "load") and hasattr(pl, "store")

try:
    from jax.experimental.pallas import triton as pltriton
except ImportError:  # pragma: no cover
    pltriton = None

try:
    from jax.experimental.pallas import tpu as pltpu
except ImportError:  # pragma: no cover
    pltpu = None


def _pallas_load(ref, idx, *, mask=None, other=None):
    if _HAS_LEGACY_PALLAS_MEMORY_OPS:
        return pl.load(ref, idx, mask=mask, other=other)

    view = ref.at[idx]
    backend = jax.default_backend()
    if backend == "tpu":
        if pltpu is None:
            raise RuntimeError("The Pallas TPU module is unavailable.")
        value = pltpu.load(view, mask=mask)
        if mask is not None and other is not None:
            value = jnp.where(mask, value, other)
        return value

    if pltriton is None:
        raise RuntimeError("The Pallas Triton module is unavailable.")
    return pltriton.load(view, mask=mask, other=other)


def _pallas_store(ref, idx, value, *, mask=None):
    if _HAS_LEGACY_PALLAS_MEMORY_OPS:
        pl.store(ref, idx, value, mask=mask)
        return

    view = ref.at[idx]
    backend = jax.default_backend()
    if backend == "tpu":
        if pltpu is None:
            raise RuntimeError("The Pallas TPU module is unavailable.")
        pltpu.store(view, value, mask=mask)
        return

    if pltriton is None:
        raise RuntimeError("The Pallas Triton module is unavailable.")
    pltriton.store(view, value, mask=mask)


def _pallas_dot(lhs, rhs):
    if hasattr(pl, "dot"):
        return pl.dot(lhs, rhs)
    return jnp.dot(lhs, rhs)


# 1. Fused kernel and reference
def group_sizes_to_offsets(group_sizes: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [
            jnp.zeros((1,), dtype=group_sizes.dtype),
            jnp.cumsum(group_sizes),
        ]
    )


def fused_mlp_one_expert(
    x_ref,
    w_up_ref,
    w_down_ref,
    group_offsets_ref,
    y_ref,
    *,
    M: int,
    D: int,
    F: int,
    BM: int,
    BF: int,
):
    """One Pallas program computes one BM-row tile for one expert"""
    del M  # M is encoded in the output shape; routing bounds the useful rows.

    expert_id = pl.program_id(0)
    pid_m = pl.program_id(1)

    expert_start = _pallas_load(group_offsets_ref, (expert_id,))
    expert_end = _pallas_load(group_offsets_ref, (expert_id + 1,))
    m_start = expert_start + pid_m * BM

    @pl.when(m_start < expert_end)
    def compute_block():
        m_offsets = m_start + jnp.arange(BM)
        m_mask = m_offsets < expert_end
        d_offsets = jnp.arange(D)

        x = _pallas_load(
            x_ref,
            (m_offsets[:, None], d_offsets[None, :]),
            mask=m_mask[:, None],
            other=0.0,
        ).astype(jnp.float32)

        acc = jnp.zeros((BM, D), dtype=jnp.float32)

        def streaming_hidden_block(f_block, current_acc):
            f_offsets = f_block * BF + jnp.arange(BF)

            w_up = _pallas_load(
                w_up_ref,
                (
                    expert_id,
                    d_offsets[:, None],
                    f_offsets[None, :],
                ),
            ).astype(jnp.float32)

            w_down = _pallas_load(
                w_down_ref,
                (
                    expert_id,
                    f_offsets[:, None],
                    d_offsets[None, :],
                ),
            ).astype(jnp.float32)

            hidden = jax.nn.gelu(
                _pallas_dot(x, w_up),
                approximate=True,
            )
            return current_acc + _pallas_dot(hidden, w_down)

        acc = jax.lax.fori_loop(
            0,
            F // BF,
            streaming_hidden_block,
            acc,
        )

        _pallas_store(
            y_ref,
            (m_offsets[:, None], d_offsets[None, :]),
            acc,
            mask=m_mask[:, None],
        )


def fused_ragged_mlp_caller(
    x: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    group_sizes: jax.Array,
    *,
    BM: int,
    BF: int,
    max_group_size: int,
    interpret: bool = False,
) -> jax.Array:
    M, D = x.shape
    E, D2, F = w_up.shape

    if D != D2:
        raise ValueError(f"x has D={D}, but w_up has D={D2}.")
    if w_down.shape != (E, F, D):
        raise ValueError(f"Expected w_down shape {(E, F, D)}, got {w_down.shape}.")
    if F % BF != 0:
        raise ValueError(f"F={F} must be divisible by BF={BF}.")

    group_offsets = group_sizes_to_offsets(group_sizes)
    grid = (E, pl.cdiv(max_group_size, BM))

    kernel = functools.partial(
        fused_mlp_one_expert,
        M=M,
        D=D,
        F=F,
        BM=BM,
        BF=BF,
    )

    call_kwargs = {
        "out_shape": jax.ShapeDtypeStruct((M, D), dtype=jnp.float32),
        "grid": grid,
        "interpret": interpret,
    }

    # In JAX >=0.9, generic masked memory operations were removed, On GPU,
    # this kernel uses the Triton memory primitives, so explicitly select the
    # Triton compiler when executing rather than interpreting
    if (
        not _HAS_LEGACY_PALLAS_MEMORY_OPS
        and jax.default_backend() == "gpu"
        and not interpret
        and pltriton is not None
        and hasattr(pltriton, "CompilerParams")
    ):
        call_kwargs["compiler_params"] = pltriton.CompilerParams()

    return pl.pallas_call(kernel, **call_kwargs)(x, w_up, w_down, group_offsets)


def reference_ragged_mlp(
    x: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    hidden = jax.lax.ragged_dot(
        x,
        w_up,
        group_sizes,
        preferred_element_type=jnp.float32,
    )
    hidden = jax.nn.gelu(hidden, approximate=True)
    return jax.lax.ragged_dot(
        hidden,
        w_down,
        group_sizes,
        preferred_element_type=jnp.float32,
    )


REFERENCE_JIT = jax.jit(reference_ragged_mlp)


def make_fused_jit(
    *,
    BM: int,
    BF: int,
    max_group_size: int,
    interpret: bool,
) -> Callable[..., jax.Array]:
    return jax.jit(
        lambda x, w_up, w_down, group_sizes: fused_ragged_mlp_caller(
            x,
            w_up,
            w_down,
            group_sizes,
            BM=BM,
            BF=BF,
            max_group_size=max_group_size,
            interpret=interpret,
        )
    )


# Experiment configuration
@dataclasses.dataclass
class SweepConfig:
    # Values to test.
    M_values: tuple[int, ...] = (1024, 4096, 16384, 65536, 131072)
    D_values: tuple[int, ...] = (128, 256, 512)
    F_values: tuple[int, ...] = (512, 1024, 2048, 4096)
    E_values: tuple[int, ...] = (1, 2, 4, 8)
    BM_values: tuple[int, ...] = (8, 16, 32)
    BF_values: tuple[int, ...] = (32, 64, 128)
    group_patterns: tuple[str, ...] = ("balanced", "mild")

    # Baseline used by structured mode.
    baseline_M: int = 16384
    baseline_D: int = 512
    baseline_F: int = 2048
    baseline_E: int = 4

    # structured: vary one problem dimension at a time, but test every BM/BF
    # full_grid: full Cartesian product of M,D,F,E,pattern,BM,BF
    # shapes_only: full M,D,F,E,pattern grid at baseline BM/BF
    mode: str = "structured"

    warmup: int = 5
    iterations: int = 20
    seed: int = 0
    shuffle_cases: bool = True
    interpret: bool = False

    input_dtype: str = "float32"
    l2_relative_error_tolerance: float = 1e-4
    normalized_max_error_tolerance: float = 1e-3

    # A conservative preflight estimate. Set to None to disable skipping
    max_estimated_problem_memory_gib: Optional[float] = 12.0

    # Optional hardware values for roofline calculations
    peak_compute_tflops: Optional[float] = None
    peak_bandwidth_gbps: Optional[float] = None

    # Optional source-level storage comparison.
    shared_memory_limit_bytes: Optional[int] = None

    # Keep long sweeps recoverable and control plot count
    resume: bool = True
    clear_jax_caches_between_problem_shapes: bool = True
    max_by_shape_plots: int = 100
    max_cases: Optional[int] = None

    def validate(self) -> None:
        valid_modes = {"structured", "full_grid", "shapes_only"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}")

        for name in (
            "M_values",
            "D_values",
            "F_values",
            "E_values",
            "BM_values",
            "BF_values",
        ):
            values = getattr(self, name)
            if not values or any(int(v) <= 0 for v in values):
                raise ValueError(f"{name} must contain positive integers.")

        if self.warmup < 0 or self.iterations <= 0:
            raise ValueError("warmup must be >= 0 and iterations must be > 0.")

        if self.input_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("input_dtype must be float32, bfloat16, or float16.")

        valid_patterns = {"balanced", "mild", "skewed", "one_dominant"}
        unknown = set(self.group_patterns) - valid_patterns
        if unknown:
            raise ValueError(f"Unknown group patterns: {sorted(unknown)}")


DEFAULT_CONFIG = SweepConfig()


# 3. Case generation and routing distributions
def _unique_preserving_order(
    values: Iterable[tuple[Any, ...]],
) -> list[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    output: list[tuple[Any, ...]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def build_problem_shapes(config: SweepConfig) -> list[tuple[int, int, int, int, str]]:
    """Return tuples of (M, D, F, E, group_pattern)."""
    if config.mode in {"full_grid", "shapes_only"}:
        shapes = itertools.product(
            config.M_values,
            config.D_values,
            config.F_values,
            config.E_values,
            config.group_patterns,
        )
        return list(shapes)

    # Structured design: interpretable one-factor sweeps around a baseline.
    bM, bD, bF, bE = (
        config.baseline_M,
        config.baseline_D,
        config.baseline_F,
        config.baseline_E,
    )

    shapes: list[tuple[int, int, int, int, str]] = []
    for pattern in config.group_patterns:
        shapes.extend((M, bD, bF, bE, pattern) for M in config.M_values)
        shapes.extend((bM, D, bF, bE, pattern) for D in config.D_values)
        shapes.extend((bM, bD, F, bE, pattern) for F in config.F_values)
        shapes.extend((bM, bD, bF, E, pattern) for E in config.E_values)

    return _unique_preserving_order(shapes)


def build_tile_pairs(config: SweepConfig) -> list[tuple[int, int]]:
    if config.mode == "shapes_only":
        baseline_BM = config.BM_values[0]
        baseline_BF = config.BF_values[0]
        return [(baseline_BM, baseline_BF)]
    return list(itertools.product(config.BM_values, config.BF_values))


def routing_weights(E: int, pattern: str) -> np.ndarray:
    if pattern == "balanced":
        weights = np.ones(E, dtype=np.float64)
    elif pattern == "mild":
        weights = np.linspace(0.85, 1.15, E, dtype=np.float64)
    elif pattern == "skewed":
        ranks = np.arange(1, E + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, 1.25)
    elif pattern == "one_dominant":
        if E == 1:
            weights = np.ones(1, dtype=np.float64)
        else:
            weights = np.full(E, 0.30 / (E - 1), dtype=np.float64)
            weights[0] = 0.70
    else:  # guarded by config validation
        raise ValueError(f"Unknown routing pattern: {pattern}")

    return weights / weights.sum()


def make_group_sizes(M: int, E: int, pattern: str) -> np.ndarray:
    """Allocate at least one token to every expert using largest remainders."""
    if M < E:
        raise ValueError(f"M={M} must be at least E={E}.")

    weights = routing_weights(E, pattern)
    remaining = M - E
    exact_extra = remaining * weights
    extra = np.floor(exact_extra).astype(np.int64)
    remainder = remaining - int(extra.sum())

    if remainder > 0:
        order = np.argsort(-(exact_extra - extra))
        extra[order[:remainder]] += 1

    group_sizes = (extra + 1).astype(np.int32)
    if int(group_sizes.sum()) != M:
        raise AssertionError("Internal routing allocation error.")
    return group_sizes


# Timing, correctness, and theoretical metrics
def benchmark_fn(
    fn: Callable[..., jax.Array],
    *args: jax.Array,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure compile+first call separately from steady-state calls."""
    start_ns = time.perf_counter_ns()
    output = fn(*args)
    jax.block_until_ready(output)
    first_call_ms = (time.perf_counter_ns() - start_ns) / 1e6

    for _ in range(warmup):
        output = fn(*args)
        jax.block_until_ready(output)

    times_ms: list[float] = []
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        output = fn(*args)
        jax.block_until_ready(output)
        times_ms.append((time.perf_counter_ns() - start_ns) / 1e6)

    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "first_call_ms": float(first_call_ms),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "std_ms": float(np.std(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "cv_percent": float(100.0 * np.std(values) / max(np.mean(values), 1e-12)),
        "times_ms": values.tolist(),
        "output": output,
    }


@jax.jit
def _correctness_metrics(y_ref: jax.Array, y_fused: jax.Array):
    difference = y_fused - y_ref
    absolute_error = jnp.abs(difference)
    reference_norm = jnp.linalg.norm(y_ref)
    difference_norm = jnp.linalg.norm(difference)
    max_reference = jnp.max(jnp.abs(y_ref))

    return (
        jnp.max(absolute_error),
        jnp.mean(absolute_error),
        difference_norm / jnp.maximum(reference_norm, 1e-12),
        jnp.max(absolute_error) / jnp.maximum(max_reference, 1e-12),
        jnp.all(jnp.isfinite(y_ref)),
        jnp.all(jnp.isfinite(y_fused)),
    )


def compute_correctness(
    y_ref: jax.Array,
    y_fused: jax.Array,
    *,
    l2_tolerance: float,
    normalized_max_tolerance: float,
) -> dict[str, Any]:
    values = _correctness_metrics(y_ref, y_fused)
    values = jax.device_get(values)

    max_abs, mean_abs, l2_rel, normalized_max, ref_finite, fused_finite = values
    acceptable = bool(
        ref_finite
        and fused_finite
        and float(l2_rel) < l2_tolerance
        and float(normalized_max) < normalized_max_tolerance
    )

    return {
        "max_absolute_error": float(max_abs),
        "mean_absolute_error": float(mean_abs),
        "l2_relative_error": float(l2_rel),
        "normalized_max_error": float(normalized_max),
        "reference_is_finite": bool(ref_finite),
        "fused_is_finite": bool(fused_finite),
        "numerically_acceptable": acceptable,
    }


def theoretical_metrics(
    *,
    M: int,
    E: int,
    D: int,
    F: int,
    BM: int,
    BF: int,
    group_sizes: Sequence[int],
    dtype_bytes: int,
    peak_compute_tflops: Optional[float],
    peak_bandwidth_gbps: Optional[float],
    shared_memory_limit_bytes: Optional[int],
) -> dict[str, Any]:
    groups = [int(v) for v in group_sizes]
    programs_per_expert = [math.ceil(group / BM) for group in groups]
    active_programs = int(sum(programs_per_expert))
    max_group_size = max(groups)
    max_programs = math.ceil(max_group_size / BM)
    launched_programs = E * max_programs
    inactive_programs = launched_programs - active_programs

    processed_token_slots = active_programs * BM
    padded_rows = processed_token_slots - M
    row_utilization = M / processed_token_slots
    grid_utilization = active_programs / launched_programs

    useful_flops = 4 * M * D * F
    actual_fused_flops = 4 * processed_token_slots * D * F
    padding_extra_flops = actual_fused_flops - useful_flops

    #   B_ref   = 2*b*(M*D + E*D*F + M*F)
    #   B_fused = 2*b*(M*D + E*D*F)
    output_bytes = 4
    reference_min_bytes = (
        dtype_bytes * M * D
        + 2 * dtype_bytes * E * D * F
        + 2 * output_bytes * M * F
        + output_bytes * M * D
    )
    fused_min_bytes = (
        dtype_bytes * M * D + 2 * dtype_bytes * E * D * F + output_bytes * M * D
    )
    hidden_activation_bytes = output_bytes * M * F
    hidden_write_read_saved_bytes = 2 * hidden_activation_bytes

    fused_no_cache_bytes = (
        dtype_bytes * M * D
        + 2 * dtype_bytes * active_programs * D * F
        + output_bytes * M * D
    )
    fused_tile_upper_bytes = active_programs * (
        (dtype_bytes + output_bytes) * BM * D + 2 * dtype_bytes * D * F
    )
    minimum_weight_bytes = 2 * dtype_bytes * E * D * F
    no_cache_weight_bytes = 2 * dtype_bytes * active_programs * D * F

    ideal_memory_speedup = reference_min_bytes / fused_min_bytes
    expansion_ratio = F / D
    fraction_of_asymptotic_benefit = M / (M + E * F)
    asymptotic_max_memory_speedup = 1 + expansion_ratio

    logical_tile_shared_bytes = dtype_bytes * (2 * BM * D + 2 * D * BF + BM * BF)
    inferred_one_extra_stage_shared_bytes = dtype_bytes * (
        2 * BM * D + 3 * D * BF + BM * BF
    )

    output: dict[str, Any] = {
        "max_group_size": max_group_size,
        "programs_per_expert": json.dumps(programs_per_expert),
        "active_programs_Q": active_programs,
        "launched_programs": launched_programs,
        "inactive_programs": inactive_programs,
        "grid_utilization": grid_utilization,
        "processed_token_slots": processed_token_slots,
        "padded_rows": padded_rows,
        "row_utilization": row_utilization,
        "padding_overhead_fraction": padded_rows / M,
        "padding_overhead_percent": 100.0 * padded_rows / M,
        "padding_overcompute_factor": processed_token_slots / M,
        "hidden_chunks_F_over_BF": F // BF,
        "useful_flops": useful_flops,
        "actual_fused_flops": actual_fused_flops,
        "padding_extra_flops": padding_extra_flops,
        "reference_min_bytes": reference_min_bytes,
        "fused_min_bytes": fused_min_bytes,
        "hidden_activation_bytes": hidden_activation_bytes,
        "hidden_intermediate_mib": hidden_activation_bytes / (1024**2),
        "hidden_write_read_saved_bytes": hidden_write_read_saved_bytes,
        "memory_saved_fraction": hidden_write_read_saved_bytes / reference_min_bytes,
        "memory_saved_percent": 100.0
        * hidden_write_read_saved_bytes
        / reference_min_bytes,
        "fused_no_cache_bytes": fused_no_cache_bytes,
        "fused_tile_upper_bytes": fused_tile_upper_bytes,
        "minimum_weight_bytes": minimum_weight_bytes,
        "no_cache_weight_bytes": no_cache_weight_bytes,
        "weight_reread_factor_no_cache": no_cache_weight_bytes / minimum_weight_bytes,
        "reference_ideal_ai_flops_per_byte": useful_flops / reference_min_bytes,
        "fused_ideal_effective_ai_flops_per_byte": useful_flops / fused_min_bytes,
        "fused_ideal_actual_ai_flops_per_byte": actual_fused_flops / fused_min_bytes,
        "fused_no_cache_actual_ai_flops_per_byte": actual_fused_flops
        / fused_no_cache_bytes,
        "fused_per_program_ai_flops_per_byte": (
            4 * BM * F / (BM * (dtype_bytes + output_bytes) + 2 * dtype_bytes * F)
        ),
        "fused_large_F_ai_approximation": 2 * BM / dtype_bytes,
        "expansion_ratio_F_over_D": expansion_ratio,
        "token_to_expert_hidden_ratio_M_over_EF": M / (E * F),
        "fraction_of_asymptotic_benefit": fraction_of_asymptotic_benefit,
        "fraction_of_asymptotic_benefit_percent": 100.0
        * fraction_of_asymptotic_benefit,
        "asymptotic_max_memory_speedup": asymptotic_max_memory_speedup,
        "ideal_memory_speedup": ideal_memory_speedup,
        "logical_tile_shared_bytes": logical_tile_shared_bytes,
        "inferred_one_extra_stage_shared_bytes": inferred_one_extra_stage_shared_bytes,
    }

    if shared_memory_limit_bytes is not None:
        output.update(
            {
                "shared_memory_limit_bytes": shared_memory_limit_bytes,
                "logical_shared_fraction_of_limit": (
                    logical_tile_shared_bytes / shared_memory_limit_bytes
                ),
                "inferred_shared_fraction_of_limit": (
                    inferred_one_extra_stage_shared_bytes / shared_memory_limit_bytes
                ),
            }
        )

    if peak_compute_tflops is not None and peak_bandwidth_gbps is not None:
        peak_flops_s = peak_compute_tflops * 1e12
        peak_bytes_s = peak_bandwidth_gbps * 1e9

        ref_compute_s = useful_flops / peak_flops_s
        ref_memory_s = reference_min_bytes / peak_bytes_s
        fused_compute_s = actual_fused_flops / peak_flops_s
        fused_ideal_memory_s = fused_min_bytes / peak_bytes_s
        fused_no_cache_memory_s = fused_no_cache_bytes / peak_bytes_s

        ref_roof_s = max(ref_compute_s, ref_memory_s)
        fused_ideal_roof_s = max(fused_compute_s, fused_ideal_memory_s)
        fused_no_cache_roof_s = max(fused_compute_s, fused_no_cache_memory_s)

        output.update(
            {
                "peak_compute_tflops": peak_compute_tflops,
                "peak_bandwidth_gbps": peak_bandwidth_gbps,
                "ridge_point_flops_per_byte": peak_flops_s / peak_bytes_s,
                "reference_roofline_ms": 1000.0 * ref_roof_s,
                "reference_roofline_limiter": (
                    "compute" if ref_compute_s >= ref_memory_s else "memory"
                ),
                "fused_ideal_roofline_ms": 1000.0 * fused_ideal_roof_s,
                "fused_ideal_roofline_limiter": (
                    "compute" if fused_compute_s >= fused_ideal_memory_s else "memory"
                ),
                "fused_no_cache_roofline_ms": 1000.0 * fused_no_cache_roof_s,
                "fused_no_cache_roofline_limiter": (
                    "compute"
                    if fused_compute_s >= fused_no_cache_memory_s
                    else "memory"
                ),
                "ideal_roofline_speedup": ref_roof_s / fused_ideal_roof_s,
                "no_cache_roofline_speedup": ref_roof_s / fused_no_cache_roof_s,
            }
        )

    return output


def estimated_problem_memory_bytes(
    *, M: int, D: int, F: int, E: int, dtype_bytes: int
) -> int:
    """Conservative live-memory estimate for preflight skipping."""
    x = M * D * dtype_bytes
    weights = 2 * E * D * F * dtype_bytes
    hidden = M * F * 4  # reference accumulation is float32
    two_outputs = 2 * M * D * 4
    group_metadata = (E + 1) * 4
    subtotal = x + weights + hidden + two_outputs + group_metadata
    return int(1.35 * subtotal)


# Persistence and environment metadata
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, default=_json_default)
    os.replace(temporary, path)


def atomic_write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def append_failure(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    atomic_write_csv(updated, path)


def environment_metadata() -> dict[str, Any]:
    devices = []
    for device in jax.devices():
        devices.append(
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
                "process_index": getattr(device, "process_index", None),
            }
        )

    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
        "jax_backend": jax.default_backend(),
        "devices": devices,
    }


def stable_case_id(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


# Main experiment loop
def dtype_from_name(name: str):
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[name]


def run_experiment(config: SweepConfig, output_dir: str | Path) -> pd.DataFrame:
    config.validate()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "results.csv"
    failures_path = output_path / "failures.csv"

    write_json(output_path / "config.json", dataclasses.asdict(config))
    write_json(output_path / "environment.json", environment_metadata())

    if config.interpret:
        print(
            "WARNING: interpret=True executes a simulation/debug path. "
            "Its timings are not meaningful hardware kernel performance."
        )

    if config.resume and results_path.exists():
        results = pd.read_csv(results_path)
        completed_ids = set(results.get("case_id", pd.Series(dtype=str)).astype(str))
        print(f"Resuming with {len(completed_ids):,} completed fused cases.")
    else:
        results = pd.DataFrame()
        completed_ids: set[str] = set()

    shapes = build_problem_shapes(config)
    tile_pairs = build_tile_pairs(config)

    rng = random.Random(config.seed)
    if config.shuffle_cases:
        rng.shuffle(shapes)

    planned = len(shapes) * len(tile_pairs)
    if config.max_cases is not None:
        planned = min(planned, config.max_cases)

    print("=" * 88)
    print("RAGGED MoE FUSED-KERNEL SPEEDUP EXPERIMENT")
    print("=" * 88)
    print(f"Backend:              {jax.default_backend()}")
    print(f"Problem shapes:       {len(shapes):,}")
    print(f"Tile pairs/shape:     {len(tile_pairs):,}")
    print(f"Maximum fused cases:  {planned:,}")
    print(f"Output directory:     {output_path.resolve()}")
    print("=" * 88)

    input_dtype = dtype_from_name(config.input_dtype)
    dtype_bytes = int(jnp.dtype(input_dtype).itemsize)
    executed_new_cases = 0

    for shape_index, (M, D, F, E, group_pattern) in enumerate(shapes, start=1):
        if config.max_cases is not None and executed_new_cases >= config.max_cases:
            break

        shape_info = {
            "M": M,
            "D": D,
            "F": F,
            "E": E,
            "group_pattern": group_pattern,
            "input_dtype": config.input_dtype,
        }

        group_sizes_host = make_group_sizes(M, E, group_pattern)
        max_group_size = int(group_sizes_host.max())
        estimated_bytes = estimated_problem_memory_bytes(
            M=M,
            D=D,
            F=F,
            E=E,
            dtype_bytes=dtype_bytes,
        )
        estimated_gib = estimated_bytes / (1024**3)

        if (
            config.max_estimated_problem_memory_gib is not None
            and estimated_gib > config.max_estimated_problem_memory_gib
        ):
            failure = {
                **shape_info,
                "BM": None,
                "BF": None,
                "stage": "preflight",
                "error_type": "EstimatedMemoryLimit",
                "error_message": (
                    f"Estimated {estimated_gib:.3f} GiB exceeds configured "
                    f"limit {config.max_estimated_problem_memory_gib:.3f} GiB."
                ),
                "traceback": "",
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            append_failure(failures_path, failure)
            print(
                f"[{shape_index}/{len(shapes)}] SKIP shape {shape_info}: "
                f"estimated {estimated_gib:.2f} GiB"
            )
            continue

        pending_tile_pairs: list[tuple[int, int, str]] = []
        for BM, BF in tile_pairs:
            identity = {
                **shape_info,
                "BM": BM,
                "BF": BF,
                "group_sizes": [int(v) for v in group_sizes_host],
                "seed": config.seed,
                "warmup": config.warmup,
                "iterations": config.iterations,
                "interpret": config.interpret,
            }
            case_id = stable_case_id(identity)
            if case_id not in completed_ids:
                pending_tile_pairs.append((BM, BF, case_id))

        if not pending_tile_pairs:
            print(f"[{shape_index}/{len(shapes)}] Already complete: {shape_info}")
            continue

        if config.shuffle_cases:
            rng.shuffle(pending_tile_pairs)

        print("\n" + "-" * 88)
        print(
            f"[{shape_index}/{len(shapes)}] M={M:,}, D={D}, F={F}, E={E}, "
            f"routing={group_pattern}, groups={group_sizes_host.tolist()}, "
            f"estimated_memory={estimated_gib:.3f} GiB"
        )

        key = jax.random.fold_in(jax.random.PRNGKey(config.seed), shape_index)
        key_x, key_up, key_down = jax.random.split(key, 3)

        try:
            x = jax.random.normal(key_x, (M, D), dtype=input_dtype)
            w_up = jax.random.normal(key_up, (E, D, F), dtype=input_dtype) / jnp.sqrt(D)
            w_down = jax.random.normal(
                key_down, (E, F, D), dtype=input_dtype
            ) / jnp.sqrt(F)
            group_sizes = jnp.asarray(group_sizes_host, dtype=jnp.int32)

            print("  Benchmarking reference once for this problem shape...")
            ref_stats = benchmark_fn(
                REFERENCE_JIT,
                x,
                w_up,
                w_down,
                group_sizes,
                warmup=config.warmup,
                iterations=config.iterations,
            )
            y_ref = ref_stats["output"]
        except Exception as error:
            append_failure(
                failures_path,
                {
                    **shape_info,
                    "BM": None,
                    "BF": None,
                    "stage": "reference",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            print(f"  REFERENCE FAILED: {type(error).__name__}: {error}")
            if config.clear_jax_caches_between_problem_shapes:
                jax.clear_caches()
            gc.collect()
            continue

        for BM, BF, case_id in pending_tile_pairs:
            if config.max_cases is not None and executed_new_cases >= config.max_cases:
                break

            if F % BF != 0:
                append_failure(
                    failures_path,
                    {
                        **shape_info,
                        "BM": BM,
                        "BF": BF,
                        "stage": "validation",
                        "error_type": "InvalidTileShape",
                        "error_message": f"F={F} is not divisible by BF={BF}.",
                        "traceback": "",
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
                print(f"  BM={BM:>3}, BF={BF:>3}: skipped because F % BF != 0")
                continue

            try:
                fused_jit = make_fused_jit(
                    BM=BM,
                    BF=BF,
                    max_group_size=max_group_size,
                    interpret=config.interpret,
                )
                fused_stats = benchmark_fn(
                    fused_jit,
                    x,
                    w_up,
                    w_down,
                    group_sizes,
                    warmup=config.warmup,
                    iterations=config.iterations,
                )
                y_fused = fused_stats["output"]

                correctness = compute_correctness(
                    y_ref,
                    y_fused,
                    l2_tolerance=config.l2_relative_error_tolerance,
                    normalized_max_tolerance=config.normalized_max_error_tolerance,
                )

                theory = theoretical_metrics(
                    M=M,
                    E=E,
                    D=D,
                    F=F,
                    BM=BM,
                    BF=BF,
                    group_sizes=group_sizes_host,
                    dtype_bytes=dtype_bytes,
                    peak_compute_tflops=config.peak_compute_tflops,
                    peak_bandwidth_gbps=config.peak_bandwidth_gbps,
                    shared_memory_limit_bytes=config.shared_memory_limit_bytes,
                )

                ref_ms = ref_stats["median_ms"]
                fused_ms = fused_stats["median_ms"]
                speedup = ref_ms / fused_ms
                ideal_speedup = theory["ideal_memory_speedup"]
                predicted_fused_ms = ref_ms / ideal_speedup

                row: dict[str, Any] = {
                    "case_id": case_id,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "backend": jax.default_backend(),
                    "input_dtype": config.input_dtype,
                    "M": M,
                    "D": D,
                    "F": F,
                    "E": E,
                    "BM": BM,
                    "BF": BF,
                    "group_pattern": group_pattern,
                    "group_sizes": json.dumps([int(v) for v in group_sizes_host]),
                    "estimated_problem_memory_gib": estimated_gib,
                    **theory,
                    **correctness,
                    "ref_compile_plus_first_ms": ref_stats["first_call_ms"],
                    "ref_mean_ms": ref_stats["mean_ms"],
                    "ref_median_ms": ref_stats["median_ms"],
                    "ref_min_ms": ref_stats["min_ms"],
                    "ref_max_ms": ref_stats["max_ms"],
                    "ref_std_ms": ref_stats["std_ms"],
                    "ref_p10_ms": ref_stats["p10_ms"],
                    "ref_p90_ms": ref_stats["p90_ms"],
                    "ref_cv_percent": ref_stats["cv_percent"],
                    "ref_run_times_ms": json.dumps(ref_stats["times_ms"]),
                    "fused_compile_plus_first_ms": fused_stats["first_call_ms"],
                    "fused_mean_ms": fused_stats["mean_ms"],
                    "fused_median_ms": fused_stats["median_ms"],
                    "fused_min_ms": fused_stats["min_ms"],
                    "fused_max_ms": fused_stats["max_ms"],
                    "fused_std_ms": fused_stats["std_ms"],
                    "fused_p10_ms": fused_stats["p10_ms"],
                    "fused_p90_ms": fused_stats["p90_ms"],
                    "fused_cv_percent": fused_stats["cv_percent"],
                    "fused_run_times_ms": json.dumps(fused_stats["times_ms"]),
                    "speedup_ref_over_fused": speedup,
                    "predicted_fused_ms_from_memory_model": predicted_fused_ms,
                    "fused_vs_predicted_latency_ratio": fused_ms / predicted_fused_ms,
                    "prediction_error_percent": 100.0
                    * (fused_ms - predicted_fused_ms)
                    / predicted_fused_ms,
                    "speedup_as_fraction_of_ideal": speedup / ideal_speedup,
                    "speedup_percent_of_ideal": 100.0 * speedup / ideal_speedup,
                    "speedup_gap_from_ideal": ideal_speedup - speedup,
                    "incremental_fusion_benefit_fraction": (
                        (speedup - 1.0) / (ideal_speedup - 1.0)
                        if ideal_speedup > 1.0
                        else float("nan")
                    ),
                    "ref_tokens_per_second": M / (ref_ms / 1000.0),
                    "fused_tokens_per_second": M / (fused_ms / 1000.0),
                    "ref_effective_tflops": (
                        theory["useful_flops"] / (ref_ms / 1000.0) / 1e12
                    ),
                    "fused_effective_tflops": (
                        theory["useful_flops"] / (fused_ms / 1000.0) / 1e12
                    ),
                    "fused_actual_tflops": (
                        theory["actual_fused_flops"] / (fused_ms / 1000.0) / 1e12
                    ),
                    "ref_minimum_traffic_gbps": (
                        theory["reference_min_bytes"] / (ref_ms / 1000.0) / 1e9
                    ),
                    "fused_minimum_traffic_gbps": (
                        theory["fused_min_bytes"] / (fused_ms / 1000.0) / 1e9
                    ),
                    "fused_no_cache_traffic_gbps": (
                        theory["fused_no_cache_bytes"] / (fused_ms / 1000.0) / 1e9
                    ),
                }

                if "reference_roofline_ms" in theory:
                    row.update(
                        {
                            "reference_roofline_efficiency": (
                                theory["reference_roofline_ms"] / ref_ms
                            ),
                            "fused_ideal_roofline_efficiency": (
                                theory["fused_ideal_roofline_ms"] / fused_ms
                            ),
                            "fused_no_cache_roofline_efficiency": (
                                theory["fused_no_cache_roofline_ms"] / fused_ms
                            ),
                        }
                    )

                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                atomic_write_csv(results, results_path)
                completed_ids.add(case_id)
                executed_new_cases += 1

                validity = (
                    "valid" if correctness["numerically_acceptable"] else "INVALID"
                )
                print(
                    f"  BM={BM:>3}, BF={BF:>3} | "
                    f"ref={ref_ms:>9.4f} ms | fused={fused_ms:>9.4f} ms | "
                    f"speedup={speedup:>7.3f}x | ideal={ideal_speedup:>7.3f}x | "
                    f"ideal-captured={100.0 * speedup / ideal_speedup:>7.2f}% | "
                    f"L2={correctness['l2_relative_error']:.3e} | {validity}"
                )

                del y_fused, fused_jit

            except Exception as error:
                append_failure(
                    failures_path,
                    {
                        **shape_info,
                        "BM": BM,
                        "BF": BF,
                        "stage": "fused",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
                print(
                    f"  BM={BM:>3}, BF={BF:>3}: FAILED {type(error).__name__}: {error}"
                )

        del x, w_up, w_down, group_sizes, y_ref
        gc.collect()
        if config.clear_jax_caches_between_problem_shapes:
            jax.clear_caches()

    if results.empty:
        print("No successful results were produced. Check failures.csv.")
        return results

    create_summary_tables(results, output_path)
    create_all_plots(
        results,
        output_path / "plots",
        max_by_shape_plots=config.max_by_shape_plots,
    )

    print("\n" + "=" * 88)
    print(f"Completed successful cases: {len(results):,}")
    print(f"New cases this invocation:  {executed_new_cases:,}")
    print(f"Results:                    {results_path.resolve()}")
    print(f"Plots:                      {(output_path / 'plots').resolve()}")
    if failures_path.exists():
        print(f"Failures/skips:             {failures_path.resolve()}")
    print("=" * 88)
    return results


# Summary tables
def create_summary_tables(results: pd.DataFrame, output_dir: Path) -> None:
    valid = results[results["numerically_acceptable"].astype(bool)].copy()
    if valid.empty:
        valid = results.copy()

    problem_columns = ["M", "D", "F", "E", "group_pattern"]
    best_indices = valid.groupby(problem_columns)["speedup_ref_over_fused"].idxmax()
    best = valid.loc[best_indices].sort_values(problem_columns)
    atomic_write_csv(best, output_dir / "best_configs.csv")

    for parameter in ("M", "D", "F", "E", "BM", "BF"):
        summary = (
            valid.groupby(parameter, as_index=False)
            .agg(
                case_count=("case_id", "count"),
                median_measured_speedup=("speedup_ref_over_fused", "median"),
                mean_measured_speedup=("speedup_ref_over_fused", "mean"),
                min_measured_speedup=("speedup_ref_over_fused", "min"),
                max_measured_speedup=("speedup_ref_over_fused", "max"),
                median_ideal_speedup=("ideal_memory_speedup", "median"),
                median_speedup_percent_of_ideal=("speedup_percent_of_ideal", "median"),
                median_prediction_error_percent=("prediction_error_percent", "median"),
                median_fused_ms=("fused_median_ms", "median"),
                median_ref_ms=("ref_median_ms", "median"),
            )
            .sort_values(parameter)
        )
        atomic_write_csv(summary, output_dir / f"summary_by_{parameter}.csv")


# Plotting
def _finish_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite_frame(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    mask = np.ones(len(data), dtype=bool)
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy()
        mask &= np.isfinite(values)
    return data.loc[mask].copy()


def create_all_plots(
    results: pd.DataFrame,
    plots_dir: Path,
    *,
    max_by_shape_plots: int,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    valid = results[results["numerically_acceptable"].astype(bool)].copy()
    if valid.empty:
        valid = results.copy()

    # 1. Measured versus mathematical ideal speedup.
    data = _finite_frame(valid, ["ideal_memory_speedup", "speedup_ref_over_fused"])
    if not data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            data["ideal_memory_speedup"], data["speedup_ref_over_fused"], alpha=0.65
        )
        lower = min(
            data["ideal_memory_speedup"].min(), data["speedup_ref_over_fused"].min()
        )
        upper = max(
            data["ideal_memory_speedup"].max(), data["speedup_ref_over_fused"].max()
        )
        ax.plot(
            [lower, upper], [lower, upper], linestyle="--", label="Measured = ideal"
        )
        ax.set_xlabel("Ideal bandwidth-model speedup (x)")
        ax.set_ylabel("Measured speedup (x)")
        ax.set_title("Measured versus mathematically predicted speedup")
        ax.grid(True, alpha=0.25)
        ax.legend()
        _finish_plot(fig, plots_dir / "measured_vs_ideal_speedup.png")

    # 2. Reference versus fused latency.
    data = _finite_frame(valid, ["ref_median_ms", "fused_median_ms"])
    data = data[(data["ref_median_ms"] > 0) & (data["fused_median_ms"] > 0)]
    if not data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data["ref_median_ms"], data["fused_median_ms"], alpha=0.65)
        lower = min(data["ref_median_ms"].min(), data["fused_median_ms"].min())
        upper = max(data["ref_median_ms"].max(), data["fused_median_ms"].max())
        ax.plot([lower, upper], [lower, upper], linestyle="--", label="Equal latency")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reference median latency (ms)")
        ax.set_ylabel("Fused median latency (ms)")
        ax.set_title("Reference versus fused latency")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        _finish_plot(fig, plots_dir / "reference_vs_fused_latency.png")

    # 3. Median speedup and ideal speedup by M.
    summary_m = (
        valid.groupby("M", as_index=False)
        .agg(
            measured_median=("speedup_ref_over_fused", "median"),
            measured_q25=("speedup_ref_over_fused", lambda x: x.quantile(0.25)),
            measured_q75=("speedup_ref_over_fused", lambda x: x.quantile(0.75)),
            ideal_median=("ideal_memory_speedup", "median"),
        )
        .sort_values("M")
    )
    if not summary_m.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(
            summary_m["M"],
            summary_m["measured_median"],
            marker="o",
            label="Measured median",
        )
        ax.fill_between(
            summary_m["M"],
            summary_m["measured_q25"],
            summary_m["measured_q75"],
            alpha=0.2,
            label="Measured IQR",
        )
        ax.plot(
            summary_m["M"],
            summary_m["ideal_median"],
            marker="s",
            linestyle="--",
            label="Ideal median",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Total routed tokens M")
        ax.set_ylabel("Speedup (x)")
        ax.set_title("Speedup scaling with token count")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        _finish_plot(fig, plots_dir / "speedup_vs_M_summary.png")

    # 4. Speedup versus eliminated hidden intermediate size.
    data = _finite_frame(valid, ["hidden_intermediate_mib", "speedup_ref_over_fused"])
    data = data[data["hidden_intermediate_mib"] > 0]
    if not data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            data["hidden_intermediate_mib"], data["speedup_ref_over_fused"], alpha=0.65
        )
        ax.set_xscale("log")
        ax.set_xlabel("Hidden intermediate H size (MiB)")
        ax.set_ylabel("Measured speedup (x)")
        ax.set_title("Fusion benefit versus avoided hidden activation")
        ax.grid(True, which="both", alpha=0.25)
        _finish_plot(fig, plots_dir / "speedup_vs_hidden_intermediate.png")

    # 5. Padding overhead versus speedup.
    data = _finite_frame(valid, ["padding_overhead_percent", "speedup_ref_over_fused"])
    if not data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            data["padding_overhead_percent"], data["speedup_ref_over_fused"], alpha=0.65
        )
        ax.set_xlabel("Padding FLOP overhead (%)")
        ax.set_ylabel("Measured speedup (x)")
        ax.set_title("Effect of ragged tile padding on speedup")
        ax.grid(True, alpha=0.25)
        _finish_plot(fig, plots_dir / "speedup_vs_padding_overhead.png")

    # 6. Prediction error versus M.
    data = _finite_frame(valid, ["M", "prediction_error_percent"])
    if not data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data["M"], data["prediction_error_percent"], alpha=0.65)
        ax.axhline(0.0, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Total routed tokens M")
        ax.set_ylabel("Fused latency prediction error (%)")
        ax.set_title("Bandwidth-model error versus token count")
        ax.grid(True, which="both", alpha=0.25)
        _finish_plot(fig, plots_dir / "prediction_error_vs_M.png")

    # 7. BM/BF median speedup heatmap.
    pivot = valid.pivot_table(
        index="BM",
        columns="BF",
        values="speedup_ref_over_fused",
        aggfunc="median",
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)), labels=[str(v) for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=[str(v) for v in pivot.index])
        ax.set_xlabel("BF hidden tile")
        ax.set_ylabel("BM token tile")
        ax.set_title("Median measured speedup by BM and BF")
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                value = pivot.iloc[row_index, column_index]
                if np.isfinite(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}x",
                        ha="center",
                        va="center",
                    )
        fig.colorbar(image, ax=ax, label="Median speedup (x)")
        _finish_plot(fig, plots_dir / "median_speedup_BM_BF_heatmap.png")

    # 8. Expert-count speedup distribution.
    expert_values = sorted(valid["E"].dropna().unique())
    if expert_values:
        groups = [
            valid.loc[valid["E"] == expert, "speedup_ref_over_fused"]
            .dropna()
            .to_numpy()
            for expert in expert_values
        ]
        groups = [group for group in groups if len(group) > 0]
        if groups:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(
                groups, tick_labels=[str(v) for v in expert_values], showfliers=True
            )
            ax.axhline(1.0, linestyle="--")
            ax.set_xlabel("Number of experts E")
            ax.set_ylabel("Measured speedup (x)")
            ax.set_title("Speedup distribution by expert count")
            ax.grid(True, axis="y", alpha=0.25)
            _finish_plot(fig, plots_dir / "speedup_by_expert_count.png")

    # 9. Correctness error by case.
    data = _finite_frame(results, ["l2_relative_error"])
    data = data[data["l2_relative_error"] > 0].reset_index(drop=True)
    if not data.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(np.arange(len(data)), data["l2_relative_error"], alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("Completed case index")
        ax.set_ylabel("L2 relative error")
        ax.set_title("Numerical correctness across experiment cases")
        ax.grid(True, which="both", alpha=0.25)
        _finish_plot(fig, plots_dir / "correctness_l2_error.png")

    create_by_shape_plots(valid, plots_dir / "by_shape", max_plots=max_by_shape_plots)


def create_by_shape_plots(
    data: pd.DataFrame, output_dir: Path, *, max_plots: int
) -> None:
    """For each D/F/E/routing family, plot M curves for every BM/BF pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    group_columns = ["D", "F", "E", "group_pattern"]
    plotted = 0

    for group_key, group in data.groupby(group_columns):
        if plotted >= max_plots:
            break
        if group["M"].nunique() < 2:
            continue

        D, F, E, pattern = group_key
        fig, ax = plt.subplots(figsize=(10, 6))

        for (BM, BF), tile_group in group.groupby(["BM", "BF"]):
            tile_group = tile_group.sort_values("M")
            ax.plot(
                tile_group["M"],
                tile_group["speedup_ref_over_fused"],
                marker="o",
                label=f"BM={BM}, BF={BF}",
            )

        ideal = (
            group.groupby("M", as_index=False)["ideal_memory_speedup"]
            .median()
            .sort_values("M")
        )
        ax.plot(
            ideal["M"],
            ideal["ideal_memory_speedup"],
            linestyle="--",
            linewidth=2.0,
            label="Ideal bandwidth model",
        )

        ax.axhline(1.0, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Total routed tokens M")
        ax.set_ylabel("Speedup (x)")
        ax.set_title(f"D={D}, F={F}, E={E}, routing={pattern}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)

        filename = f"D{D}_F{F}_E{E}_{pattern}.png"
        _finish_plot(fig, output_dir / filename)
        plotted += 1


# CLI
def parse_integer_list(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_string_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="ragged_moe_results")
    parser.add_argument("--mode", choices=["structured", "full_grid", "shapes_only"])
    parser.add_argument("--m-values", type=parse_integer_list)
    parser.add_argument("--d-values", type=parse_integer_list)
    parser.add_argument("--f-values", type=parse_integer_list)
    parser.add_argument("--e-values", type=parse_integer_list)
    parser.add_argument("--bm-values", type=parse_integer_list)
    parser.add_argument("--bf-values", type=parse_integer_list)
    parser.add_argument("--group-patterns", type=parse_string_list)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-memory-gib", type=float)
    parser.add_argument("--peak-compute-tflops", type=float)
    parser.add_argument("--peak-bandwidth-gbps", type=float)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--keep-jax-caches", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    values = dataclasses.asdict(DEFAULT_CONFIG)
    mapping = {
        "mode": "mode",
        "m_values": "M_values",
        "d_values": "D_values",
        "f_values": "F_values",
        "e_values": "E_values",
        "bm_values": "BM_values",
        "bf_values": "BF_values",
        "group_patterns": "group_patterns",
        "warmup": "warmup",
        "iterations": "iterations",
        "seed": "seed",
        "max_cases": "max_cases",
        "max_memory_gib": "max_estimated_problem_memory_gib",
        "peak_compute_tflops": "peak_compute_tflops",
        "peak_bandwidth_gbps": "peak_bandwidth_gbps",
    }

    for arg_name, config_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            values[config_name] = value

    if args.interpret:
        values["interpret"] = True
    if args.no_resume:
        values["resume"] = False
    if args.no_shuffle:
        values["shuffle_cases"] = False
    if args.keep_jax_caches:
        values["clear_jax_caches_between_problem_shapes"] = False

    # dataclasses.asdict turns tuples into tuples, but normalize explicit lists too.
    for name in (
        "M_values",
        "D_values",
        "F_values",
        "E_values",
        "BM_values",
        "BF_values",
        "group_patterns",
    ):
        values[name] = tuple(values[name])

    return SweepConfig(**values)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
