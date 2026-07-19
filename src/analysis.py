import math
import json
from pathlib import Path
from typing import Optional, Sequence, Any

import pandas as pd


def _format_bytes(num_bytes: float) -> str:
    """Format a byte count using binary units."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:,.3f} {unit}"
        value /= 1024.0

    return f"{value:,.3f} TiB"


def _format_flops(num_flops: float) -> str:
    """Format a FLOP count."""
    units = [
        ("EFLOP", 1e18),
        ("PFLOP", 1e15),
        ("TFLOP", 1e12),
        ("GFLOP", 1e9),
        ("MFLOP", 1e6),
        ("KFLOP", 1e3),
    ]

    for name, scale in units:
        if abs(num_flops) >= scale:
            return f"{num_flops / scale:,.3f} {name}"

    return f"{num_flops:,.0f} FLOP"


def _format_seconds(seconds: float) -> str:
    """Format a latency."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:,.3f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:,.3f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:,.3f} ms"
    return f"{seconds:,.6f} s"


def analyze_ragged_moe(
    *,
    M: int,
    E: int,
    D: int,
    F: int,
    BM: int,
    BF: int,
    group_sizes: Sequence[int],
    dtype_bytes: int = 4,
    ref_ms: Optional[float] = None,
    fused_ms: Optional[float] = None,
    measured_speedup: Optional[float] = None,
    peak_compute_tflops: Optional[float] = None,
    peak_bandwidth_gbps: Optional[float] = None,
    compiled_shared_bytes: Optional[int] = None,
    shared_memory_limit_bytes: Optional[int] = None,
    print_results: bool = True,
) -> dict[str, Any]:
    """
    Analyze a fused ragged MoE MLP forward pass.

    Mathematical operation
    ----------------------
        H = GELU(ragged_dot(X, W_up, group_sizes))
        Y = ragged_dot(H, W_down, group_sizes)

    Shapes
    ------
        X:      (M, D)
        W_up:   (E, D, F)
        W_down: (E, F, D)
        Y:      (M, D)

    Important assumptions
    ---------------------
    1. One multiply-add counts as 2 FLOPs.
    2. GELU FLOPs are excluded from the standard 4*M*D*F count.
    3. The minimum traffic model assumes each expert's weights are fetched
       from HBM only once and reused perfectly afterward.
    4. The no-cache fused model assumes each active Pallas program fetches
       its expert's complete up/down weights independently.
    5. Shared-memory formulas are source-level estimates. The compiler may
       allocate additional staging, pipelining, layout, or scratch buffers.
    """

    # Validation
    integer_values = {
        "M": M,
        "E": E,
        "D": D,
        "F": F,
        "BM": BM,
        "BF": BF,
        "dtype_bytes": dtype_bytes,
    }

    for name, value in integer_values.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")

    groups = [int(value) for value in group_sizes]

    if len(groups) != E:
        raise ValueError(
            f"group_sizes must contain E={E} values, but received {len(groups)}."
        )

    if any(value < 0 for value in groups):
        raise ValueError("group_sizes cannot contain negative values.")

    if sum(groups) != M:
        raise ValueError(f"group_sizes sum to {sum(groups)}, but M={M}.")

    if F % BF != 0:
        raise ValueError(f"F={F} must be divisible by BF={BF} for the current kernel.")

    if (ref_ms is None) != (fused_ms is None):
        raise ValueError("Provide both ref_ms and fused_ms, or provide neither.")

    if ref_ms is not None and (ref_ms <= 0 or fused_ms <= 0):
        raise ValueError("Measured latencies must be positive.")

    if measured_speedup is not None and measured_speedup <= 0:
        raise ValueError("measured_speedup must be positive.")

    if peak_compute_tflops is not None and peak_compute_tflops <= 0:
        raise ValueError("peak_compute_tflops must be positive.")

    if peak_bandwidth_gbps is not None and peak_bandwidth_gbps <= 0:
        raise ValueError("peak_bandwidth_gbps must be positive.")

    # Grid and routing
    programs_per_expert = [math.ceil(group_size / BM) for group_size in groups]

    padded_rows_per_expert = [
        programs * BM - group_size
        for programs, group_size in zip(programs_per_expert, groups)
    ]

    active_programs = sum(programs_per_expert)

    max_group_size = max(groups)
    max_programs_per_expert = math.ceil(max_group_size / BM)

    # The current rectangular grid is:
    #   (E, ceil(max(group_sizes) / BM))
    launched_programs = E * max_programs_per_expert
    inactive_programs = launched_programs - active_programs

    processed_token_slots = active_programs * BM
    padded_rows = processed_token_slots - M

    row_utilization = M / processed_token_slots
    padding_overcompute_factor = processed_token_slots / M
    padding_overhead_fraction = padded_rows / M

    grid_utilization = (
        active_programs / launched_programs if launched_programs > 0 else 0.0
    )

    hidden_chunks = F // BF

    # FLOPs

    # Up projection:
    #   2*M*D*F
    #
    # Down projection:
    #   2*M*F*D
    #
    # Total useful:
    #   4*M*D*F
    useful_flops = 4 * M * D * F

    # Every active program executes a BM-row tile.
    actual_fused_flops = 4 * processed_token_slots * D * F
    padding_extra_flops = actual_fused_flops - useful_flops

    # Ideal global-memory traffic
    # Reference:
    #   read X                 = b*M*D
    #   read W_up              = b*E*D*F
    #   write H                = b*M*F
    #   read H                 = b*M*F
    #   read W_down            = b*E*F*D
    #   write Y                = b*M*D
    #
    # Total:
    #   2*b*(M*D + E*D*F + M*F)
    reference_min_bytes = 2 * dtype_bytes * (M * D + E * D * F + M * F)

    # Fused minimum:
    #   read X
    #   read each expert's W_up and W_down once
    #   write Y
    #
    # Total:
    #   2*b*(M*D + E*D*F)
    fused_min_bytes = 2 * dtype_bytes * (M * D + E * D * F)

    # Full hidden activation write + read eliminated by fusion
    hidden_activation_bytes = dtype_bytes * M * F
    hidden_write_read_saved_bytes = 2 * hidden_activation_bytes

    memory_saved_fraction = hidden_write_read_saved_bytes / reference_min_bytes

    # Fused no-cache traffic estimate
    # Each active program streams one expert's complete W_up and W_down:
    #
    #   weight bytes per program = 2*b*D*F
    #
    # Exact useful X/Y traffic:
    #
    #   2*b*M*D
    fused_no_cache_bytes = (
        2 * dtype_bytes * M * D + 2 * dtype_bytes * active_programs * D * F
    )

    # A tile-level upper estimate that also counts padded X/Y row slots:
    #
    #   2*b*D*Q*(BM + F)
    fused_tile_upper_bytes = 2 * dtype_bytes * D * active_programs * (BM + F)

    minimum_weight_bytes = 2 * dtype_bytes * E * D * F
    no_cache_weight_bytes = 2 * dtype_bytes * active_programs * D * F

    weight_reread_factor_no_cache = no_cache_weight_bytes / minimum_weight_bytes

    # Arithmetic intensity
    # Effective AI uses useful algorithmic FLOPs.
    reference_ideal_ai = useful_flops / reference_min_bytes
    fused_ideal_effective_ai = useful_flops / fused_min_bytes

    # Actual fused AI includes padded-row FLOPs.
    fused_ideal_actual_ai = actual_fused_flops / fused_min_bytes
    fused_no_cache_actual_ai = actual_fused_flops / fused_no_cache_bytes

    # Per-program tile-level approximation:
    #
    #   FLOPs = 4*BM*D*F
    #   bytes = 2*b*D*(BM + F)
    #   AI = 2*BM*F / [b*(BM + F)]
    fused_per_program_ai = 2 * BM * F / (dtype_bytes * (BM + F))

    # When F >> BM:
    #
    #   AI ≈ 2*BM/b
    fused_large_F_ai_approximation = 2 * BM / dtype_bytes

    # Ideal speedup model
    # Pure bandwidth speedup:
    #
    #   S = B_ref / B_fused
    #
    #     = 1 + MF / [D(M + EF)]
    ideal_memory_speedup = reference_min_bytes / fused_min_bytes

    expansion_ratio = F / D

    # lambda = M / (E*F)
    token_to_expert_hidden_ratio = M / (E * F)

    fraction_of_asymptotic_benefit = M / (M + E * F)

    # As M -> infinity:
    #
    #   S_max = 1 + F/D
    asymptotic_max_memory_speedup = 1 + expansion_ratio

    # Equivalent alternate expression:
    #
    #   S = 1 + (F/D) * M/(M + EF)
    ideal_memory_speedup_alternate = (
        1 + expansion_ratio * fraction_of_asymptotic_benefit
    )

    # Shared-memory source-level estimates
    # Current kernel has no separate BD. It computes all D outputs.
    #
    # Logical values:
    #   X tile:       BM*D
    #   accumulator:  BM*D
    #   W_up tile:    D*BF
    #   W_down tile:  BF*D
    #   H tile:       BM*BF
    #
    # Assuming all values use dtype_bytes:
    logical_tile_shared_bytes = dtype_bytes * (2 * BM * D + 2 * D * BF + BM * BF)

    inferred_one_extra_stage_shared_bytes = dtype_bytes * (
        2 * BM * D + 3 * D * BF + BM * BF
    )

    # Measured performance
    timing_results: dict[str, Any] = {}

    if ref_ms is not None and fused_ms is not None:
        ref_seconds = ref_ms / 1000.0
        fused_seconds = fused_ms / 1000.0

        timing_speedup = ref_ms / fused_ms

        if measured_speedup is not None:
            speedup_difference = abs(timing_speedup - measured_speedup)

            if speedup_difference > 1e-3:
                raise ValueError(
                    "measured_speedup disagrees with ref_ms / fused_ms: "
                    f"{measured_speedup:.6f} versus {timing_speedup:.6f}."
                )

        measured_speedup = timing_speedup

        timing_results.update(
            {
                "ref_ms": ref_ms,
                "fused_ms": fused_ms,
                "measured_speedup": measured_speedup,
                "ref_tokens_per_second": M / ref_seconds,
                "fused_tokens_per_second": M / fused_seconds,
                "ref_effective_tflops": useful_flops / ref_seconds / 1e12,
                "fused_effective_tflops": useful_flops / fused_seconds / 1e12,
                "fused_actual_tflops": actual_fused_flops / fused_seconds / 1e12,
                # Estimated bandwidth based on different byte models.
                "ref_minimum_traffic_gbps": reference_min_bytes / ref_seconds / 1e9,
                "fused_minimum_traffic_gbps": fused_min_bytes / fused_seconds / 1e9,
                "fused_no_cache_traffic_gbps": fused_no_cache_bytes
                / fused_seconds
                / 1e9,
            }
        )

    elif measured_speedup is not None:
        timing_results["measured_speedup"] = measured_speedup

    if measured_speedup is not None:
        timing_results.update(
            {
                "speedup_as_fraction_of_ideal": measured_speedup / ideal_memory_speedup,
                "speedup_percent_of_ideal": 100.0
                * measured_speedup
                / ideal_memory_speedup,
                "speedup_gap_from_ideal": ideal_memory_speedup - measured_speedup,
                "speedup_gap_percent_of_ideal": 100.0
                * (ideal_memory_speedup - measured_speedup)
                / ideal_memory_speedup,
                # Fraction of improvement beyond 1x that was captured.
                "incremental_fusion_benefit_fraction": (
                    (measured_speedup - 1) / (ideal_memory_speedup - 1)
                    if ideal_memory_speedup > 1
                    else float("nan")
                ),
            }
        )

    # Roofline analysis
    roofline_results: dict[str, Any] = {}

    if peak_compute_tflops is not None and peak_bandwidth_gbps is not None:
        peak_flops_per_second = peak_compute_tflops * 1e12
        peak_bytes_per_second = peak_bandwidth_gbps * 1e9

        ridge_point_flops_per_byte = peak_flops_per_second / peak_bytes_per_second

        # Reference ideal lower bound
        reference_compute_bound_seconds = useful_flops / peak_flops_per_second

        reference_memory_bound_seconds = reference_min_bytes / peak_bytes_per_second

        reference_roofline_seconds = max(
            reference_compute_bound_seconds,
            reference_memory_bound_seconds,
        )

        reference_limiting_resource = (
            "compute"
            if reference_compute_bound_seconds >= reference_memory_bound_seconds
            else "memory"
        )

        # Fused optimistic lower bound:
        # actual padded FLOPs plus ideal memory traffic
        fused_compute_bound_seconds = actual_fused_flops / peak_flops_per_second

        fused_ideal_memory_bound_seconds = fused_min_bytes / peak_bytes_per_second

        fused_ideal_roofline_seconds = max(
            fused_compute_bound_seconds,
            fused_ideal_memory_bound_seconds,
        )

        fused_ideal_limiting_resource = (
            "compute"
            if fused_compute_bound_seconds >= fused_ideal_memory_bound_seconds
            else "memory"
        )

        # Pessimistic no-cache fused estimate
        fused_no_cache_memory_bound_seconds = (
            fused_no_cache_bytes / peak_bytes_per_second
        )

        fused_no_cache_roofline_seconds = max(
            fused_compute_bound_seconds,
            fused_no_cache_memory_bound_seconds,
        )

        fused_no_cache_limiting_resource = (
            "compute"
            if fused_compute_bound_seconds >= fused_no_cache_memory_bound_seconds
            else "memory"
        )

        roofline_results.update(
            {
                "peak_compute_tflops": peak_compute_tflops,
                "peak_bandwidth_gbps": peak_bandwidth_gbps,
                "ridge_point_flops_per_byte": ridge_point_flops_per_byte,
                "reference_compute_bound_ms": reference_compute_bound_seconds * 1000,
                "reference_memory_bound_ms": reference_memory_bound_seconds * 1000,
                "reference_roofline_ms": reference_roofline_seconds * 1000,
                "reference_roofline_limiter": reference_limiting_resource,
                "fused_compute_bound_ms": fused_compute_bound_seconds * 1000,
                "fused_ideal_memory_bound_ms": fused_ideal_memory_bound_seconds * 1000,
                "fused_ideal_roofline_ms": fused_ideal_roofline_seconds * 1000,
                "fused_ideal_roofline_limiter": fused_ideal_limiting_resource,
                "fused_no_cache_memory_bound_ms": fused_no_cache_memory_bound_seconds
                * 1000,
                "fused_no_cache_roofline_ms": fused_no_cache_roofline_seconds * 1000,
                "fused_no_cache_roofline_limiter": fused_no_cache_limiting_resource,
                "ideal_roofline_speedup": reference_roofline_seconds
                / fused_ideal_roofline_seconds,
                "no_cache_roofline_speedup": reference_roofline_seconds
                / fused_no_cache_roofline_seconds,
            }
        )

        if ref_ms is not None and fused_ms is not None:
            ref_seconds = ref_ms / 1000.0
            fused_seconds = fused_ms / 1000.0

            roofline_results.update(
                {
                    # A value of 1.0 would mean the measured latency equals
                    # the ideal lower bound.
                    "reference_roofline_efficiency": reference_roofline_seconds
                    / ref_seconds,
                    "fused_ideal_roofline_efficiency": fused_ideal_roofline_seconds
                    / fused_seconds,
                    "fused_no_cache_roofline_efficiency": fused_no_cache_roofline_seconds
                    / fused_seconds,
                    "reference_compute_utilization": (useful_flops / ref_seconds)
                    / peak_flops_per_second,
                    "fused_effective_compute_utilization": (
                        useful_flops / fused_seconds
                    )
                    / peak_flops_per_second,
                    "fused_actual_compute_utilization": (
                        actual_fused_flops / fused_seconds
                    )
                    / peak_flops_per_second,
                }
            )

    # Shared-memory comparison
    shared_results: dict[str, Any] = {
        "logical_tile_shared_bytes": logical_tile_shared_bytes,
        "inferred_one_extra_stage_shared_bytes": inferred_one_extra_stage_shared_bytes,
    }

    if compiled_shared_bytes is not None:
        shared_results["compiled_shared_bytes"] = compiled_shared_bytes

    if shared_memory_limit_bytes is not None:
        shared_results["shared_memory_limit_bytes"] = shared_memory_limit_bytes

        shared_results["logical_shared_fraction_of_limit"] = (
            logical_tile_shared_bytes / shared_memory_limit_bytes
        )

        shared_results["inferred_shared_fraction_of_limit"] = (
            inferred_one_extra_stage_shared_bytes / shared_memory_limit_bytes
        )

        if compiled_shared_bytes is not None:
            shared_results["compiled_shared_fraction_of_limit"] = (
                compiled_shared_bytes / shared_memory_limit_bytes
            )

            shared_results["compiled_shared_fits"] = (
                compiled_shared_bytes <= shared_memory_limit_bytes
            )

    # Per-expert records
    per_expert = []

    offset = 0

    for expert_id, (
        group_size,
        program_count,
        expert_padding,
    ) in enumerate(
        zip(
            groups,
            programs_per_expert,
            padded_rows_per_expert,
        )
    ):
        per_expert.append(
            {
                "expert_id": expert_id,
                "start_offset": offset,
                "end_offset": offset + group_size,
                "group_size": group_size,
                "active_programs": program_count,
                "processed_row_slots": program_count * BM,
                "padded_rows": expert_padding,
                "row_utilization": (
                    group_size / (program_count * BM) if program_count > 0 else 1.0
                ),
            }
        )

        offset += group_size

    # Flat result dictionary
    results: dict[str, Any] = {
        "M": M,
        "E": E,
        "D": D,
        "F": F,
        "BM": BM,
        "BF": BF,
        "dtype_bytes": dtype_bytes,
        "group_sizes": groups,
        "programs_per_expert": programs_per_expert,
        "padded_rows_per_expert": padded_rows_per_expert,
        "max_group_size": max_group_size,
        "max_programs_per_expert": max_programs_per_expert,
        "active_programs_Q": active_programs,
        "launched_programs": launched_programs,
        "inactive_programs": inactive_programs,
        "grid_utilization": grid_utilization,
        "processed_token_slots": processed_token_slots,
        "padded_rows": padded_rows,
        "row_utilization": row_utilization,
        "padding_overcompute_factor": padding_overcompute_factor,
        "padding_overhead_fraction": padding_overhead_fraction,
        "padding_overhead_percent": 100.0 * padding_overhead_fraction,
        "hidden_chunks_F_over_BF": hidden_chunks,
        "useful_flops": useful_flops,
        "actual_fused_flops": actual_fused_flops,
        "padding_extra_flops": padding_extra_flops,
        "reference_min_bytes": reference_min_bytes,
        "fused_min_bytes": fused_min_bytes,
        "hidden_activation_bytes": hidden_activation_bytes,
        "hidden_write_read_saved_bytes": hidden_write_read_saved_bytes,
        "memory_saved_fraction": memory_saved_fraction,
        "memory_saved_percent": 100.0 * memory_saved_fraction,
        "fused_no_cache_bytes": fused_no_cache_bytes,
        "fused_tile_upper_bytes": fused_tile_upper_bytes,
        "minimum_weight_bytes": minimum_weight_bytes,
        "no_cache_weight_bytes": no_cache_weight_bytes,
        "weight_reread_factor_no_cache": weight_reread_factor_no_cache,
        "reference_ideal_ai_flops_per_byte": reference_ideal_ai,
        "fused_ideal_effective_ai_flops_per_byte": fused_ideal_effective_ai,
        "fused_ideal_actual_ai_flops_per_byte": fused_ideal_actual_ai,
        "fused_no_cache_actual_ai_flops_per_byte": fused_no_cache_actual_ai,
        "fused_per_program_ai_flops_per_byte": fused_per_program_ai,
        "fused_large_F_ai_approximation": fused_large_F_ai_approximation,
        "expansion_ratio_F_over_D": expansion_ratio,
        "token_to_expert_hidden_ratio_M_over_EF": token_to_expert_hidden_ratio,
        "fraction_of_asymptotic_benefit": fraction_of_asymptotic_benefit,
        "fraction_of_asymptotic_benefit_percent": 100.0
        * fraction_of_asymptotic_benefit,
        "asymptotic_max_memory_speedup": asymptotic_max_memory_speedup,
        "ideal_memory_speedup": ideal_memory_speedup,
        "ideal_memory_speedup_alternate": ideal_memory_speedup_alternate,
        **timing_results,
        **roofline_results,
        **shared_results,
        "per_expert": per_expert,
    }

    # Print report
    if print_results:
        print("=" * 78)
        print("RAGGED MoE FUSED-KERNEL ANALYSIS")
        print("=" * 78)

        print("\n1. Problem configuration")
        print("-" * 78)
        print(f"M total tokens:                 {M:,}")
        print(f"E experts:                      {E:,}")
        print(f"D model dimension:              {D:,}")
        print(f"F hidden dimension:             {F:,}")
        print(f"Expansion ratio F / D:          {expansion_ratio:.6f}")
        print(f"BM token tile:                  {BM:,}")
        print(f"BF hidden tile:                 {BF:,}")
        print(f"Hidden chunks F / BF:           {hidden_chunks:,}")
        print(f"Bytes per element:              {dtype_bytes}")
        print(f"Group sizes:                    {groups}")

        print("\n2. Grid and routing")
        print("-" * 78)
        print(f"Programs per expert:            {programs_per_expert}")
        print(f"Active programs Q:              {active_programs:,}")
        print(f"Rectangular programs launched:  {launched_programs:,}")
        print(f"Inactive grid programs:         {inactive_programs:,}")
        print(f"Grid utilization:               {100 * grid_utilization:.4f}%")
        print(f"Processed token slots:          {processed_token_slots:,}")
        print(f"Useful token rows:              {M:,}")
        print(f"Padded rows:                    {padded_rows:,}")
        print(f"Padding per expert:             {padded_rows_per_expert}")
        print(f"Row utilization:                {100 * row_utilization:.6f}%")
        print(f"Padding FLOP overhead:          {100 * padding_overhead_fraction:.6f}%")

        print("\n3. FLOP analysis")
        print("-" * 78)
        print(f"Useful FLOPs 4*M*D*F:           {_format_flops(useful_flops)}")
        print(f"Actual fused FLOPs:             {_format_flops(actual_fused_flops)}")
        print(f"Padding-only extra FLOPs:       {_format_flops(padding_extra_flops)}")
        print(f"Overcompute factor:             {padding_overcompute_factor:.9f}x")

        print("\n4. Ideal global-memory traffic")
        print("-" * 78)
        print(f"Reference minimum traffic:      {_format_bytes(reference_min_bytes)}")
        print(f"Fused minimum traffic:          {_format_bytes(fused_min_bytes)}")
        print(
            f"One hidden H allocation:        {_format_bytes(hidden_activation_bytes)}"
        )
        print(
            f"Hidden H write + read removed:  "
            f"{_format_bytes(hidden_write_read_saved_bytes)}"
        )
        print(f"Reference traffic eliminated:   {100 * memory_saved_fraction:.4f}%")
        print(f"Fused no-cache estimate:        {_format_bytes(fused_no_cache_bytes)}")
        print(
            f"Fused tile-level upper model:   {_format_bytes(fused_tile_upper_bytes)}"
        )
        print(f"No-cache weight reread factor:  {weight_reread_factor_no_cache:.4f}x")

        print("\n5. Arithmetic intensity")
        print("-" * 78)
        print(f"Reference ideal AI:             {reference_ideal_ai:.6f} FLOP/byte")
        print(
            f"Fused ideal effective AI:       {fused_ideal_effective_ai:.6f} FLOP/byte"
        )
        print(f"Fused ideal actual AI:          {fused_ideal_actual_ai:.6f} FLOP/byte")
        print(
            f"Fused no-cache actual AI:       {fused_no_cache_actual_ai:.6f} FLOP/byte"
        )
        print(f"Fused per-program AI:           {fused_per_program_ai:.6f} FLOP/byte")
        print(
            f"Large-F AI approximation:       "
            f"{fused_large_F_ai_approximation:.6f} FLOP/byte"
        )

        print("\n6. Ideal fusion-benefit model")
        print("-" * 78)
        print(f"M / (E*F):                      {token_to_expert_hidden_ratio:.6f}")
        print(
            f"Fraction of max benefit:        "
            f"{100 * fraction_of_asymptotic_benefit:.4f}%"
        )
        print(f"Asymptotic max speedup 1+F/D:   {asymptotic_max_memory_speedup:.6f}x")
        print(f"Ideal bandwidth speedup:        {ideal_memory_speedup:.9f}x")

        if measured_speedup is not None:
            print("\n7. Measured comparison")
            print("-" * 78)

            if ref_ms is not None and fused_ms is not None:
                print(f"Reference latency:              {ref_ms:.6f} ms")
                print(f"Fused latency:                  {fused_ms:.6f} ms")

            print(f"Measured speedup:               {measured_speedup:.6f}x")
            print(
                f"Measured / ideal:               "
                f"{timing_results['speedup_percent_of_ideal']:.4f}%"
            )
            print(
                f"Gap below ideal:                "
                f"{timing_results['speedup_gap_percent_of_ideal']:.4f}%"
            )
            print(
                f"Incremental benefit captured:   "
                f"{100 * timing_results['incremental_fusion_benefit_fraction']:.4f}%"
            )

            if ref_ms is not None and fused_ms is not None:
                print(
                    f"Reference effective TFLOP/s:    "
                    f"{timing_results['ref_effective_tflops']:.6f}"
                )
                print(
                    f"Fused effective TFLOP/s:        "
                    f"{timing_results['fused_effective_tflops']:.6f}"
                )
                print(
                    f"Fused actual TFLOP/s:           "
                    f"{timing_results['fused_actual_tflops']:.6f}"
                )
                print(
                    f"Reference tokens/s:             "
                    f"{timing_results['ref_tokens_per_second']:,.3f}"
                )
                print(
                    f"Fused tokens/s:                 "
                    f"{timing_results['fused_tokens_per_second']:,.3f}"
                )

        if roofline_results:
            print("\n8. Roofline analysis")
            print("-" * 78)
            print(f"Peak compute:                   {peak_compute_tflops:.3f} TFLOP/s")
            print(f"Peak bandwidth:                 {peak_bandwidth_gbps:.3f} GB/s")
            print(
                f"Ridge point:                    "
                f"{roofline_results['ridge_point_flops_per_byte']:.6f} "
                f"FLOP/byte"
            )
            print(
                f"Reference roofline latency:     "
                f"{roofline_results['reference_roofline_ms']:.6f} ms "
                f"({roofline_results['reference_roofline_limiter']}-limited)"
            )
            print(
                f"Fused ideal roofline latency:   "
                f"{roofline_results['fused_ideal_roofline_ms']:.6f} ms "
                f"({roofline_results['fused_ideal_roofline_limiter']}-limited)"
            )
            print(
                f"Fused no-cache roofline:        "
                f"{roofline_results['fused_no_cache_roofline_ms']:.6f} ms "
                f"({roofline_results['fused_no_cache_roofline_limiter']}-limited)"
            )
            print(
                f"Ideal roofline speedup:         "
                f"{roofline_results['ideal_roofline_speedup']:.6f}x"
            )
            print(
                f"No-cache roofline speedup:      "
                f"{roofline_results['no_cache_roofline_speedup']:.6f}x"
            )

            if ref_ms is not None and fused_ms is not None:
                print(
                    f"Reference roofline efficiency:  "
                    f"{100 * roofline_results['reference_roofline_efficiency']:.4f}%"
                )
                print(
                    f"Fused ideal roof efficiency:    "
                    f"{100 * roofline_results['fused_ideal_roofline_efficiency']:.4f}%"
                )
                print(
                    f"Fused actual compute use:       "
                    f"{100 * roofline_results['fused_actual_compute_utilization']:.4f}%"
                )

        print("\n9. Per-program storage estimates")
        print("-" * 78)
        print(
            f"Logical tile estimate:          "
            f"{_format_bytes(logical_tile_shared_bytes)}"
        )
        print(
            f"One-extra-stage estimate:       "
            f"{_format_bytes(inferred_one_extra_stage_shared_bytes)}"
        )

        if compiled_shared_bytes is not None:
            print(
                f"Compiler-requested shared mem:  "
                f"{_format_bytes(compiled_shared_bytes)}"
            )

        if shared_memory_limit_bytes is not None:
            print(
                f"Per-block shared-memory limit:  "
                f"{_format_bytes(shared_memory_limit_bytes)}"
            )

            if compiled_shared_bytes is not None:
                print(
                    f"Compiler request / limit:       "
                    f"{100 * shared_results['compiled_shared_fraction_of_limit']:.4f}%"
                )
                print(
                    f"Kernel fits shared limit:       "
                    f"{shared_results['compiled_shared_fits']}"
                )

        print("\n10. Per-expert breakdown")
        print("-" * 78)
        print(
            f"{'Expert':>7} "
            f"{'Start':>10} "
            f"{'End':>10} "
            f"{'Tokens':>10} "
            f"{'Programs':>10} "
            f"{'Padded':>10} "
            f"{'Utilization':>14}"
        )

        for row in per_expert:
            print(
                f"{row['expert_id']:>7} "
                f"{row['start_offset']:>10,} "
                f"{row['end_offset']:>10,} "
                f"{row['group_size']:>10,} "
                f"{row['active_programs']:>10,} "
                f"{row['padded_rows']:>10,} "
                f"{100 * row['row_utilization']:>13.4f}%"
            )

        print("=" * 78)

    return results


def save_ragged_analysis(
    results: dict[str, Any],
    *,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> None:
    """
    Save the analysis to JSON and/or a one-row CSV.

    The per-expert list remains structured in JSON and is converted into
    a string representation in CSV.
    """

    if json_path is None and csv_path is None:
        raise ValueError("Provide json_path, csv_path, or both.")

    if json_path is not None:
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        with json_file.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

        print(f"Saved JSON analysis to: {json_file}")

    if csv_path is not None:
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert nested values to JSON strings for a clean one-row CSV.
        csv_ready = {}

        for key, value in results.items():
            if isinstance(value, (list, dict, tuple)):
                csv_ready[key] = json.dumps(value)
            else:
                csv_ready[key] = value

        pd.DataFrame([csv_ready]).to_csv(
            csv_file,
            index=False,
        )

        print(f"Saved CSV analysis to: {csv_file}")


def memory_request_calculator(dtype_bytes, BM, D, BF):
    res = dtype_bytes * (2 * BM * D + 3 * D * BF + BM * BF)
    print(res)
