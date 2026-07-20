"""Post-hoc analysis of a completed sweep.

Reads results.csv/failures.csv from an experiment output directory and derives
the summaries quoted in the README:

  1. Speedup vs. token count M (where fusion starts paying off).
  2. Cost vs. expert count E for both implementations -- the finding that
     XLA's ragged_dot scales with E while the fused kernel does not.
  3. Median speedup by tile size at a matched D, which the sweep's own
     heatmap cannot show without a selection effect.
  4. Roofline efficiency, computed from a supplied peak compute/bandwidth pair.
     A sweep run without peaks configured stores no roofline columns at all, so
     they are derived here from the measured latencies and byte/FLOP counts.
  5. The source-level shared-memory model against the sizes the compiler
     actually requested in RESOURCE_EXHAUSTED failures -- which it predicts
     only to within roughly 0.7-1.7x, not exactly.

Usage
-----
    python src/analyze_results.py results/structured_float32 \
        --peak-compute-tflops 30.3 --peak-bandwidth-gbps 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# NVIDIA L4, float32 (non-tensor-core) and HBM bandwidth.
L4_PEAK_TFLOPS = 30.3
L4_PEAK_GBPS = 300.0

DTYPE_BYTES = 4


def speedup_vs_tokens(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("M").agg(
        cases=("M", "size"),
        median_speedup=("speedup_ref_over_fused", "median"),
        max_speedup=("speedup_ref_over_fused", "max"),
        median_ideal=("ideal_memory_speedup", "median"),
        percent_of_ideal=("speedup_percent_of_ideal", "median"),
    )


def cost_vs_experts(results: pd.DataFrame, M: int, D: int, F: int) -> pd.DataFrame:
    """Latency vs. E at fixed problem size: useful FLOPs are constant, so any
    trend is implementation cost, not work."""
    subset = results[(results.M == M) & (results.D == D) & (results.F == F)]
    table = subset.groupby("E").agg(
        reference_ms=("ref_median_ms", "median"),
        fused_ms=("fused_median_ms", "median"),
        reference_bytes=("reference_min_bytes", "median"),
        useful_gflops=("useful_flops", "median"),
    )
    table["useful_gflops"] /= 1e9
    table["reference_mib"] = table.pop("reference_bytes") / 1024**2
    table["speedup"] = table.reference_ms / table.fused_ms
    return table


def tile_comparison(results: pd.DataFrame, D: int = 64) -> pd.DataFrame:
    """Median speedup by (BM, BF) at a single D.

    Aggregating tiles across all problem sizes is misleading: large BF only
    compiles at small D, so its median is drawn from an easier subset. Fixing D
    puts every tile on the same cases.
    """
    subset = results[results.D == D]
    return subset.pivot_table(
        index="BM", columns="BF", values="speedup_ref_over_fused", aggfunc="median"
    )


def save_tile_heatmap(results: pd.DataFrame, path: Path, D: int = 64) -> None:
    """Write the matched-D tile heatmap used in the README.

    The sweep's own heatmap pools every problem size, which favours tiles that
    only compile on easy (small-D) cases. This one fixes D so each cell is
    drawn from the same set of problems.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = tile_comparison(results, D=D)
    figure, axes = plt.subplots(figsize=(7, 5))
    image = axes.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    axes.set_xticks(range(len(pivot.columns)), labels=[str(v) for v in pivot.columns])
    axes.set_yticks(range(len(pivot.index)), labels=[str(v) for v in pivot.index])
    axes.set_xlabel("BF hidden tile")
    axes.set_ylabel("BM token tile")
    axes.set_title(f"Median speedup by tile (D={D}, matched cases)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            if np.isfinite(value):
                axes.text(column, row, f"{value:.2f}x", ha="center", va="center")
    figure.colorbar(image, ax=axes, label="Median speedup (x)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def padding_effect(results: pd.DataFrame) -> pd.DataFrame:
    """Speedup grouped by ragged-tile padding overhead.

    Padding is the mechanism behind the small-M losses: each expert's token
    count is rounded up to a multiple of BM, so when M/E is small relative to
    BM most of the tile is wasted compute.
    """
    bucketed = results.copy()
    bucketed["padding_bucket"] = pd.cut(
        bucketed.padding_overhead_percent,
        [-1, 1, 10, 50, 10_000],
        labels=["<1%", "1-10%", "10-50%", ">50%"],
    )
    return bucketed.groupby("padding_bucket", observed=True).agg(
        cases=("M", "size"),
        median_speedup=("speedup_ref_over_fused", "median"),
        median_M=("M", "median"),
    )


def roofline(
    results: pd.DataFrame,
    peak_compute_tflops: float,
    peak_bandwidth_gbps: float,
) -> pd.DataFrame:
    peak_flops = peak_compute_tflops * 1e12
    peak_bytes = peak_bandwidth_gbps * 1e9

    frame = results.copy()
    ref_compute_s = frame.useful_flops / peak_flops
    ref_memory_s = frame.reference_min_bytes / peak_bytes
    fused_compute_s = frame.actual_fused_flops / peak_flops
    fused_memory_s = frame.fused_min_bytes / peak_bytes

    frame["reference_roofline_ms"] = 1e3 * np.maximum(ref_compute_s, ref_memory_s)
    frame["fused_roofline_ms"] = 1e3 * np.maximum(fused_compute_s, fused_memory_s)
    frame["reference_efficiency"] = frame.reference_roofline_ms / frame.ref_median_ms
    frame["fused_efficiency"] = frame.fused_roofline_ms / frame.fused_median_ms
    frame["reference_limiter"] = np.where(
        ref_compute_s >= ref_memory_s, "compute", "memory"
    )
    frame["fused_limiter"] = np.where(
        fused_compute_s >= fused_memory_s, "compute", "memory"
    )
    return frame


def shared_memory_model(failures: pd.DataFrame) -> pd.DataFrame:
    """Compare the source-level tile estimate against the compiler's request.

    The kernel holds x (BM, D), both weight tiles (D, BF) and (BF, D), and the
    hidden tile (BM, BF); the accumulator is (BM, D). That is

        bytes = dtype * (2*BM*D + 2*D*BF + BM*BF)
    """
    frame = failures.copy()
    frame["requested"] = frame.error_message.str.extract(r"requested (\d+)").astype(
        float
    )
    frame["available"] = frame.error_message.str.extract(r"available: (\d+)").astype(
        float
    )
    frame = frame.dropna(subset=["requested"])
    frame["predicted"] = DTYPE_BYTES * (
        2 * frame.BM * frame.D + 2 * frame.D * frame.BF + frame.BM * frame.BF
    )
    frame["requested_over_predicted"] = frame.requested / frame.predicted
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--peak-compute-tflops", type=float, default=L4_PEAK_TFLOPS)
    parser.add_argument("--peak-bandwidth-gbps", type=float, default=L4_PEAK_GBPS)
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    results = pd.read_csv(args.output_dir / "results.csv")
    print(f"Cases: {len(results):,} | numerically valid: {int(results.numerically_acceptable.sum()):,}")
    print(f"Max L2 relative error: {results.l2_relative_error.max():.2e}\n")

    print("Speedup vs. routed tokens M")
    print(speedup_vs_tokens(results).to_string(), "\n")

    print("Cost vs. expert count E (M=16384, D=128, F=512; useful FLOPs constant)")
    print(cost_vs_experts(results, M=16384, D=128, F=512).to_string(), "\n")

    print("Median speedup by tile, at matched D=64")
    print(tile_comparison(results, D=64).to_string(), "\n")

    print("Speedup by ragged-tile padding overhead")
    print(padding_effect(results).to_string(), "\n")

    save_tile_heatmap(results, args.output_dir / "plots" / "tile_heatmap_matched_D64.png")
    print()

    ridge = args.peak_compute_tflops * 1e12 / (args.peak_bandwidth_gbps * 1e9)
    print(
        f"Roofline @ {args.peak_compute_tflops} TFLOP/s, "
        f"{args.peak_bandwidth_gbps} GB/s (ridge {ridge:.0f} FLOP/byte)"
    )
    roofed = roofline(results, args.peak_compute_tflops, args.peak_bandwidth_gbps)
    large = roofed[roofed.M >= 16384]
    print(
        large.groupby("E")
        .agg(
            cases=("M", "size"),
            reference_efficiency=("reference_efficiency", "median"),
            fused_efficiency=("fused_efficiency", "median"),
        )
        .to_string()
    )
    print(
        "Limiter -- reference:",
        roofed.reference_limiter.value_counts().to_dict(),
        "| fused:",
        roofed.fused_limiter.value_counts().to_dict(),
    )
    print(f"Best fused throughput: {results.fused_effective_tflops.max():.1f} TFLOP/s\n")

    failures_path = args.output_dir / "failures.csv"
    if failures_path.exists():
        failures = pd.read_csv(failures_path)
        print(f"Failures: {len(failures):,}")
        print(failures.error_type.value_counts().to_string())
        smem = shared_memory_model(failures)
        if not smem.empty:
            print(
                f"\nShared-memory limit reported by compiler: "
                f"{int(smem.available.iloc[0]):,} bytes"
            )
            # Broken out by D as well as BF: the ratio depends on both, so a
            # BF-only median hides the spread and overstates the model.
            print("Compiler request / source-level model (median):")
            print(
                smem.pivot_table(
                    index="D",
                    columns="BF",
                    values="requested_over_predicted",
                    aggfunc="median",
                ).to_string()
            )
            exact = (smem.requested - smem.predicted).abs() < 1
            print(f"Exact matches: {exact.mean():.1%} of {len(smem):,} failures")


if __name__ == "__main__":
    main()
