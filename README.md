# Fused MoE MLP Kernel in Pallas (JAX)

A custom GPU kernel that fuses the expert MLP of a Mixture-of-Experts
transformer, up-projection → GELU → down-projection, into a single Pallas
kernel, replacing the two `jax.lax.ragged_dot` calls that would otherwise
materialize the `(M, F)` hidden activation in HBM.

Grew out of my [Chinchilla scaling-laws project](https://github.com/KrushnaBhanushali/chinchilla-scaling-jax),
and idea from Vlad Feinberg post on learning pre-training and scaling.(https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html) where the MoE model's expert MLP is exactly this pair of ragged matmuls.

## The kernel

Tokens are pre-sorted by expert. The grid is `(E, max_group_size / BM)`:
each program owns one `BM`-token block of one expert's token range, streams
the expert's `W_up`/`W_down` in `BF`-sized chunks of the hidden dimension,
and accumulates the `(BM, D)` output in registers the `(BM, BF)` hidden
activation never leaves on-chip memory

```
X: (M, D)  sorted by expert, ragged group sizes
W_up:   (E, D, F)      H = GELU(X @ W_up[e])      (never written to HBM)
W_down: (E, F, D)      Y = H @ W_down[e]          accumulated over BF chunks
```

## What's here

| File | Contents |
|---|---|
| `src/kernel.py` | The fused kernel, its `pallas_call` wrapper, and the `ragged_dot` reference implementations |
| `src/practice.py` | Warm-up Pallas kernels (iota, copy, masked block ops) the learning progression from Pallas Docs|
| `src/benchmark.py` | Latency/throughput sweep vs. the XLA reference: median-of-N timing, TFLOP/s, tokens/s, numerical-acceptance gate |
| `src/analysis.py` | Roofline & memory-traffic model: arithmetic intensity, best/worst-case HBM traffic, shared-memory estimates per block config |
| `tests/check_correctness.py` | Element-wise comparison against `ragged_dot` reference (abs/rel/L2 error, worst-entry report) |
| `profiling/profile_kernels.py` | Traces fused vs. reference with `jax.profiler` (perfetto/xprof) |
| `src/ragged_moe_speedup_experiment.py` | Self-contained sweep harness: structured/full-grid case generation, 4 routing patterns, memory-safe preflight, resume, correctness gates, ideal-bandwidth-model predictions, summary tables + 9 plot families. Version-portable (compat layer for the post-0.8 Pallas memory-op API) |
| `notebooks/ragged_moe_colab_runner.ipynb` | Runs the sweep on a Colab GPU|

## Run

```bash
pip install -r requirements.txt

# Correctness (CPU-friendly: interpret mode)
PYTHONPATH=. python tests/check_correctness.py

# Benchmark sweep (needs a GPU; edit M/D/F/BM/BF at the bottom)
PYTHONPATH=. python -c "from src.benchmark import run_sweep; \
    print(run_sweep(M_values=[2**14, 2**16, 2**18], D=128, F=1024, BM=128, BF=16))"

# Profile traces (GPU), then: xprof --logdir jax_profiles
PYTHONPATH=. python profiling/profile_kernels.py
```

**JAX is pinned to 0.8.0** — the kernel uses the `pl.load`/`pl.store` masked
memory ops, which were removed from Pallas in later releases.

## Results

*(benchmark figures and profiler traces landing here)*
