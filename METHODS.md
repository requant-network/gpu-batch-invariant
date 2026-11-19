# METHODS — mathematical contract, kernel designs, timing boundaries

Study: *Batch-invariant deterministic inference kernels on SM120: measuring batch-induced nondeterminism and its cost.* Bounded spike, campaign `requant-new-studies-20261119`.

## 1. Question and acceptance criteria

**Q1 (problem).** When the same input row is computed alone (batch 1) and inside larger batches {2, 7, 32, 128}, do standard GPU inference kernels produce bitwise-identical outputs? Which mechanisms cause differences, and how large are they (fraction of differing elements, max ULP)?

**Q2 (fix).** Do fixed-reduction-order ("batch-invariant") variants produce a probe row that is **bitwise identical across ALL batch sizes** (acceptance test), while staying within a documented tolerance of the cuBLAS/standard reference (same math, different order)?

**Q3 (cost).** What is the resident median-latency overhead of the invariant variants at the same shapes, and does an end-to-end transformer-like block replay bitwise-equal under different surrounding batch padding?

Acceptance is **bitwise** (`memcmp` of storage bits), not tolerance-based. Tolerances are reported separately against an exact oracle. Zero difference in the standard path is a valid finding and is reported as such.

## 2. Input contract (exactness-friendly synthetic data)

- RNG: splitmix64, seeds **20261119** and **777** (two independent cohorts for every main claim). The identical RNG is re-implemented in pure Python in `tests/validate_results.py`.
- Every input element is `k/256` with integer `k ∈ [−512, 512]` — i.e. an exact multiple of 2^-8 in [−2, 2]. These values are **exactly representable in fp16, bf16 and fp32**, so storage introduces no rounding and all measured differences come from compute order, not input conversion.
- GEMM: `C[M,N] = A[M,K] · B[K,N]`, row-major, probe = row 0 of C. Shapes inspired by LLM layers (hidden 4096, FFN 11008):
  - `hidden_square`: K=N=4096 (attention/o projection class)
  - `ffn_up`: K=4096, N=11008
  - `ffn_down`: K=11008, N=4096
- Row kernels: widths N ∈ {4096, 11008}, fp32. RMSNorm weights drawn from the same grid; eps=1e-5.
- Attention: H=32 heads, D=128, S ∈ {128, 512}, fp16 storage / fp32 compute.
- Batch sizes {1, 2, 7, 32, 128}; MAXB=128 rows allocated once. Peak VRAM < 4 GB (largest buffers: fp32 128×11008 ≈ 5.6 MB; attention S=512 B=128 Q/K/V/O ≈ 4×128·32·512·128·2 B = 512 MB; cuBLASLt workspace 32 MiB).

## 3. Exact oracle and proven error bound

For the fp32 GEMM probe row (hidden_square, K=4096, seed 20261119) an **exact** reference is computed on CPU in fp64:

- products `a_k·b_k` are multiples of 2^-16 with |p| ≤ 4;
- any partial sum of ≤ 4096 such products is a multiple of 2^-16 bounded by 16384 < 2^14, needing ≤ 30 significand bits < 53;
- therefore **sequential fp64 accumulation is exact at every step** (no rounding), and the fp64 result equals the real-number dot product exactly.

Both GPU rows (invariant and cuBLAS batch-1) are checked against the standard forward-error bound for fp32 summation of K terms:

```
|fl(Σ a_k b_k) − Σ a_k b_k| ≤ γ_K · Σ |a_k b_k|,   γ_K = K·u / (1 − K·u),  u = 2^-24
```

with `Σ|a_k b_k|` computed exactly in fp64. The bound is proven, not fitted; measured errors must lie inside it (they do, with ≥3 orders of magnitude margin).

**Independence:** the C++ host oracle (`invariant` mode) and a pure-Python fp64 re-derivation from the seed (`tests/validate_results.py`, different code path, different language) must agree **bitwise** on all 4096 elements, including the exact `Σ|a_k b_k|` row. A deliberately perturbed value beyond the bound must be rejected by the bound checker (negative control).

ULP distance uses the monotone ordered-integer map of IEEE bit patterns (`ordered32/ordered16`: negatives mapped by bitwise complement, positives by sign-bit set), so adjacent representable values differ by exactly 1 ULP.

## 4. Kernel designs

### 4.1 Standard (batch-dependent) paths

- **GEMM:** cuBLASLt, `CUBLAS_COMPUTE_32F`, scale type fp32, storage fp16/bf16/fp32. On-device heuristic (`cublasLtMatmulAlgoGetHeuristic`, top-1, 32 MiB workspace cap with fatal guard) selects algo/tile/split-K **per M**. FP16/BF16 storage with fp32 compute executes SM120 tensor-core kernels; fp32 storage executes FFMA CUDA-core kernels (TF32 not requested). Algo ID, tile ID, split-K factor and reduction scheme are recorded for every launch. Column-major views: A_lt=B (N×K), B_lt=A (K×M), C_lt=C (N×M).
- **Row softmax / RMSNorm:** one block per row; thread count from a batch-dependent heuristic `B≤2→1024, B≤7→512, B≤32→256, else 128` (models real stacks that pick block sizes from shape). Two-pass softmax (block max, then Σexp, then scale); strided per-thread partials + warp-shuffle tree + shared-memory tree. Different thread counts ⇒ different reduction trees ⇒ different rounding.
- **Attention:** naive tiled path, one block per (query, head, batch); thread heuristic `B≤2→256, B≤32→128, else 64`. Pass 1: scores[s]=dot(q,k_s) via strided-d partials + block tree; pass 2: two-pass softmax over S scores; pass 3: out[d]=Σ_s p[s]·V[s,d] ascending s, fp32 accumulator, fp16 output.

### 4.2 Batch-invariant (fixed-order) variants

- **GEMM (`gemm_inv_kernel`):** fixed 32×32 output tile, 1024 threads (32,32), shared-memory A/B tiles, **single-pass ascending-K accumulation in fp32 via `fmaf`**, inner 32-step unroll in fixed order, **no split-K**. Grid = (N/32, ceil(M/32)): the batch dimension is **padded to a multiple of the tile**, and each output tile is computed independently of every other tile, so the probe row's bits cannot depend on M. N is a multiple of 32 for all shapes.
- **Softmax / RMSNorm:** identical kernel code, **frozen 256-thread config** (`ROW_BLOCK_INV`) for every batch ⇒ identical per-thread striding and identical reduction tree ⇒ identical rounding. Two-pass order fixed.
- **Attention:** identical kernel code, **frozen 128 threads** (`ATTN_THREADS_INV`) for every batch; ascending-d dot partials, fixed block tree, two-pass softmax, ascending-s output accumulation — all independent of batch.
- **Block path:** RMSNorm → QKV GEMM (invariant) → QKV repack (elementwise) → attention (frozen) → o merge (elementwise) → o GEMM (invariant) → RMSNorm → FFN up GEMM → SiLU (elementwise, fp32 compute) → FFN down GEMM. Every reduction is fixed-order; elementwise ops are trivially batch-invariant.

## 5. Comparison protocol (experiment 1)

For each kernel/shape/dtype/seed: allocate MAXB rows, fill from the seeded grid, run batch 1 first and keep its probe row as reference; then run batches {2,7,32,128} on the **same device buffers** and `memcmp` the probe row against the reference. Report `diff_count`, `diff_fraction`, `max_ulp`, `max_abs`. Additionally every cuBLASLt GEMM is launched twice at identical shape (run-to-run control). The acceptance harness (experiment 2) is the same comparator applied to the invariant variants across all five batches; standard kernels are also pushed through it as negative controls (targeted cases where experiment 1 observed differences must fail; zero-difference cases must pass).

## 6. Timing boundaries (experiment 3a)

- **Resident kernel boundary:** CUDA events recorded immediately around the kernel launch on the same stream; inputs/outputs already resident on device; **no H2D/D2H transfers, no allocations, no heuristic queries inside the timed region**. cuBLASLt descriptors/algos are created once outside timing.
- Warmup + repeated launches: GEMM 10 warmup + 30 reps; row kernels 20 warmup + 100 reps; attention 3 warmup + 10 reps (naive attention at S=512, B=128 takes ~1.2–1.5 s per launch).
- Report **median**, plus min/max range in `results/run.json`. Overhead % = (inv_median − std_median)/std_median × 100. Both variants timed at identical shapes on the same device in the same session.

## 7. End-to-end replay (experiment 3b)

Block config: seq=128, hidden=4096, H=32, D=128, FFN=11008, fp16 storage/fp32 compute, seed 20261119. Run the block at B=1 and at B=7 (same probe tokens at row 0, different surrounding padding rows drawn from the same seed stream). Compare the B=1 output (128×4096 = 524288 fp16 elements) bitwise against the corresponding slice of the B=7 output. Repeat the B=1 run to check temporal determinism. Run the identical protocol with standard (cuBLASLt + heuristic) kernels as the **control**: it must diverge (and does).

## 8. Validation and negative controls

`tests/validate_results.py` (stdlib, exit 0 = pass) runs 52 checks: environment/provenance (SM120 RTX 5060 Ti, driver, arch, CVD=0 in every mode header; SHA-256 of executed sources and binary), selftest controls, experiment-1 counts and M=1 self-comparison sanity, 28/28 acceptance, targeted/zero-diff negative controls, the independent Python fp64 oracle (bitwise), proven-bound checks with perturbation rejection, cost-record sanity, and bitwise block replay.

Negative controls retained as evidence: single-bit-flip comparator detection (fp32+fp16), bound-checker accept/reject, RNG determinism, standard kernels failing the acceptance harness on targeted cases (9396 and 9644 differing elements), and the standard block's 35.55% divergence.

## 9. Environment and etiquette

Server `me4me` (Ubuntu 24.04, kernel 6.8.0-142, 12 cores, 62 GB RAM), CUDA 13.3 (nvcc V13.3.73), gcc 13.3, python 3.12.3 stdlib only. GPU 0 = RTX 5060 Ti 16 GB (cc 12.0, 36 SMs), driver 610.43.03. Build: `nvcc -std=c++17 -O3 -arch=sm_120 cuda/bik.cu -o bik -lcublasLt -lcublas`. Every GPU execution: `CUDA_VISIBLE_DEVICES=0` under `flock /tmp/requant-new-studies-gpu.lock`; long runs detached via nohup and polled. GPU 1 (CMP 50HX, Turing, cc 7.5) never touched. No downloads; no services disturbed; no leftover processes.