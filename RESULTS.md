# RESULTS — measured on me4me, GPU 0 (RTX 5060 Ti, SM120)

All numbers below were produced by `cuda/bik.cu` (built `-arch=sm_120`) on the server and machine-validated by `tests/validate_results.py` (**52/52 checks passed**, see `results/validation.txt`). Nothing is extrapolated.

## Environment

| field | value |
|---|---|
| GPU used | NVIDIA GeForce RTX 5060 Ti (GPU 0), compute capability 12.0, 36 SMs, 15885 MiB |
| NVIDIA driver | 610.43.03 |
| CUDA | 13.3 (nvcc V13.3.73, driver API 13030) |
| Build | `nvcc -std=c++17 -O3 -arch=sm_120 cuda/bik.cu -o bik -lcublasLt -lcublas` |
| Pinning | `CUDA_VISIBLE_DEVICES=0`; every GPU run under `flock /tmp/requant-new-studies-gpu.lock` |
| Host / OS | me4me / Linux-6.8.0-142-generic-x86_64-with-glibc2.39 |
| Python | 3.12.3 (stdlib only) |
| Run UTC | 2026-11-19T16:16:29Z (invariant-mode rerun n/a) |
| Other GPU | GPU 1 = CMP 50HX (cc 7.5, Turing) — **never used** (hosts other services) |

cuBLASLt paths: `CUBLAS_COMPUTE_32F` with FP16/BF16 storage (SM120 tensor-core kernels selected by the on-device heuristic) and FP32 storage (FFMA CUDA-core kernels; TF32 not requested). The binary contains only SM120 SASS, so it cannot execute on the Turing card at all. Selected algo/tile/split-K IDs are recorded per launch below.

## Experiment 1 — batch-induced bitwise nondeterminism (standard kernels)

Protocol: fixed probe row (row 0) computed alone (batch 1) and inside batches {2, 7, 32, 128}; outputs compared **bitwise** (memcmp of storage bits) against the batch-1 result; ULP distance on differing elements. Two input cohorts: seeds 20261119 and 777. 140 records total.

**Headline: 37/140 records differ from the batch-1 probe.** Breakdown by kernel family:

| kernel family | records | differing | max diff fraction | max ULP |
|---|---|---|---|---|
| gemm_cublaslt | 90 | 34 | 0.4268 | 31897 |
| softmax_std | 20 | 3 | 0.8927 | 2 |
| rmsnorm_std | 10 | 0 | 0.0000 | 0 |
| attn_std | 20 | 0 | 0.0000 | 0 |

Run-to-run control: every cuBLASLt GEMM was also launched twice at the **same** shape; 0/90 records differed — cuBLASLt is run-to-run deterministic here (even with split-K INPLACE reduction). The nondeterminism is **batch-induced**, not temporal.

### 1a. cuBLASLt GEMM — full evidence table (90 records)

Shapes: hidden_square K=N=4096; ffn_up K=4096,N=11008; ffn_down K=11008,N=4096. `red` = CUBLASLT_REDUCTION_SCHEME (0=NONE, 1=INPLACE, 2=COMPUTE_TYPE, 4=OUTPUT_TYPE). `diff` = probe-row elements differing from batch 1.

| dtype | shape | M | seed | algo | tile | splitK | red | diff | frac | max ULP |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | hidden_square | 1 | 20261119 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 2 | 20261119 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 7 | 20261119 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 32 | 20261119 | 67 | 16 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 128 | 20261119 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 1 | 777 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 2 | 777 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 7 | 777 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 32 | 777 | 67 | 16 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | hidden_square | 128 | 777 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | ffn_up | 1 | 20261119 | 13 | 0 | 16 | 2 | 0 | 0.0000 | 0 |
| fp16 | ffn_up | 2 | 20261119 | 21 | 5 | 16 | 4 | 4698 | 0.4268 | 1262 |
| fp16 | ffn_up | 7 | 20261119 | 21 | 5 | 16 | 4 | 4698 | 0.4268 | 1262 |
| fp16 | ffn_up | 32 | 20261119 | 67 | 489 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | ffn_up | 128 | 20261119 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | ffn_up | 1 | 777 | 13 | 0 | 16 | 2 | 0 | 0.0000 | 0 |
| fp16 | ffn_up | 2 | 777 | 21 | 5 | 16 | 4 | 4591 | 0.4171 | 14581 |
| fp16 | ffn_up | 7 | 777 | 21 | 5 | 16 | 4 | 4591 | 0.4171 | 14581 |
| fp16 | ffn_up | 32 | 777 | 67 | 489 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | ffn_up | 128 | 777 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | ffn_down | 1 | 20261119 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | ffn_down | 2 | 20261119 | 23 | 18 | 1 | 0 | 1 | 0.0002 | 1 |
| fp16 | ffn_down | 7 | 20261119 | 23 | 18 | 1 | 0 | 1 | 0.0002 | 1 |
| fp16 | ffn_down | 32 | 20261119 | 21 | 15 | 1 | 0 | 1 | 0.0002 | 1 |
| fp16 | ffn_down | 128 | 20261119 | 21 | 15 | 3 | 1 | 1507 | 0.3679 | 190 |
| fp16 | ffn_down | 1 | 777 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp16 | ffn_down | 2 | 777 | 23 | 18 | 1 | 0 | 1 | 0.0002 | 1 |
| fp16 | ffn_down | 7 | 777 | 23 | 18 | 1 | 0 | 1 | 0.0002 | 1 |
| fp16 | ffn_down | 32 | 777 | 21 | 15 | 1 | 0 | 1 | 0.0002 | 1 |
| fp16 | ffn_down | 128 | 777 | 21 | 15 | 3 | 1 | 1434 | 0.3501 | 815 |
| bf16 | hidden_square | 1 | 20261119 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 2 | 20261119 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 7 | 20261119 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 32 | 20261119 | 67 | 16 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 128 | 20261119 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 1 | 777 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 2 | 777 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 7 | 777 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 32 | 777 | 67 | 16 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | hidden_square | 128 | 777 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_up | 1 | 20261119 | 13 | 0 | 16 | 2 | 0 | 0.0000 | 0 |
| bf16 | ffn_up | 2 | 20261119 | 21 | 5 | 16 | 4 | 4582 | 0.4162 | 31721 |
| bf16 | ffn_up | 7 | 20261119 | 21 | 5 | 16 | 4 | 4582 | 0.4162 | 31721 |
| bf16 | ffn_up | 32 | 20261119 | 67 | 489 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_up | 128 | 20261119 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_up | 1 | 777 | 13 | 0 | 16 | 2 | 0 | 0.0000 | 0 |
| bf16 | ffn_up | 2 | 777 | 21 | 5 | 16 | 4 | 4595 | 0.4174 | 31594 |
| bf16 | ffn_up | 7 | 777 | 21 | 5 | 16 | 4 | 4595 | 0.4174 | 31594 |
| bf16 | ffn_up | 32 | 777 | 67 | 489 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_up | 128 | 777 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 1 | 20261119 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 2 | 20261119 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 7 | 20261119 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 32 | 20261119 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 128 | 20261119 | 21 | 15 | 3 | 1 | 1455 | 0.3552 | 370 |
| bf16 | ffn_down | 1 | 777 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 2 | 777 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 7 | 777 | 23 | 18 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 32 | 777 | 21 | 15 | 1 | 0 | 0 | 0.0000 | 0 |
| bf16 | ffn_down | 128 | 777 | 21 | 15 | 3 | 1 | 1500 | 0.3662 | 31897 |
| fp32 | hidden_square | 1 | 20261119 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | hidden_square | 2 | 20261119 | 11 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | hidden_square | 7 | 20261119 | 20 | 16 | 9 | 2 | 4 | 0.0010 | 1 |
| fp32 | hidden_square | 32 | 20261119 | 20 | 16 | 3 | 1 | 0 | 0.0000 | 0 |
| fp32 | hidden_square | 128 | 20261119 | 20 | 24 | 9 | 1 | 4 | 0.0010 | 1 |
| fp32 | hidden_square | 1 | 777 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | hidden_square | 2 | 777 | 11 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | hidden_square | 7 | 777 | 20 | 16 | 9 | 2 | 1 | 0.0002 | 1 |
| fp32 | hidden_square | 32 | 777 | 20 | 16 | 3 | 1 | 0 | 0.0000 | 0 |
| fp32 | hidden_square | 128 | 777 | 20 | 24 | 9 | 1 | 1 | 0.0002 | 1 |
| fp32 | ffn_up | 1 | 20261119 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | ffn_up | 2 | 20261119 | 11 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | ffn_up | 7 | 20261119 | 11 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | ffn_up | 32 | 20261119 | 20 | 18 | 5 | 1 | 2 | 0.0002 | 1 |
| fp32 | ffn_up | 128 | 20261119 | 20 | 24 | 4 | 1 | 1 | 0.0001 | 1 |
| fp32 | ffn_up | 1 | 777 | 13 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | ffn_up | 2 | 777 | 11 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | ffn_up | 7 | 777 | 11 | 0 | 1 | 0 | 0 | 0.0000 | 0 |
| fp32 | ffn_up | 32 | 777 | 20 | 18 | 5 | 1 | 2 | 0.0002 | 1 |
| fp32 | ffn_up | 128 | 777 | 20 | 24 | 4 | 1 | 2 | 0.0002 | 1 |
| fp32 | ffn_down | 1 | 20261119 | 13 | 0 | 6 | 2 | 0 | 0.0000 | 0 |
| fp32 | ffn_down | 2 | 20261119 | 11 | 0 | 1 | 0 | 107 | 0.0261 | 4 |
| fp32 | ffn_down | 7 | 20261119 | 20 | 16 | 9 | 2 | 138 | 0.0337 | 8 |
| fp32 | ffn_down | 32 | 20261119 | 20 | 16 | 30 | 1 | 221 | 0.0540 | 8 |
| fp32 | ffn_down | 128 | 20261119 | 20 | 24 | 9 | 1 | 138 | 0.0337 | 8 |
| fp32 | ffn_down | 1 | 777 | 13 | 0 | 6 | 2 | 0 | 0.0000 | 0 |
| fp32 | ffn_down | 2 | 777 | 11 | 0 | 1 | 0 | 117 | 0.0286 | 16 |
| fp32 | ffn_down | 7 | 777 | 20 | 16 | 9 | 2 | 169 | 0.0413 | 4 |
| fp32 | ffn_down | 32 | 777 | 20 | 16 | 30 | 1 | 264 | 0.0645 | 16 |
| fp32 | ffn_down | 128 | 777 | 20 | 24 | 9 | 1 | 169 | 0.0413 | 4 |

Observed mechanisms:

- **fp16/bf16 ffn_up**: batch 1 uses algo 13 (splitK=16, COMPUTE_TYPE reduction); batches 2/7 switch to algo 21 tile 5 (splitK=16, **OUTPUT_TYPE** reduction) → 41.6–42.7% of the 11008 probe elements differ, up to **14581 ULP (fp16)** / **31721 ULP (bf16)**. Batches 32/128 revert to splitK=1 and match batch 1 bitwise.
- **fp16/bf16 ffn_down**: algo/tile switches at every batch even without split-K (1-element, 1-ULP diffs at M=2/7/32); M=128 uses splitK=3 INPLACE → 1434–1500 elements differ, up to 815 ULP (fp16) / 31897 ULP (bf16).
- **fp32 hidden_square**: splitK=9 at M=7/128 → 1–4 elements, 1 ULP. **fp32 ffn_down**: splitK ∈ {9,30} and even a different non-split algo at M=2 → up to 264 elements (6.45%), ≤16 ULP. **fp32 ffn_up**: splitK 4/5 at M=32/128 → 1–2 elements, 1 ULP.
- **fp16/bf16 hidden_square**: zero difference at every batch — the heuristic keeps one non-split algorithm for all M ∈ {1..128}. A valid zero-difference finding.

### 1b. Row softmax (fp32) — 20 records

Standard block-size heuristic: B≤2→1024 threads, B≤7→512, B≤32→256, else 128 (different reduction trees).

| N | B | seed | threads | diff | frac | max ULP |
|---|---|---|---|---|---|---|
| 4096 | 1 | 20261119 | 1024 | 0 | 0.0000 | 0 |
| 4096 | 2 | 20261119 | 1024 | 0 | 0.0000 | 0 |
| 4096 | 7 | 20261119 | 512 | 0 | 0.0000 | 0 |
| 4096 | 32 | 20261119 | 256 | 0 | 0.0000 | 0 |
| 4096 | 128 | 20261119 | 128 | 0 | 0.0000 | 0 |
| 4096 | 1 | 777 | 1024 | 0 | 0.0000 | 0 |
| 4096 | 2 | 777 | 1024 | 0 | 0.0000 | 0 |
| 4096 | 7 | 777 | 512 | 0 | 0.0000 | 0 |
| 4096 | 32 | 777 | 256 | 0 | 0.0000 | 0 |
| 4096 | 128 | 777 | 128 | 0 | 0.0000 | 0 |
| 11008 | 1 | 20261119 | 1024 | 0 | 0.0000 | 0 |
| 11008 | 2 | 20261119 | 1024 | 0 | 0.0000 | 0 |
| 11008 | 7 | 20261119 | 512 | 0 | 0.0000 | 0 |
| 11008 | 32 | 20261119 | 256 | 0 | 0.0000 | 0 |
| 11008 | 128 | 20261119 | 128 | 9644 | 0.8761 | 2 |
| 11008 | 1 | 777 | 1024 | 0 | 0.0000 | 0 |
| 11008 | 2 | 777 | 1024 | 0 | 0.0000 | 0 |
| 11008 | 7 | 777 | 512 | 0 | 0.0000 | 0 |
| 11008 | 32 | 777 | 256 | 9827 | 0.8927 | 2 |
| 11008 | 128 | 777 | 128 | 9661 | 0.8776 | 2 |

Only width 11008 shows differences (≤2 ULP fp32, ~88% of elements); width 4096 is bitwise stable across all configs for both cohorts — the reordered sums round to the same fp32 value there.

### 1c. RMSNorm (fp32) and naive tiled attention (fp16 I/O)

- RMSNorm: **0/10 records differ** (widths 4096/11008, both seeds, all batches). Same heuristic as softmax, but the sum-of-squares of 2^-8-grid inputs is coarse enough (multiples of 2^-16) that all reduction trees round identically.
- Attention (H=32, D=128, S∈{128,512}, fp16 storage, fp32 compute; thread heuristic 256/128/64 by batch): **0/20 records differ**. Config changes perturb fp32 scores/probabilities below the fp16 output rounding threshold — fp16 storage absorbs the order noise for these cohorts.

## Experiment 2 — batch-invariant variants: acceptance + accuracy

Acceptance test: probe row **bitwise identical across ALL of {1,2,7,32,128}**, both seeds, every shape.

| kernel | cases | passed | total differing elements |
|---|---|---|---|
| gemm_invariant | 18 | 18 | 0 |
| softmax_invariant | 4 | 4 | 0 |
| rmsnorm_invariant | 2 | 2 | 0 |
| attn_invariant | 4 | 4 | 0 |

**28/28 acceptance cases PASS, 0 differing elements.**

### 2a. Accuracy vs exact fp64 oracle and vs cuBLAS batch-1 (GEMM, 18 cases)

Oracle: exact fp64 CPU accumulation (provably exact for this input grid, METHODS.md §3), independently re-derived in pure Python and verified **bitwise equal** to the C++ oracle row (4096 elements). Bound: γ_K·Σ|a_k b_k| with γ_K = K·u/(1−K·u), u=2^-24.

| dtype | shape | seed | accept | max ULP vs cuBLAS M=1 | max abs vs cuBLAS M=1 | max err invariant vs exact | max err cuBLAS vs exact | proven bound |
|---|---|---|---|---|---|---|---|---|
| fp16 | hidden_square | 20261119 | PASS | 0 | 0 | 0.104431 | 0.104431 | 1.03988 |
| fp16 | hidden_square | 777 | PASS | 0 | 0 | 0.0933533 | 0.0933533 | 1.03591 |
| fp16 | ffn_up | 20261119 | PASS | 0 | 0 | 0.12384 | 0.12384 | 1.0486 |
| fp16 | ffn_up | 777 | PASS | 0 | 0 | 0.121658 | 0.121658 | 1.04558 |
| fp16 | ffn_down | 20261119 | PASS | 0 | 0 | 0.124756 | 0.124756 | 7.48716 |
| fp16 | ffn_down | 777 | PASS | 1 | 0.25 | 0.195145 | 0.195145 | 7.45639 |
| bf16 | hidden_square | 20261119 | PASS | 0 | 0 | 0.990936 | 0.990936 | 1.03992 |
| bf16 | hidden_square | 777 | PASS | 0 | 0 | 0.988541 | 0.988541 | 1.03583 |
| bf16 | ffn_up | 20261119 | PASS | 0 | 0 | 0.958282 | 0.958282 | 1.0486 |
| bf16 | ffn_up | 777 | PASS | 0 | 0 | 0.954025 | 0.954025 | 1.04557 |
| bf16 | ffn_down | 20261119 | PASS | 0 | 0 | 1.89359 | 1.89359 | 7.48727 |
| bf16 | ffn_down | 777 | PASS | 0 | 0 | 0.993973 | 0.993973 | 7.456 |
| fp32 | hidden_square | 20261119 | PASS | 23 | 0.000396729 | 0.000396729 | 1.52588e-05 | 1.03988 |
| fp32 | hidden_square | 777 | PASS | 14 | 0.000305176 | 0.000289917 | 1.52588e-05 | 1.03591 |
| fp32 | ffn_up | 20261119 | PASS | 23 | 0.000701904 | 0.000686646 | 1.52588e-05 | 1.0486 |
| fp32 | ffn_up | 777 | PASS | 20 | 0.000335693 | 0.000320435 | 1.52588e-05 | 1.04558 |
| fp32 | ffn_down | 20261119 | PASS | 62 | 0.00143433 | 0.00143433 | 3.05176e-05 | 7.48716 |
| fp32 | ffn_down | 777 | PASS | 288 | 0.00119019 | 0.00120544 | 4.57764e-05 | 7.45639 |

- fp16/bf16: invariant GEMM matches cuBLAS batch-1 **bitwise in 11/12 cases** (1 case: 1 ULP); both within the proven bound (max err 0.195 fp16 / 1.894 bf16 vs bounds 1.04–7.49).
- fp32: ≤288 ULP (≤1.43e-3 absolute) vs cuBLAS batch-1; max err vs exact 1.43e-3 ≪ bound 7.49. cuBLAS itself is more accurate (≤4.6e-5) — different order, same math, both inside the documented tolerance.

### 2b. Row-kernel and attention tolerances (vs standard batch-1)

| kernel | config | seed | accept | tolerance metric |
|---|---|---|---|---|
| softmax_invariant | N=4096, 256 threads | 20261119 | True | max_rel_vs_std_b1=0 |
| rmsnorm_invariant | N=4096, 256 threads | 20261119 | True | max_rel_vs_std_b1=0 |
| softmax_invariant | N=4096, 256 threads | 777 | True | max_rel_vs_std_b1=0 |
| rmsnorm_invariant | N=4096, 256 threads | 777 | True | max_rel_vs_std_b1=0 |
| softmax_invariant | N=11008, 256 threads | 20261119 | True | max_rel_vs_std_b1=0 |
| softmax_invariant | N=11008, 256 threads | 777 | True | max_rel_vs_std_b1=1.51752e-07 |
| attn_invariant | S=128, H=32, D=128, 128 threads | 20261119 | True | max_ulp_vs_std_b1=0, max_abs=0 |
| attn_invariant | S=128, H=32, D=128, 128 threads | 777 | True | max_ulp_vs_std_b1=0, max_abs=0 |
| attn_invariant | S=512, H=32, D=128, 128 threads | 20261119 | True | max_ulp_vs_std_b1=0, max_abs=0 |
| attn_invariant | S=512, H=32, D=128, 128 threads | 777 | True | max_ulp_vs_std_b1=0, max_abs=0 |

### 2c. Negative controls (must reject / must detect)

| control | outcome |
|---|---|
| selftest.single_bit_flip_detected | True |
| selftest.identical_buffers_pass | True |
| selftest.fp16_flip_detected | True |
| selftest.bound_checker_accepts_inside | True |
| selftest.bound_checker_rejects_outside | True |
| selftest.rng_deterministic | True |
| selftest.pass | True |
| negative_control_std_gemm (targeted) | std_fails_acceptance=True, total_diff_elements=9396 |
| negative_control_std_softmax (targeted) | std_fails_acceptance=True, total_diff_elements=9644 |
| negative_control_std_gemm (zero_diff_case) | std_fails_acceptance=False, total_diff_elements=0 |
| negative_control_std_softmax (zero_diff_case) | std_fails_acceptance=False, total_diff_elements=0 |

Targeted standard-kernel cases (where experiment 1 observed batch dependence) **fail** the acceptance harness — proving it detects nondeterminism. Zero-difference cases correctly **pass** — the harness does not force false positives. The Python oracle also rejects a deliberately perturbed value beyond the proven bound.

## Experiment 3a — latency cost (resident kernel, median of repeated launches)

Boundary: CUDA-event timing around the kernel launch only, resident inputs/outputs (no H2D/D2H transfers, no allocation). GEMM: 10 warmup + 30 reps; row kernels: 20 warmup + 100 reps; attention: 3 warmup + 10 reps. Median reported; full min/max in `results/run.json`.

| kernel | config | std median ms | invariant median ms | overhead % |
|---|---|---|---|---|
| gemm | dtype=fp16, shape=hidden_square, N=4096, M=1 | 0.0547 | 0.6345 | +1059.5 |
| gemm | dtype=fp16, shape=hidden_square, N=4096, M=128 | 0.1054 | 2.3254 | +2106.1 |
| gemm | dtype=fp16, shape=ffn_up, N=11008, M=1 | 0.2399 | 1.7461 | +627.9 |
| gemm | dtype=fp16, shape=ffn_up, N=11008, M=128 | 0.2610 | 6.7558 | +2488.8 |
| gemm | dtype=fp16, shape=ffn_down, N=4096, M=1 | 0.2229 | 1.8627 | +735.6 |
| gemm | dtype=fp16, shape=ffn_down, N=4096, M=128 | 0.2563 | 6.9728 | +2620.7 |
| gemm | dtype=bf16, shape=hidden_square, N=4096, M=1 | 0.0347 | 0.6250 | +1701.7 |
| gemm | dtype=bf16, shape=hidden_square, N=4096, M=128 | 0.1044 | 2.3129 | +2114.4 |
| gemm | dtype=bf16, shape=ffn_up, N=11008, M=1 | 0.2396 | 1.7486 | +629.8 |
| gemm | dtype=bf16, shape=ffn_up, N=11008, M=128 | 0.2607 | 6.7538 | +2490.6 |
| gemm | dtype=bf16, shape=ffn_down, N=4096, M=1 | 0.2218 | 1.8618 | +739.6 |
| gemm | dtype=bf16, shape=ffn_down, N=4096, M=128 | 0.2563 | 6.9748 | +2621.8 |
| gemm | dtype=fp32, shape=hidden_square, N=4096, M=1 | 0.1713 | 0.7055 | +311.8 |
| gemm | dtype=fp32, shape=hidden_square, N=4096, M=128 | 0.2820 | 2.6179 | +828.3 |
| gemm | dtype=fp32, shape=ffn_up, N=11008, M=1 | 0.4425 | 1.8117 | +309.4 |
| gemm | dtype=fp32, shape=ffn_up, N=11008, M=128 | 0.7157 | 6.9581 | +872.2 |
| gemm | dtype=fp32, shape=ffn_down, N=4096, M=1 | 0.4365 | 1.8881 | +332.5 |
| gemm | dtype=fp32, shape=ffn_down, N=4096, M=128 | 0.6842 | 7.3655 | +976.5 |
| softmax | N=4096, M=1 | 0.0038 | 0.0067 | +74.8 |
| rmsnorm | N=4096, M=1 | 0.0037 | 0.0075 | +100.9 |
| softmax | N=4096, M=128 | 0.0121 | 0.0077 | -36.0 |
| rmsnorm | N=4096, M=128 | 0.0117 | 0.0076 | -34.5 |
| softmax | N=11008, M=1 | 0.0056 | 0.0117 | +109.8 |
| softmax | N=11008, M=128 | 0.0401 | 0.0228 | -43.2 |
| attn | S=128, M=1 | 1.1290 | 0.6744 | -40.3 |
| attn | S=128, M=128 | 80.0565 | 92.7420 | +15.8 |
| attn | S=512, M=1 | 17.6641 | 10.4727 | -40.7 |
| attn | S=512, M=128 | 1157.7809 | 1544.0381 | +33.4 |

- **Invariant GEMM (naive fixed 32×32 tile, scalar FMA, no tensor cores): +309% … +2621% (4.1×–27.2× slower)** vs cuBLASLt. This is the price of the simplest provably fixed-order design, not a tuned one.
- **Row kernels (config freeze only): +75…+110% at batch 1, but −34…−43% at batch 128** — the frozen 256-thread config beats the standard heuristic's 128-thread choice at large batch. Determinism is free (even beneficial) here.
- **Attention (frozen 128 threads): −40% at batch 1, +16…+33% at batch 128** vs the 256/64-thread heuristic.

## Experiment 3b — end-to-end transformer-like block, bitwise replay

Block: RMSNorm → QKV GEMM → attention → o GEMM → RMSNorm → FFN (up GEMM → SiLU → down GEMM). Config: seq=128, hidden=4096, H=32, D=128, FFN=11008, fp16 storage / fp32 compute, seed 20261119. Probe: the B=1 output (524288 elements) compared against the same tokens computed inside batch B=7 (different surrounding padding).

| path | elements compared | differing | diff fraction | max ULP (fp16) | bitwise replay |
|---|---|---|---|---|---|
| invariant | 524288 | 0 | 0.000000 | 0 | **PASS** |
| standard | 524288 | 186393 | 0.355516 | 32254 | FAIL (expected) |

Invariant block: B=1 vs B=7 outputs bitwise equal (524288 elements), and a repeated run is also bitwise equal (`invariant_repeat_equal=True`). Standard block: **35.55% of elements differ, up to 32254 ULP** — batch padding alone changes the model output.

## Provenance hashes

| file | SHA-256 |
|---|---|
| cuda/bik.cu | `3293b39b0bc2e1d332656a4aedb42ba1d297ddb6972ebffaa01fc4dfbd4eeef2` |
| Makefile | `5b015f0b4a6c5a9adf742a13ddf4d5958b4a673cb14c35c9dbcca3dab5e7c5ec` |
| run.py | `4ed6790a23f8dd1e97cf65242c8cbb80954312529a0d2295c5475e47aa52f670` |
| tests/validate_results.py | `8a41dd70a8216ed80cef71911e8afd9c404c7a84167fb3379fd4f615c2faee39` |
| bik binary (all modes) | `f7bfa8755e59039e7b0844693cb702883abed616a3100569ace86c8771994fc9` |

Raw per-mode captures: `results/raw_{selftest,measure,invariant,cost,block}.json`; merged `results/run.json`; validator transcript `results/validation.txt` (52/52 PASS).
