# FINAL REPORT — Batch-invariant deterministic inference kernels on SM120

Campaign: `requant-new-studies-20261119` (bounded spike). Target: RTX 5060 Ti, SM120 (Blackwell), CUDA 13.3, driver 610.43.03. All numbers measured on `me4me`; 52/52 machine-validation checks pass, including an independent pure-Python fp64 oracle that matches the C++ exact oracle bitwise.

## 1. Conclusions

**C1 — Batch-size-induced nondeterminism is real, large, and mechanism-specific.** 37/140 probe-row comparisons differ when the same row moves from batch 1 into batches {2,7,32,128}. The dominant mechanism is **cuBLASLt heuristic switching**: algorithm, tile, split-K factor and reduction scheme all change with M. Worst case: fp16 FFN-up GEMM at M=2/7 switches to a splitK=16 OUTPUT_TYPE-reduction kernel and **42.7% of the row's elements change, by up to 14581 ULP** (bf16: up to 31721 ULP). Even without split-K, a tile/algo switch flips 1-ULP bits (fp16/bf16 FFN-down at M=2/7/32). fp32 GEMMs differ by ≤16 ULP on ≤6.45% of elements. Batch-dependent block-size heuristics in softmax flip ~88% of a width-11008 row by ≤2 ULP fp32.

**C2 — It is batch-induced, not temporal.** 0/90 same-shape repeat launches differed (including split-K INPLACE reductions). A serving stack that pins batch size gets run-to-run determinism from cuBLASLt on this device; a stack that pads/changes batch size does not get cross-batch determinism.

**C3 — Fixed reduction order eliminates it completely, provably.** 28/28 invariant acceptance cases are bitwise identical across all five batch sizes and both seeds (0 differing elements): fixed-tile ascending-K GEMM with batch padded to the tile multiple, frozen-config two-pass softmax/RMSNorm, frozen-config attention. Accuracy is preserved: fp16/bf16 invariant GEMM matches cuBLAS batch-1 bitwise in 11/12 cases (worst 1 ULP); fp32 within 288 ULP / 1.43e-3 absolute; all errors inside the proven γ_K·Σ|ab| bound with ≥3 orders of magnitude margin, verified against an exact fp64 oracle re-derived independently in Python.

**C4 — End-to-end consequence: batch padding alone changes model outputs.** The transformer-like block (RMSNorm→QKV→attention→o→RMSNorm→FFN) with standard kernels diverges on **35.55% of 524288 output elements (max 32254 ULP fp16)** between B=1 and B=7. With invariant kernels the replay is **bitwise equal**, and repeat runs are bitwise equal. This is exactly the property required for proof replay in verifiable inference (see §4).

**C5 — The cost of determinism is design-dependent, not intrinsic.**
- Naive fixed-order GEMM (scalar FMA, no tensor cores): **4.1×–27.2× slower** (+309%…+2621%) than cuBLASLt. This is the price of the simplest provably invariant design; it is an upper bound, not a verdict on tuned invariant tensor-core kernels.
- Row kernels: freezing the config costs +75…+110% at batch 1 but is **34–43% faster at batch 128** (the standard heuristic's 128-thread choice is worse than the frozen 256). Determinism here is free or beneficial.
- Attention (frozen 128 threads): −40% at batch 1, +16…+33% at batch 128.

## 2. Negative results (valid findings, retained)

- **fp16/bf16 hidden_square GEMM: zero batch dependence.** The heuristic keeps one non-split algorithm for all M∈{1..128}; all 20 records bitwise equal. Not every shape is affected.
- **RMSNorm: zero batch dependence** on both widths/seeds despite the changing block heuristic — sum-of-squares of the 2^-8 grid is coarse enough that all reduction trees round identically.
- **Naive attention: zero batch dependence** for these cohorts — fp16 output storage absorbs the sub-ULP fp32 order noise from config changes. (A fp32-output attention would likely expose it; not tested here.)
- **Softmax width 4096: zero batch dependence**; only width 11008 exposes ≤2-ULP differences. Batch sensitivity of row kernels is width- and data-dependent.
- **cuBLASLt fp32 is more accurate than the naive invariant GEMM** (max err vs exact 4.6e-5 vs 1.43e-3): fixed ascending order is not the most accurate order; it is the *reproducible* one. Both are within the proven bound.
- Initial negative-control selection (hidden_square fp16, softmax 4096) happened to pick zero-difference cases; the harness correctly passed them. Controls were retargeted to cases where experiment 1 observed differences, and those correctly fail (9396 / 9644 differing elements). Lesson: negative controls must be chosen from observed-failure cases, and zero-difference cases must be allowed to pass.

## 3. Scope and honesty notes

Bounded spike: 3 GEMM shapes, 2 row widths, 2 sequence lengths, batches ≤128, 2 seeds, VRAM < 4 GB, one GPU (SM120 only; the Turing CMP 50HX was never used). Findings are specific to CUDA 13.3 / driver 610.43.03 cuBLASLt heuristics on this device. The invariant GEMM is deliberately naive; no tuned tensor-core invariant kernel was built. Attention is a naive tiled path, not FlashAttention. No numbers were extrapolated; every table in RESULTS.md is generated from the committed results JSON.

## 4. Relation to prior Requant work (cited, not re-run)

- **gpu-verifiable-inference** (exact fused inference proofs on SM120): proof replay requires the verifier to reproduce the prover's bits exactly. C4 shows batch padding breaks that replay for standard kernels (35.55% of elements) and that fixed-order kernels restore it. Batch-invariance should be treated as a **prerequisite contract** for any fused-kernel proof system deployed under variable batching.
- **gpu-fp4-research** (low-precision GEMM accuracy): complementary error axis — precision-induced error at fixed shape vs order-induced error at fixed precision across batches. The exact-oracle + proven-bound methodology used here transfers directly to FP4/FP6 accuracy claims.

## 5. Next research decisions

1. **Tuned batch-invariant tensor-core GEMM on SM120** (highest value): fixed tile/split-K=1 with HMMA and a fixed K-order (or deterministic split-K with a fixed reduction tree), targeting ≤20% overhead vs cuBLASLt. The 4–27× measured here is the naive-design baseline to beat.
2. **Pin cuBLASLt algo per shape** as a cheap mitigation: force one algo/split-K/reduction for all M (algo config is serializable). Measure whether a single forced algo retains acceptable latency across M∈{1..128} — this may give cross-batch determinism without custom kernels.
3. **Extend the nondeterminism census** to FlashAttention-style kernels, fp32-output attention, LayerNorm, and MoE routing (where batch-dependent top-k ties can change discrete decisions, not just bits).
4. **Integrate with gpu-verifiable-inference**: add a batch-invariance acceptance gate to the proof-replay pipeline; replay proofs under padded batches as a standard negative control.
5. **Version/library sensitivity sweep**: repeat experiment 1 across CUDA/driver versions and a second GPU architecture to map how heuristic switching moves (the mechanism is library-version-specific).

## 6. Verification status

- Build: `-arch=sm_120` only; binary SHA-256 recorded per mode.
- 52/52 validation checks pass on the server and on the Windows host (independent re-run of the validator).
- Independent Python fp64 oracle: bitwise match on 4096 elements + exact Σ|ab|; bound-checker rejects out-of-bound perturbation.
- Acceptance: 28/28 bitwise-invariant cases; block replay 524288/524288 bitwise equal; repeat-run equal.
- Negative controls: all detect/reject as designed; standard block diverges as expected.
- Provenance: GPU name + compute cap + driver + arch + CVD=0 stamped in every results JSON; SHA-256 of all executed sources; UTC timestamps; no leftover GPU processes; no services touched.