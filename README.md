# gpu-batch-invariant

**Batch-invariant deterministic inference kernels on SM120: measuring batch-induced nondeterminism and its cost.**

Status: **COMPLETE — all experiments ran on the RTX 5060 Ti (SM120, Blackwell); 52/52 machine-validation checks pass; invariant kernels bitwise-replay across batch sizes; end-to-end block replay bitwise-equal.**

Requant Network research campaign `requant-new-studies-20261119` (bounded spike). All documentation and code comments in English.

## Main result

Standard GPU inference kernels are **not bitwise stable when the same input row is computed inside different batch sizes**: 37/140 probe comparisons differed, driven by cuBLASLt algorithm/split-K switching (up to **42.7% of a row's elements, 14581 ULP in fp16**) and batch-dependent reduction configs in softmax. Fixed-reduction-order ("batch-invariant") variants eliminate this completely — **28/28 acceptance cases bitwise-identical across batches {1, 2, 7, 32, 128}**, verified against an independent exact fp64 CPU oracle — and an end-to-end transformer-like block replays **bitwise-equal (524288/524288 elements)** under different batch padding, where the standard path diverges on **35.55% of elements (max 32254 ULP)**.

Cost of determinism (resident median latency): naive fixed-order GEMM **4.1×–27.2× slower** than cuBLASLt (no tensor cores); row kernels (softmax/RMSNorm) and attention with frozen configs are **free or faster at batch 128** (−34%…−43% for rows) and cost +16%…+110% at batch 1.

## What has actually run

Everything below executed on the shared GPU server `me4me` (192.168.0.70), **GPU 0 only** — NVIDIA GeForce RTX 5060 Ti, compute capability 12.0 (SM120), driver 610.43.03, CUDA 13.3 — compiled `-arch=sm_120`, pinned `CUDA_VISIBLE_DEVICES=0`, every GPU run serialized under `flock /tmp/requant-new-studies-gpu.lock`. GPU 1 (CMP 50HX, Turing) was never used. GPU name + compute cap + driver are stamped in **every** results JSON header.

| experiment | what ran | records | outcome |
|---|---|---|---|
| selftest | comparator/bound-checker/RNG negative controls | 7 checks | all pass |
| exp1 `measure` | cuBLASLt GEMM fp16/bf16/fp32 (3 shapes), softmax, RMSNorm, tiled attention; probe row at batches {1,2,7,32,128}, 2 seeds; bitwise memcmp + ULP; run-to-run control | 140 | 37 differ; 0 run-to-run diffs |
| exp2 `invariant` | fixed-order GEMM/softmax/RMSNorm/attention acceptance across all batches + exact fp64 oracle + tolerances + 4 negative controls | 33 | 28/28 acceptance PASS |
| exp3a `cost` | resident median latency, invariant vs standard, same shapes (warmup + repeated launches) | 28 | GEMM +309…+2621%; rows −43…+110% |
| exp3b `block` | RMSNorm→QKV→attention→o→RMSNorm→FFN block, bitwise replay B=1 vs B=7 | 1 | invariant 0 diff; standard 35.55% diff |
| validation | `tests/validate_results.py` incl. **independent pure-Python fp64 oracle** (bitwise match) | 52 checks | **52/52 PASS** (server + local) |

Raw evidence: `results/run.json` (merged), `results/raw_*.json` (per-mode), `results/validation.txt`. Full measured tables: [RESULTS.md](RESULTS.md). Method contract: [METHODS.md](METHODS.md). Conclusions and next decisions: [FINAL_REPORT.md](FINAL_REPORT.md).

## Repository layout

```
cuda/bik.cu            all kernels + 5 modes (selftest|measure|invariant|cost|block), JSONL out
run.py                 stdlib orchestrator: env metadata, SHA-256 source hashes, --finalize
tests/validate_results.py  52 machine checks incl. independent Python fp64 oracle
results/               measured JSON + validation transcript (committed)
Makefile               sm_120 build
```

## Reproduce

On the GPU server (Ubuntu 24.04, CUDA 13.3, python3.12 stdlib):

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_120 cuda/bik.cu -o bik -lcublasLt -lcublas
flock /tmp/requant-new-studies-gpu.lock -c 'CUDA_VISIBLE_DEVICES=0 python3 run.py --all'
python3 tests/validate_results.py        # exit 0 = 52/52 checks pass
```

`run.py --finalize` re-stamps GPU provenance into existing raw results without GPU work. No downloads, no third-party packages: synthetic seeded data (splitmix64, exact 2^-8 grid) + exact CPU oracles.

## Scope limitations (explicit)

- **Bounded spike**: 3 LLM-inspired shapes (4096×4096, 4096→11008, 11008→4096), seq 128/512, batches ≤128, VRAM < 4 GB. Not a full model, not a serving stack.
- The invariant GEMM is a **deliberately naive** fixed-order design (32×32 shared tile, scalar fp32 FMA, ascending K, no split-K, batch padded to tile multiple). Its 4–27× overhead is an upper bound for simple designs, **not** the cost of a tuned tensor-core batch-invariant GEMM (future work).
- cuBLASLt findings are specific to CUDA 13.3 / driver 610.43.03 / SM120 heuristics on this device; other libraries, versions or GPUs may switch algorithms differently. Zero-difference cases are reported as valid findings, not forced failures.
- Attention is a naive tiled score·V path (fp16 I/O, fp32 compute), not FlashAttention; its measured batch-insensitivity here is partly fp16 output rounding absorbing order noise.
- Two seeds per claim; exactness claims rest on a proven-bound fp64 oracle plus an independent Python re-derivation, not on float tolerance alone.

## Related work (cited, not re-run)

- [`requant-network/gpu-verifiable-inference`](https://github.com/requant-network/gpu-verifiable-inference) — exact fused inference proofs on SM120. **Determinism is a prerequisite for proof replay**: a verifier replaying a fused kernel must reproduce the prover's bits; this study quantifies how batch padding alone breaks that replay (35.55% of block elements) and shows fixed-order kernels restore bitwise replay.
- [`requant-network/gpu-fp4-research`](https://github.com/requant-network/gpu-fp4-research) — low-precision GEMM accuracy. Complementary axis: they study precision-induced error at fixed shape; we study order-induced error at fixed precision across batch sizes, with the same exact-oracle methodology.

## License

Research artifact, Requant Network campaign. See FINAL_REPORT.md for conclusions and next research decisions.