# Fixed-order Tensor Core replay on SM120

The FP16 Tensor Core kernel reproduced every tested probe bitwise across
batch sizes and positions on one RTX 5060 Ti. Its resident GEMM median was
1.38–10.94× faster than the original scalar fixed-order kernel and took
1.03–18.81× as long as cuBLASLt across the 15 measured shapes and batches.
A synthetic transformer-like block also replayed bitwise within the Tensor
Core path. Switching from the scalar path to Tensor Cores changed 51–53%
of that block's output elements.

## Kernel and replay checks

Each warp computes one 16×16 output tile, taking K tiles in ascending order
with no split-K. Inputs are FP16 and the `wmma` accumulator is FP32. Outputs
are rounded to FP16. K and N are multiples of 16. The input storage includes
whole batch tiles and the kernel writes only the active output rows.
Disassembly of the executed binary contains `HMMA.16816.F32` in the new
kernel. [Build and device provenance](results/tensor/provenance.txt)

- **GEMM replay.** Each of two processes ran three shapes, two seeds and
  batches 1, 2, 7, 32 and 128. All 30 combinations per process matched the
  singleton probe at both the first and last active row. The last row crosses
  a tile boundary at batch 32 and 128. Every whole active batch also matched
  on repeated execution after two different output prefills.
  [First process](results/tensor/tensor-1.jsonl) · [Second process](results/tensor/tensor-2.jsonl)
- **Numerical reference.** The seeded inputs are multiples of 2^-8. All
  exact FP64 partial sums fit within 32 significand bits for these shapes.
  The reference rounds the resulting FP64 dot products directly to FP16.
  Out of 38,400 probe elements across the six distinct cases, two differed
  by one FP16 ULP, one per `ffn_down` seed. The largest absolute error divided
  by `max(1, abs(exact))` was 0.0004812 against a harness limit of 0.002.
  This is a numerical comparison for the seeded grid. The within-path replay
  checks compare storage bits directly.
- **Block replay.** The existing block kept its frozen RMSNorm and attention
  kernels and used Tensor Cores for its four GEMMs. At sequence length 128,
  both seeds matched all 524,288 finite probe outputs across a singleton, the
  first and last positions in a seven-sequence batch, and a repeated singleton.
  The scalar singleton differed from the Tensor Core singleton at 268,796
  elements for seed 20261119 and 279,061 for seed 777. The legacy control
  reproduced the earlier scalar replay and 186,393 standard-path differences.
  [Tensor Core block](results/tensor/tensor-block.jsonl) · [Legacy control](results/tensor/legacy-block.jsonl)
- **Validation.** CUDA Compute Sanitizer memcheck reported zero errors in
  the Tensor Core kernel during both new modes. The rebuilt comparator
  self-test and all 52 checks on the original archived results passed.
  [Memcheck](results/tensor/sanitizer.txt) · [Comparator self-test](results/tensor/selftest.json)

## Resident GEMM latency

The ranges are the two process medians, in microseconds. Each median uses
30 CUDA-event timings after 10 warmup launches. Transfers, allocation and
cuBLASLt algorithm selection are outside the timed boundary. All paths use
the same FP16 buffers and seed 20261119. No whole-block timing was taken.

| Shape | Batch | cuBLASLt µs | Scalar µs | Tensor Core µs |
|---|---:|---:|---:|---:|
| 4096→4096 | 1 | 37.1–37.2 | 626.4–630.6 | 58.0–58.1 |
| 4096→4096 | 2 | 55.9–56.4 | 631.0–632.9 | 58.3–58.4 |
| 4096→4096 | 7 | 56.0–56.5 | 630.2–636.3 | 58.2–58.3 |
| 4096→4096 | 32 | 34.4–35.3 | 612.2–621.8 | 97.8–97.8 |
| 4096→4096 | 128 | 104.2–104.5 | 2302.5–2302.7 | 380.4–381.8 |
| 4096→11008 | 1 | 239.2–239.3 | 1740.1–1740.6 | 453.7–456.9 |
| 4096→11008 | 2 | 262.5–264.1 | 1740.3–1742.0 | 456.1–456.4 |
| 4096→11008 | 7 | 274.1–275.2 | 1740.9–1741.0 | 455.9–457.3 |
| 4096→11008 | 32 | 258.2–259.4 | 1740.6–1740.9 | 1034.8–1035.0 |
| 4096→11008 | 128 | 258.9–260.1 | 6743.1–6748.8 | 4868.4–4877.7 |
| 11008→4096 | 1 | 221.7–222.1 | 1859.4–1859.5 | 273.0–273.1 |
| 11008→4096 | 2 | 219.5–220.1 | 1859.9–1860.2 | 273.8–274.9 |
| 11008→4096 | 7 | 220.3–221.2 | 1858.7–1859.1 | 273.3–273.5 |
| 11008→4096 | 32 | 225.5–225.6 | 1858.8–1858.8 | 269.1–271.6 |
| 11008→4096 | 128 | 255.5–256.3 | 6964.1–6966.0 | 3827.9–3844.7 |

## Reproduction and decision

Build with `make` on CUDA 13.3. Select the study GPU explicitly before running
the benchmark. Use the shared lock when reproducing on the lab server.

```bash
make
cuobjdump --dump-sass bik
flock /tmp/requant-new-studies-gpu.lock -c './bik selftest && ./bik tensor && ./bik tensor-block && ./bik block'
python3 tests/validate_results.py  # Validate the original archived study.
```

Qualify the Tensor Core path as a distinct numerical profile before using it
for replay. Scope is one RTX 5060 Ti and the specified synthetic FP16 kernels
and block. Real-model fidelity, other precisions and bitwise replay on an
independent GPU remain open. The batch-128 FFN cases quantify the remaining
performance gap for this implementation.
