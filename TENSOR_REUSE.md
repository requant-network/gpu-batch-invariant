# Reusing Tensor Core weight fragments on SM120

At batch 128, the FP16 up projection fell from 4.89 ms to 0.95 ms on the
RTX 5060 Ti. The new kernel produced identical FP16 output bits to the
original Tensor Core kernel. The benefit depended on shape and batch. The
4096-square GEMM at batch 64 took about 26% longer with the new kernel.

## What changed

One warp loads a 16×16 weight fragment and applies it to four independent
16-row output tiles. Each tile keeps its own FP32 accumulator, visits K in
the original order and rounds to FP16 at the end. The trial uses this path
when M is a multiple of 64 and uses the original single-tile path for the
other batch sizes. The existing `tensor` mode retains its single-tile path.

The compiled single-tile kernel had the same 272 SM120 instruction
encodings as the original retained benchmark binary. The new kernel also
contains FP32-accumulating HMMA instructions.
[Instruction comparison](results/tensor-reuse/prior-kernel.txt) ·
[Build and GPU provenance](results/tensor-reuse/provenance.txt)

## Exact comparisons

Each of two processes compared every active output element between the
single-tile and four-tile paths for three shapes, two seeds and batches
64, 128 and 256. All 18 comparisons per process passed, covering 17,203,200
FP16 elements. Six batch-7 controls matched when the new launcher selected
the original path. The output buffers used different prefills before each
side of a comparison. [Forward process](results/tensor-reuse/reuse-forward.jsonl) ·
[Reverse process](results/tensor-reuse/reuse-reverse.jsonl)

The synthetic block used weight reuse for all four GEMMs at 128 tokens
per sequence. For both seeds its full singleton output matched the
single-tile block on all 524,288 elements. The first and last positions in
a seven-sequence batch and the repeated singleton also matched exactly.
[Block replay](results/tensor-reuse/reuse-block.jsonl)

CUDA Compute Sanitizer reported zero memory access errors in the new
kernel in both modes. The rebuilt comparator self-test also passed.
[Memcheck](results/tensor-reuse/sanitizer.txt) · [Self-test](results/tensor-reuse/selftest.json)

## Resident GEMM latency

Ranges show the two process medians in microseconds. Each process used
10 warmup launches and 30 CUDA-event measurements per path with weights
and activations already resident. One process timed cuBLASLt, the
single-tile path and then the four-tile path. The other used the reverse
order. Transfers, allocations and whole-block runtime are outside this
measurement.

| Shape | Batch | cuBLASLt µs | One tile µs | Four tiles µs | One tile / four tiles |
|---|---:|---:|---:|---:|---:|
| 4096→4096 | 64 | 60.4 | 221.8–222.1 | 279.0–283.4 | 0.78–0.80× |
| 4096→4096 | 128 | 104.1–105.5 | 379.8–380.2 | 317.3–318.3 | 1.19–1.20× |
| 4096→4096 | 256 | 198.8–202.6 | 750.5–754.3 | 620.2–621.8 | 1.21× |
| 4096→11008 | 64 | 246.2–247.5 | 2292.5 | 489.1–491.8 | 4.66–4.69× |
| 4096→11008 | 128 | 258.3–259.4 | 4886.1–4890.4 | 945.0–946.6 | 5.17× |
| 4096→11008 | 256 | 485.1–485.2 | 10110.7–10124.4 | 2210.6–2237.0 | 4.52–4.58× |
| 11008→4096 | 64 | 226.1–227.5 | 1305.0–1331.7 | 878.0–879.1 | 1.49–1.51× |
| 11008→4096 | 128 | 254.7–255.2 | 3824.3–3838.8 | 906.6–906.8 | 4.22–4.23× |
| 11008→4096 | 256 | 519.8–521.5 | 9162.9–9166.4 | 1776.7–1779.1 | 5.15–5.16× |

The batch-128 up projection still took 3.65–3.66× the cuBLASLt median.
Use the measured shape and batch when selecting an experimental path.
The performance and parity evidence apply to the synthetic FP16 kernels
and block on one card. A quantized Qwen model and a second physical GPU
remain separate qualification requirements.

## Reproduce

Select the intended RTX explicitly with `CUDA_VISIBLE_DEVICES` before
running on a multi-GPU host. Use the shared lock on the lab server.

```bash
make
flock /tmp/requant-new-studies-gpu.lock -c './bik tensor-reuse && BIK_REUSE_TIMING_REVERSED=1 ./bik tensor-reuse && ./bik tensor-reuse-block'
```
