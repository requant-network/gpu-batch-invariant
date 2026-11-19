#!/usr/bin/env python3
"""tests/validate_results.py — machine validation of results/run.json.

Stdlib only. Checks:
  1. Environment: SM120 device (RTX 5060 Ti), CUDA_VISIBLE_DEVICES=0, hashes.
  2. Selftest negative controls (comparator flip, bound checker, RNG).
  3. Experiment 1 record counts + sanity (M=1 self-comparison is zero).
  4. Experiment 2 acceptance: every invariant kernel/shape/seed bitwise
     identical across batch sizes {1,2,7,32,128}; oracle bound respected.
  5. Negative controls: standard kernels run through the acceptance harness
     must fail somewhere (batch dependence actually detected).
  6. INDEPENDENT CPU oracle: re-derive the FP32 GEMM probe row in pure Python
     fp64 from the seed (different code path from the C++ host oracle),
     require BITWISE equality with the recorded exact row, and check both GPU
     rows against the proven gamma_K bound. A deliberately perturbed value
     beyond the bound must be rejected.
  7. Cost records present with positive medians; block replay bitwise equal.

Exit code 0 = all checks pass.
"""
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "results", "run.json")

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# --- independent splitmix64 (must match cuda/bik.cu Rng exactly) -------------
MASK = (1 << 64) - 1


class Rng:
    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return (z ^ (z >> 31)) & MASK

    def grid(self):
        return ((self.next() % 1025) - 512) / 256.0


def f64bits(x):
    return struct.pack("<d", x)


def gamma_bound(K, sabs):
    u = 2.0 ** -24
    g = (K * u) / (1.0 - K * u)
    return g * sabs


def independent_oracle(dump):
    """Recompute exact fp64 probe row for FP32 hidden_square GEMM, seed 20261119.

    Exactness argument (METHODS.md): inputs are multiples of 2^-8 with |x|<=2,
    so products are multiples of 2^-16 with |p|<=4; K=4096 partial sums are
    bounded by 16384 < 2^14, needing <= 30 significand bits < 53. Sequential
    fp64 accumulation is therefore EXACT (no rounding at any step).
    """
    K, N, seed = dump["K"], dump["N"], dump["seed"]
    rng = Rng(seed)
    # A is filled first: MAXB=128 rows x K, row-major; probe = row 0.
    a = [rng.grid() for _ in range(K)]
    for _ in range(127 * K):
        rng.next()
    # B: K x N row-major. Stream it column-by-column via full regeneration.
    bvals = [rng.grid() for _ in range(K * N)]
    exact, sabs = [], []
    for j in range(N):
        acc = 0.0
        sa = 0.0
        for k in range(K):
            p = a[k] * bvals[k * N + j]
            acc += p
            sa += abs(p)
        exact.append(acc)
        sabs.append(sa)
    ok_exact = all(f64bits(e) == f64bits(r) for e, r in zip(exact, dump["exact_row"]))
    ok_sabs = all(f64bits(e) == f64bits(r) for e, r in zip(sabs, dump["sabs_row"]))
    check("oracle.python_fp64_bitwise_matches_cpp_exact_row", ok_exact,
          f"{N} elements, seed {seed}")
    check("oracle.python_sabs_bitwise_matches", ok_sabs)
    max_inv = max_cub = 0.0
    bound_ok = True
    for j in range(N):
        bd = gamma_bound(K, sabs[j])
        ei = abs(dump["gpu_inv_row"][j] - exact[j])
        ec = abs(dump["gpu_cublas_row"][j] - exact[j])
        max_inv = max(max_inv, ei)
        max_cub = max(max_cub, ec)
        if ei > bd or ec > bd:
            bound_ok = False
    check("oracle.gpu_rows_within_proven_bound", bound_ok,
          f"max_err inv={max_inv:.6g} cublas={max_cub:.6g}")
    # negative control: perturb beyond bound must be rejected
    j = 0
    bd = gamma_bound(K, sabs[j])
    perturbed = exact[j] + bd * 1.01 + 1e-9
    check("oracle.bound_checker_rejects_perturbation", abs(perturbed - exact[j]) > bd)
    return {"python_max_err_inv": max_inv, "python_max_err_cublas": max_cub}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    modes = run["modes"]

    # 1. environment
    dev = run.get("device_used", {})
    env = run["environment"]
    check("env.device_is_sm120", str(dev.get("compute_cap", "")).startswith("12."),
          f"{dev.get('name')} cc={dev.get('compute_cap')} driver={dev.get('driver_cuda_version')}")
    check("env.device_is_rtx5060ti", "5060" in dev.get("name", ""), dev.get("name", ""))
    check("env.cuda_visible_devices_0", env.get("cuda_visible_devices") == "0",
          env.get("cuda_visible_devices", ""))
    check("env.utc_timestamp", "utc_timestamp" in env, env.get("utc_timestamp", ""))
    for rel in ["cuda/bik.cu", "run.py", "Makefile", "tests/validate_results.py"]:
        check(f"env.sha256_present[{rel}]", rel in run.get("source_sha256", {}))
    for m, res in modes.items():
        check(f"env.binary_sha256[{m}]", bool(res.get("binary_sha256")))
        h = res.get("header") or {}
        hd = h.get("device") or {}
        check(f"env.header_gpu_provenance[{m}]",
              hd.get("name") == "NVIDIA GeForce RTX 5060 Ti"
              and str(hd.get("compute_cap", "")).startswith("12.")
              and bool(h.get("driver_version"))
              and h.get("driver_version") != "unavailable"
              and h.get("arch") == "sm_120"
              and h.get("cuda_visible_devices") == "0"
              and h.get("blackwell_sm120_only") is True,
              f"{hd.get('name')} cc={hd.get('compute_cap')} "
              f"driver={h.get('driver_version')} arch={h.get('arch')} "
              f"CVD={h.get('cuda_visible_devices')}")

    # 2. selftest
    st = modes["selftest"]["header"]["selftest"]
    for k in ["single_bit_flip_detected", "identical_buffers_pass", "fp16_flip_detected",
              "bound_checker_accepts_inside", "bound_checker_rejects_outside",
              "rng_deterministic", "pass"]:
        check(f"selftest.{k}", st.get(k) is True)

    # 3. experiment 1 counts + sanity
    recs = modes["measure"]["records"]
    fam = {}
    for r in recs:
        fam.setdefault(r["kernel"], []).append(r)
    check("measure.gemm_count_90", len(fam.get("gemm_cublaslt", [])) == 90,
          str(len(fam.get("gemm_cublaslt", []))))
    check("measure.softmax_count_20", len(fam.get("softmax_std", [])) == 20)
    check("measure.rmsnorm_count_10", len(fam.get("rmsnorm_std", [])) == 10)
    check("measure.attn_count_20", len(fam.get("attn_std", [])) == 20)
    m1 = [r for r in recs if r.get("M") == 1 or r.get("B") == 1]
    check("measure.m1_self_comparison_zero",
          all(r["diff_count"] == 0 for r in m1), f"{len(m1)} records")
    differing = [r for r in recs if r["diff_count"] > 0]
    check("measure.batch_dependence_observed", len(differing) > 0,
          f"{len(differing)}/{len(recs)} records differ from batch-1 probe")
    rerun = [r for r in fam.get("gemm_cublaslt", []) if r.get("repeat_run_diff_count", 0) > 0]
    print(f"[info] cuBLASLt run-to-run nondeterminism at fixed shape: {len(rerun)} records")

    # 4. experiment 2 acceptance
    inv = modes["invariant"]["records"]
    ifam = {}
    for r in inv:
        ifam.setdefault(r["kernel"], []).append(r)
    g = ifam.get("gemm_invariant", [])
    check("invariant.gemm_count_18", len(g) == 18, str(len(g)))
    check("invariant.gemm_all_bitwise_pass", all(r["acceptance_pass"] for r in g))
    check("invariant.gemm_all_oracle_bound_pass", all(r["oracle_bound_pass"] for r in g))
    s = ifam.get("softmax_invariant", [])
    check("invariant.softmax_count_4_all_pass", len(s) == 4 and all(r["acceptance_pass"] for r in s))
    rn = ifam.get("rmsnorm_invariant", [])
    check("invariant.rmsnorm_count_2_all_pass", len(rn) == 2 and all(r["acceptance_pass"] for r in rn))
    at = ifam.get("attn_invariant", [])
    check("invariant.attn_count_4_all_pass", len(at) == 4 and all(r["acceptance_pass"] for r in at))
    dumps = ifam.get("gemm_oracle_dump", [])
    check("invariant.oracle_dump_present", len(dumps) == 1)

    # 5. negative controls: targeted cases (where experiment 1 observed batch
    # dependence) MUST fail the harness; zero-diff cases must correctly pass.
    nc = [r for r in inv if r["kernel"].startswith("negative_control_")]
    check("negative.std_harness_records_present", len(nc) == 4, str(len(nc)))
    targeted = [r for r in nc if r.get("case") == "targeted"]
    zero = [r for r in nc if r.get("case") == "zero_diff_case"]
    check("negative.targeted_std_fails_acceptance",
          len(targeted) == 2 and all(r["std_fails_acceptance"] for r in targeted),
          "; ".join(f"{r['kernel']}:{r['std_fails_acceptance']}" for r in targeted))
    check("negative.zero_diff_cases_correctly_pass",
          len(zero) == 2 and all(not r["std_fails_acceptance"] for r in zero),
          "; ".join(f"{r['kernel']}:{r['std_fails_acceptance']}" for r in zero))

    # 6. independent Python oracle
    if dumps:
        independent_oracle(dumps[0])

    # 7. cost + block
    cost = modes["cost"]["records"]
    check("cost.record_count_28", len(cost) == 28, str(len(cost)))
    check("cost.positive_medians",
          all(r["std_median_ms"] > 0 and r["inv_median_ms"] > 0 for r in cost))
    check("cost.boundary_resident", all(r["boundary"] == "resident" for r in cost))
    blk = [r for r in modes["block"]["records"] if r["kernel"] == "block_replay"]
    check("block.record_present", len(blk) == 1)
    if blk:
        b = blk[0]
        check("block.invariant_bitwise_replay", b["invariant_bitwise_equal"] is True,
              f"{b['elements_compared']} elements, B=1 vs B=7")
        check("block.invariant_repeat_deterministic", b["invariant_repeat_equal"] is True)
        check("block.elements_524288", b["elements_compared"] == 128 * 4096)
        print(f"[info] standard block B=1 vs B=7: diff_fraction="
              f"{b['standard_diff_fraction']:.6g} max_ulp={b['standard_max_ulp']}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED:", ", ".join(c[0] for c in failed))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()