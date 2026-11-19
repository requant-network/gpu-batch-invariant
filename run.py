#!/usr/bin/env python3
"""run.py — orchestrator for the gpu-batch-invariant study (stdlib only).

Runs on the GPU server (Ubuntu, python3.12). Assumes ./bik is already built:
    /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_120 cuda/bik.cu -o bik \
        -lcublasLt -lcublas

GPU etiquette (campaign CONTEXT.md): this script must be launched under the
cross-process GPU lock and pinned to GPU 0 (RTX 5060 Ti, SM120):
    flock /tmp/requant-new-studies-gpu.lock -c \
        'CUDA_VISIBLE_DEVICES=0 python3 run.py --all'

Each bik mode emits a JSONL stream (header object, record objects, footer
object). run.py parses it, attaches environment metadata + SHA-256 hashes of
the executed sources, and writes results/raw_<mode>.json plus a merged
results/run.json. No GPU work happens in this script itself.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

MODES = ["selftest", "measure", "invariant", "cost", "block"]
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              check=False).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def environment():
    env = {
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "nvcc": cmd_out(["/usr/local/cuda/bin/nvcc", "--version"]),
        "nvidia_smi": cmd_out(["nvidia-smi",
                               "--query-gpu=index,name,driver_version,compute_cap,memory.total",
                               "--format=csv"]),
    }
    # Device info actually used by the binary (from the selftest header) is
    # merged into each mode below; this is the driver-level view.
    return env


def source_hashes():
    files = ["cuda/bik.cu", "Makefile", "run.py", "tests/validate_results.py"]
    out = {}
    for rel in files:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            out[rel] = sha256_file(p)
    return out


def gpu0_driver(env):
    """Parse driver version of GPU 0 from the nvidia-smi CSV in env."""
    try:
        lines = env["nvidia_smi"].splitlines()
        hdr = [c.strip() for c in lines[0].split(",")]
        di = hdr.index("driver_version")
        ii = hdr.index("index")
        for ln in lines[1:]:
            cols = [c.strip() for c in ln.split(",")]
            if cols[ii] == "0":
                return cols[di]
    except Exception:  # noqa: BLE001
        pass
    return "unavailable"


def enrich_header(res, env):
    """Stamp every mode header with full GPU provenance (directive: GPU name +
    compute cap + driver in EVERY results JSON). arch is fixed by the build."""
    h = res.get("header") or {}
    dev = h.get("device") or {}
    h["driver_version"] = gpu0_driver(env)
    h["arch"] = "sm_120"
    h["cuda_visible_devices"] = env.get("cuda_visible_devices", "<unset>")
    h["gpu_index"] = 0
    h["blackwell_sm120_only"] = (
        str(dev.get("compute_cap", "")).startswith("12.")
        and "5060" in dev.get("name", "")
        and env.get("cuda_visible_devices") == "0")
    res["header"] = h
    return res


def run_mode(mode, outdir):
    binary = os.path.join(HERE, "bik")
    proc = subprocess.run([binary, mode], capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"bik {mode} exited {proc.returncode}")
    header, records, footer = None, [], None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        m = obj.get("mode", "")
        if m.endswith("_header") or mode == "selftest":
            header = obj
        elif m.endswith("_footer"):
            footer = obj
        else:
            records.append(obj)
    result = {
        "mode": mode,
        "header": header,
        "record_count": len(records),
        "records": records,
        "footer": footer,
        "binary_sha256": sha256_file(binary),
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }
    path = os.path.join(outdir, f"raw_{mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"[run] {mode}: {len(records)} records -> {path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--finalize", action="store_true",
                    help="no GPU work: re-read existing raw_*.json, stamp GPU "
                         "provenance into every header, rebuild run.json")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if args.finalize:
        env = environment()
        hashes = source_hashes()
        merged = {"study": "gpu-batch-invariant", "environment": env,
                  "source_sha256": hashes, "modes": {}}
        for m in MODES:
            p = os.path.join(args.outdir, f"raw_{m}.json")
            with open(p, encoding="utf-8") as f:
                res = json.load(f)
            enrich_header(res, env)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            merged["modes"][m] = res
            h = res.get("header") or {}
            if "device" in h:
                merged.setdefault("device_used", h["device"])
        path = os.path.join(args.outdir, "run.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1)
        print(f"[run] finalized (provenance stamped, no GPU work) -> {path}",
              flush=True)
        return
    modes = MODES if args.all else [args.mode]
    if not modes or modes == [None]:
        ap.error("specify --all or --mode")
    os.makedirs(args.outdir, exist_ok=True)
    env = environment()
    hashes = source_hashes()
    merged = {"study": "gpu-batch-invariant", "environment": env,
              "source_sha256": hashes, "modes": {}}
    for m in modes:
        merged["modes"][m] = enrich_header(run_mode(m, args.outdir), env)
    # device identity from the binary itself (authoritative for what ran)
    for m, res in merged["modes"].items():
        h = res.get("header") or {}
        if "device" in h:
            merged.setdefault("device_used", h["device"])
            break
    if args.all:
        path = os.path.join(args.outdir, "run.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1)
        print(f"[run] merged -> {path}", flush=True)
    else:
        # single-mode re-run: refresh that mode inside an existing run.json
        path = os.path.join(args.outdir, "run.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            old["modes"].update(merged["modes"])
            old["environment"] = env
            old["source_sha256"] = hashes
            old["last_partial_rerun_utc"] = env["utc_timestamp"]
            for m, res in merged["modes"].items():
                h = res.get("header") or {}
                if "device" in h:
                    old["device_used"] = h["device"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(old, f, indent=1)
            print(f"[run] updated {path} (mode(s): {', '.join(modes)})", flush=True)


if __name__ == "__main__":
    main()