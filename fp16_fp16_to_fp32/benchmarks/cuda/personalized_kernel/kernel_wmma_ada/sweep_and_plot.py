"""
Sweep hgemm_wmma_ada across square sizes and plot TFLOP/s vs N for:
  - kernel WMMA propio
  - cuBLAS (Tensor Cores)

Usage:
    python3 sweep_and_plot.py
Outputs:
    sweep_results.csv
    sweep_tflops.png
"""

import csv
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt

HERE        = os.path.dirname(os.path.abspath(__file__))
BINARY      = os.path.join(HERE, "hgemm_wmma_ada")
CSV_PATH    = os.path.join(HERE, "sweep_results.csv")
PLOT_PATH   = os.path.join(HERE, "sweep_tflops.png")
PEAK_TFLOPS = 165.2  # RTX 4090 FP16 + FP32-acc Tensor Core peak

SIZES = [1024, 2048, 4096, 8192, 16384]

# Output lines from hgemm_wmma_ada:
#   "Kernel WMMA propio: 0.123 ms   140.50 TFLOP/s"
#   "cuBLAS (TC)       : 0.110 ms   158.30 TFLOP/s"
LINE_RE = re.compile(
    r"^(Kernel WMMA propio|cuBLAS \(TC\))\s*:\s*([\d.]+)\s*ms\s+([\d.]+)\s*TFLOP/s"
)

def run_one(n):
    out = subprocess.check_output(
        [BINARY, str(n), str(n), str(n)], text=True
    )
    parsed = {}
    for line in out.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            label, ms, tflops = m.group(1), float(m.group(2)), float(m.group(3))
            key = "kernel" if label == "Kernel WMMA propio" else "cublas"
            parsed[key] = {"ms": ms, "tflops": tflops}
    if "kernel" not in parsed or "cublas" not in parsed:
        print(out)
        raise RuntimeError(f"could not parse output for N={n}")
    return parsed

def main():
    if not os.path.isfile(BINARY) or not os.access(BINARY, os.X_OK):
        sys.exit(f"binary not found or not executable: {BINARY}")

    rows = []
    for n in SIZES:
        print(f"running N={n} ...", flush=True)
        r = run_one(n)
        rows.append({
            "N": n,
            "kernel_ms": r["kernel"]["ms"],
            "kernel_tflops": r["kernel"]["tflops"],
            "cublas_ms": r["cublas"]["ms"],
            "cublas_tflops": r["cublas"]["tflops"],
            "ratio_pct": 100.0 * r["kernel"]["tflops"] / r["cublas"]["tflops"],
            "kernel_pct_peak": 100.0 * r["kernel"]["tflops"] / PEAK_TFLOPS,
            "cublas_pct_peak": 100.0 * r["cublas"]["tflops"] / PEAK_TFLOPS,
        })

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV_PATH}")

    ns      = [r["N"] for r in rows]
    kernel  = [r["kernel_tflops"] for r in rows]
    cublas  = [r["cublas_tflops"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, kernel, marker="o", linewidth=2, label="Kernel WMMA propio")
    ax.plot(ns, cublas, marker="s", linewidth=2, label="cuBLAS (TC)")
    ax.axhline(PEAK_TFLOPS, linestyle="--", color="grey", linewidth=1,
               label=f"Peak FP16+FP32-acc TC RTX 4090 ({PEAK_TFLOPS} TFLOP/s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N (matriz cuadrada NxNxN)")
    ax.set_ylabel("TFLOP/s")
    ax.set_title("HGEMM FP16 — Kernel WMMA vs cuBLAS vs Peak (RTX 4090)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    for n, k, c in zip(ns, kernel, cublas):
        ax.annotate(f"{100*k/c:.0f}%", (n, k),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="tab:blue")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"wrote {PLOT_PATH}")

if __name__ == "__main__":
    main()
