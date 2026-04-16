import pandas as pd
import os
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)

csv_files = [
        "../results/benchmark_mma-matmul.csv",
    "../results/rtx4090_benchmark_jit.csv",
    "../results/rtx4090_pytorch_eager.csv",
    "../results/rtx4090_torch_aoti_benchmark.csv",
    "../results/rtx4090_benchmark_cuda.csv" 
]


# Dictionary for readable names
kernels_names = {
    "../results/benchmark_mma-matmul.csv": "MMA MatMul with kernel 3.2",
    "../results/rtx4090_benchmark_jit.csv": "Torch JIT",
    "../results/rtx4090_pytorch_eager.csv": "PyTorch Eager",
    "../results/rtx4090_torch_aoti_benchmark.csv": "Torch AOTI",
    "../results/rtx4090_benchmark_cuda.csv": "Native CUDA"
}


pp_base_fp15_acfp16 = 330.3

# Load all CSVs and tag each row with its kernel name
dataframes = []
for file in csv_files:
    if not os.path.exists(file):
        print(f"⚠️ Warning: File '{file}' not found.")
        continue
    df = pd.read_csv(file)
    df = df.dropna(subset=["Time_ms", "TFLOPS"])
    df["Kernel"] = kernels_names[file]
    dataframes.append(df)

df_all = pd.concat(dataframes, ignore_index=True)

# Group by matrix size (Type, M, N, K) so we compare kernels on the same workload
configs = df_all.groupby(["Type", "M", "N", "K"])

rows = []
for (typ, M, N, K), group in configs:
    for _, row in group.iterrows():
        rows.append({
            "Type": typ,
            "M": M, "N": N, "K": K,
            "Kernel": row["Kernel"],
            "Execution Time (ms)": round(row["Time_ms"], 4),
            "TFLOP/s": round(row["TFLOPS"], 2),
            "% 4090 peak FP16/16": round(row["TFLOPS"] / pp_base_fp15_acfp16 * 100, 2),
        })

result = (
    pd.DataFrame(rows)
    .sort_values(by=["Type", "M", "N", "K", "TFLOP/s"], ascending=[True, True, True, True, False])
    .reset_index(drop=True)
)

# Print one table per matrix configuration
for (typ, M, N, K), group in result.groupby(["Type", "M", "N", "K"], sort=False):
    print(f"\n{'='*60}")
    print(f"  {typ} — {M}x{N}x{K}")
    print(f"{'='*60}")
    print(group[["Kernel", "Execution Time (ms)", "TFLOP/s", "% 4090 peak FP16/16"]].to_markdown(index=False))

output_path = "../results/peak_performance_summary.csv"
result.to_csv(output_path, index=False)
print(f"\n✅ Saved to {output_path}")

