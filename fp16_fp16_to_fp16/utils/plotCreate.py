import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Get the absolute path of the directory where this script is located
script_dir = Path(__file__).parent.resolve()

# Change the current working directory to the script's folder
os.chdir(script_dir)

# Initial configuration
csv_files = [
        "../results/benchmark_mma-matmul.csv",
    "../results/rtx4090_benchmark_jit.csv",
    "../results/rtx4090_pytorch_eager.csv",
    "../results/rtx4090_torch_aoti_benchmark.csv",
    "../results/rtx4090_benchmark_cuda.csv" 
]

tflops_col = "TFLOPS"

# Dictionary for readable names
method_names = {
    "../results/benchmark_mma-matmul.csv": "MMA MatMul with kernel 3.2",
    "../results/rtx4090_benchmark_jit.csv": "Torch JIT",
    "../results/rtx4090_pytorch_eager.csv": "PyTorch Eager",
    "../results/rtx4090_torch_aoti_benchmark.csv": "Torch AOTI",
    "../results/rtx4090_benchmark_cuda.csv": "Native CUDA"
}


dataframes = []

# Read and process each file
for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
                 # 1. ORDENAR numéricamente por dimensiones (esto es la clave)
        df = df.sort_values(by=['M', 'N', 'K'])
    
        # 2. Crear la etiqueta después de ordenar
        df['Config_M_N_K'] = df['M'].astype(str) + "x" + df['N'].astype(str) + "x" + df['K'].astype(str)
        
        
        # Extract necessary columns, INCLUDING 'Type' and numeric dims for sorting
        df_summary = df[['Type', 'Config_M_N_K', 'M', 'N', 'K', tflops_col]].copy()
        df_summary['Method'] = method_names[file]
        
        dataframes.append(df_summary)
    else:
        print(f"⚠️ Warning: File '{file}' not found.")

if not dataframes:
    print("❌ Could not load any files. Check the paths.")
else:
    # Join all dataframes
    df_total = pd.concat(dataframes, ignore_index=True)

    # Identify available modes in the 'Type' column
    modes = df_total['Type'].unique()
    print(f"Data types detected: {modes}\n")

    # Generate a plot for each Mode
    for mode in modes:
        print(f"=== Processing plot for mode: {mode} ===")

        # Filter only the data for this mode
        df_mode = df_total[df_total['Type'] == mode]

        # Create the pivot table for this specific mode
        df_pivot = df_mode.pivot_table(
            index='Config_M_N_K',
            columns='Method',
            values=tflops_col,
            aggfunc='mean'
        )

        # Sort configs numerically by (M, N, K) instead of lexicographically
        sorted_configs = (
            df_mode[['Config_M_N_K', 'M', 'N', 'K']]
            .drop_duplicates('Config_M_N_K')
            .sort_values(by=['M', 'N', 'K'])
            ['Config_M_N_K']
        )
        df_pivot = df_pivot.reindex(sorted_configs)

        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Generate the bars
        df_pivot.plot(kind='bar', figsize=(12, 6), width=0.8, colormap='viridis')
        
        plt.title(f'Throughput (TFLOPS) MatMul (FP16/FP16 --> FP16) - RTX 4090 - Mode: {mode}', fontsize=16, pad=15)
        plt.xlabel('Matrix Size (M x N x K)', fontsize=12)
        plt.ylabel('Throughput (TFLOPS)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Execution Mode', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot with a dynamic name based on the mode
        output_filename = f"../plots/compare_tflops_{mode.lower()}.png"
        plt.savefig(output_filename, dpi=300)
        print(f"✅ Plot saved as '{output_filename}'.\n")
        
        # Show the plot on screen
        plt.show()