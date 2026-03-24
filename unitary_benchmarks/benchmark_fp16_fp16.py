import torch
import pandas as pd
import itertools
import time
from tqdm import tqdm
import os
from pathlib import Path

# Obtener la ruta absoluta del directorio donde está este script
script_dir = Path(__file__).parent.resolve()

# Cambiar el directorio de trabajo actual a la carpeta del script
os.chdir(script_dir)

# --- Configuración para RTX 4090 ---
torch.backends.cuda.matmul.allow_fp16_accumulation = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
torch._inductor.config.max_autotune_gemm_backends = "TRITON"


torch._logging.set_logs(output_code=True)


# 1. Definir la función base
def matmul_fn(a, b):
    return torch.matmul(a, b)


def run_specific_benchmarks():
    results = []
    
    # --- DEFINICIÓN DE CASOS ---
    
    # Lista de dimensiones base
    # Nota: 2046 no es potencia de 2. Para máximo rendimiento se suele usar 2048.
    # Pero he dejado 2046 tal como pediste.
    dims_base = [ 32768] 
    
    # Benchmark 1: Matrices Cuadradas (M = N = K)
    # Genera: (1024,1024,1024), (2046,2046,2046)...
    bench_1_combs = [("Square", d, d, d) for d in dims_base]
    
    # Benchmark 2: M y N varían, K fijo en 32768
    # Genera: (1024, 1024, 32768), (1024, 2046, 32768)...
    K_fixed = 32768
    bench_2_combs = []
    # Producto cartesiano solo de M y N
    #mn_combs = list(itertools.product(dims_base, dims_base)) 
    for i in dims_base:
        bench_2_combs.append(("Fixed_K", i, i, K_fixed))

    # Unimos ambas listas de tareas
    all_tasks = bench_1_combs + bench_2_combs
    
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Ejecutando {len(all_tasks)} pruebas específicas...")

    # Iteramos sobre la lista unificada
    for label, M, N, K in tqdm(all_tasks, desc="Benchmarking"):
        dtype = torch.float16
        try:
            # Limpieza de memoria para evitar fragmentación
            torch.cuda.empty_cache()

            # Crear tensores
            a = torch.zeros((M, K), device=device, dtype=dtype)
            b = torch.zeros((K, N), device=device, dtype=dtype)

            # --- WARM-UP ---
            # Ejecutamos varias veces para que Triton compile el kernel
            for _ in range(10):
                matmul_fn(a, b)
            torch.cuda.synchronize()

            # --- MEDICIÓN ---
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Ajustamos iteraciones: menos iteraciones para matrices gigantes
            iters=100
            #iters = 100 if (M*N*K) < (4096**3) else 20
            
            start.record()
            for _ in range(iters):
                matmul_fn(a, b) # Usamos la función compilada
            end.record()

            torch.cuda.synchronize()
            
            # Cálculos
            avg_time_ms = start.elapsed_time(end) / iters
            avg_time_sec = avg_time_ms / 1000.0
            
            # TFLOPS = 2 * M * N * K / Tiempo / 10^12
            flops = 2.0 * M * N * K
            tflops = flops / (avg_time_sec * 1e12)

            results.append({
                "Type": label, # 'Square' o 'Fixed_K'
                "M": M, "N": N, "K": K,
                "Time_ms": avg_time_ms,
                "TFLOPS": tflops
            })

            del a, b

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ OOM en {M}x{N}x{K}. Saltando.")
            else:
                print(f"\n❌ Error en {M}x{N}x{K}: {e}")
        except Exception as e:
            print(f"\n❌ Error general: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_specific_benchmarks()
    
    filename = "../unique_results/rtx4090_pytorch_eager_specific.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Benchmark completado. Guardado en {filename}")
    
    # Mostrar tabla bonita en consola
    print("\n--- Resultados: Matrices Cuadradas ---")
    print(df[df["Type"] == "Square"].to_markdown(index=False))
    
    print("\n--- Resultados: K Fijo (32768) ---")
    # Mostramos los mejores 5 del segundo benchmark
    print(df[df["Type"] == "Fixed_K"].sort_values("TFLOPS", ascending=False).head(5).to_markdown(index=False))