import torch
import pandas as pd
import itertools
from tqdm import tqdm

# Optimización para RTX 4090 (Arquitectura Ada Lovelace)
torch.backends.cuda.matmul.allow_tf32 = True 
device = torch.device("cuda")

def run_exhaustive_benchmark():
    # Dimensiones solicitadas
    dims = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    precisions = [torch.float16]
    
    # Generar todas las combinaciones posibles de M, N, K (1000 combinaciones)
    combinations = list(itertools.product(dims, dims, dims))
    results = []

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Ejecutando {len(combinations)} combinaciones por precisión...")

    for dtype in precisions:
        dtype_str = "fp16"
        
        for M, N, K in tqdm(combinations, desc=f"Testing {dtype_str}"):
            try:
                # Liberar caché para evitar fragmentación
                torch.cuda.empty_cache()

                # Inicializar matrices A (MxK) y B (KxN)
                a = torch.randn((M, K), device=device, dtype=dtype)
                b = torch.randn((K, N), device=device, dtype=dtype)

                # Warm-up rápido
                for _ in range(3):
                    torch.matmul(a, b)
                
                torch.cuda.synchronize()

                # Medición con CUDA Events
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                # Ajustamos iteraciones según carga para no eternizar el script
                iters = 5 if (M*N*K) > (4096**3) else 20
                for _ in range(iters):
                    torch.matmul(a, b)
                end.record()

                torch.cuda.synchronize()
                
                # Tiempo promedio en segundos
                avg_time = (start.elapsed_time(end) / 1000) / iters
                
                # Operaciones Flotantes: 2 * M * N * K
                flops = 2.0 * M * N * K
                tflops = flops / (avg_time * 1e12)

                results.append({
                    "Precision": dtype_str,
                    "M": M, "N": N, "K": K,
                    "Time_ms": avg_time * 1000,
                    "TFLOPS": tflops
                })

                del a, b # Limpieza explícita

            except RuntimeError as e:
                # En caso de Out of Memory (OOM) saltamos la combinación
                print(f"\n⚠️ Saltando {M}x{N}x{K} por falta de memoria.")
                continue

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_exhaustive_benchmark()
    
    # Calcular Speedup relativo (FP16 vs FP32) para las mismas dimensiones
    # Esto requiere pivotar la tabla o un join posterior si lo necesitas para análisis.
    
    df.to_csv("rtx4090_benchmark_fp16.csv", index=False)
    print("\n✅ Benchmark completado. Datos guardados en 'rtx4090_benchmark_fp16.csv'")
    
    # Mostrar top 5 resultados en TFLOPS
    print("\n🔥 Top 5 Rendimiento (TFLOPS):")
    print(df.sort_values(by="TFLOPS", ascending=False).head(5))