import torch
import pandas as pd
import itertools
import time
from tqdm import tqdm

# --- Configuración para RTX 4090 ---
torch.set_float32_matmul_precision('high') # Habilita TF32 (TensorFloat-32) internamente
device = torch.device("cuda")

# 1. Definir la función base
def matmul_fn(a, b):
    return torch.matmul(a, b)

# 2. Compilar con torch.compile
# mode="max-autotune": El mejor para 4090. Perfila configs de Triton y elige la más rápida.
# fullgraph=False: Permitimos que PyTorch maneje grafos parciales si fuera necesario.
fast_matmul = torch.compile(matmul_fn, mode="max-autotune")

def run_exhaustive_benchmark():
    # Nota: max-autotune tarda un poco en compilar cada nueva forma.
    dims = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    
    precisions = [torch.float16]
    
    # Generar combinaciones
    combinations = list(itertools.product(dims, dims, dims))
    results = []

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 Modo: torch.compile(mode='max-autotune')")
    print(f"📊 Ejecutando {len(combinations)} combinaciones...")

    for dtype in precisions:
        dtype_str = "fp16"
        
        for M, N, K in tqdm(combinations, desc=f"Benchmarking {dtype_str}"):
            try:
                # Limpieza de memoria
                torch.cuda.empty_cache()

                # Crear tensores
                a = torch.randn((M, K), device=device, dtype=dtype)
                b = torch.randn((K, N), device=device, dtype=dtype)

                # --- FASE DE WARM-UP Y COMPILACIÓN JIT ---
                # La primera ejecución aquí disparará la compilación de Triton.
                # Es CRUCIAL que esto ocurra antes de start.record()
                for _ in range(3):
                    fast_matmul(a, b)
                
                torch.cuda.synchronize()

                # --- FASE DE MEDICIÓN ---
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                # Ajuste dinámico de iteraciones para no eternizar tamaños grandes
                #total_ops = M * N * K
                #if total_ops > (8192**3): iters = 5
                #elif total_ops > (4096**3): iters = 10
                #else: iters = 50

                #for _ in range(iters):
                start.record()
                # Ajustamos iteraciones según carga para no eternizar el script
                iters = 5 if (M*N*K) > (4096**3) else 20
                for _ in range(iters):
                    torch.matmul(a, b)
                end.record()

                torch.cuda.synchronize()
                
                # Cálculos
                avg_time_ms = start.elapsed_time(end) / iters
                avg_time_sec = avg_time_ms / 1000.0
                
                # TFLOPS = 2 * M * N * K / Tiempo / 10^12
                flops = 2.0 * M * N * K
                tflops = flops / (avg_time_sec * 1e12)

                results.append({
                    "Precision": dtype_str,
                    "M": M, "N": N, "K": K,
                    "Time_ms": avg_time_ms,
                    "TFLOPS": tflops
                })

                del a, b

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM en {M}x{N}x{K}. Saltando.")
                    torch.cuda.empty_cache()
                else:
                    print(f"\n❌ Error en {M}x{N}x{K}: {e}")
                continue
            except Exception as e:
                # Captura errores de compilación de Triton si ocurren
                print(f"\n❌ Error de compilación/ejecución: {e}")
                continue

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Verificar disponibilidad de triton
    try:
        import triton
        print(f"✅ Triton versión {triton.__version__} detectada.")
    except ImportError:
        print("⚠️ Advertencia: Triton no instalado. 'max-autotune' podría no funcionar bien.")

    df = run_exhaustive_benchmark()
    
    filename = "rtx4090_torch_compile_benchmark.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Benchmark completado. Guardado en {filename}")
    
    print("\n🔥 Top 5 Rendimiento (TFLOPS) con torch.compile:")
    # Ordenar por TFLOPS descendente
    print(df.sort_values(by="TFLOPS", ascending=False).head(5).to_markdown())