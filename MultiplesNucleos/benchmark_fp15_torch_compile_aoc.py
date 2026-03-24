import torch
import pandas as pd
import itertools
import time
from tqdm import tqdm

# Configuración RTX 4090
torch.set_float32_matmul_precision('high')
device = torch.device("cuda")

# Definimos la función base
def matmul_fn(a, b):
    return torch.matmul(a, b)

def run_aot_benchmark():
    # Reducimos las dimensiones para el ejemplo porque AOT tarda en compilar CADA una
    dims = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    precisions = [torch.float16]
    
    # Generar combinaciones
    combinations = list(itertools.product(dims, dims, dims))
    
    # Almacén de modelos compilados (Simula los binarios .so)
    # Clave: (M, N, K) -> Valor: Función Compilada Optimizada
    aot_cache = {}

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"⚙️  Fase 1: COMPILACIÓN AOT (Ahead-Of-Time)...")
    print("   Generando kernels estáticos específicos para cada tamaño.")
    print("   Esto tardará un poco, pero la ejecución será instantánea después.")

    # --- FASE 1: BUILD / COMPILACIÓN ---
    for M, N, K in tqdm(combinations, desc="Compilando Modelos"):
        try:
            # 1. Crear tensores dummy para definir la forma
            a = torch.randn((M, K), device=device, dtype=torch.float16)
            b = torch.randn((K, N), device=device, dtype=torch.float16)

            # 2. Compilar ESPECÍFICAMENTE para estas dimensiones
            # dynamic=False: Obliga a generar código estático ultra-rápido para este tamaño exacto
            # mode="max-autotune": Usa Triton para buscar el mejor kernel
            opt_model = torch.compile(matmul_fn, mode="max-autotune", dynamic=False)

            # 3. Forzar la compilación real ejecutando una vez (Warmup/Trace)
            opt_model(a, b)
            
            # 4. Guardar el modelo "listo para usar" en nuestro caché
            aot_cache[(M, N, K)] = opt_model
            
            del a, b, opt_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Falló compilación para {M}x{N}x{K}: {e}")

    print(f"\n🚀 Fase 2: EJECUCIÓN (Benchmark puro sin overhead)...")
    
    results = []
    
    # --- FASE 2: RUNTIME / BENCHMARK ---
    for M, N, K in tqdm(combinations, desc="Benchmarking"):
        if (M, N, K) not in aot_cache:
            continue

        try:
            # Recuperar el modelo ya compilado
            model = aot_cache[(M, N, K)]
            
            # Datos frescos
            a = torch.randn((M, K), device=device, dtype=torch.float16)
            b = torch.randn((K, N), device=device, dtype=torch.float16)
            
            torch.cuda.synchronize()
            
            # Medición Pura
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            #iters = 50 # Podemos hacer muchas iteraciones porque es muy rápido
            iters = 50 if (M*N*K) > (4096**3) else 80
            start.record()
            for _ in range(iters):
                model(a, b) # Llamada al kernel estático
            end.record()
            
            torch.cuda.synchronize()
            
            avg_time_ms = start.elapsed_time(end) / iters
            avg_time_sec = avg_time_ms / 1000.0
            flops = 2.0 * M * N * K
            tflops = flops / (avg_time_sec * 1e12)
            
            results.append({
                "Mode": "AOT_Static",
                "M": M, "N": N, "K": K,
                "Time_ms": avg_time_ms,
                "TFLOPS": tflops
            })
            
            del a, b

        except Exception as e:
            print(f"Error en ejecución: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_aot_benchmark()
    print("\n🔥 Resultados AOT (Static Shapes):")
    print(df.sort_values(by="TFLOPS", ascending=False).head().to_markdown())
    df.to_csv("rtx4090_aot_benchmark.csv", index=False)