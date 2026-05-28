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
torch.backends.cuda.matmul.allow_fp16_accumulation = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
torch._inductor.config.max_autotune_gemm_backends = "TRITON"


torch._logging.set_logs(output_code=True)


# 1. Definir la función base
def matmul_fn(a, b):
    return torch.matmul(a, b)


def run_benchmarks(custom_tasks=None):
    results = []

    if custom_tasks is not None:
        all_tasks = custom_tasks
    else:
        dims_base = [1024, 2048, 4096, 8192, 16384, 32768]
        bench_1_combs = [("Square", d, d, d) for d in dims_base]
        K_fixed = 8192
        bench_2_combs = [("Fixed_K", i, i, K_fixed) for i in dims_base]
        all_tasks = bench_1_combs + bench_2_combs
    
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Ejecutando {len(all_tasks)} pruebas específicas...")

    # 1. Compilamos la función ANTES del bucle para que use Triton
    compiled_matmul_fn = torch.compile(matmul_fn)

    for label, M, N, K in tqdm(all_tasks, desc="Benchmarking"):
        # 2. CAMBIAR A FLOAT16
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
                compiled_matmul_fn(a, b)
            torch.cuda.synchronize()

            # --- MEDICIÓN ---
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Ajustamos iteraciones: menos iteraciones para matrices gigantes
            iters=100
            #iters = 100 if (M*N*K) < (4096**3) else 20
            
            start.record()
            for _ in range(iters):
                compiled_matmul_fn(a, b) # Usamos la función compilada
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

def _parse_mnk(s):
    parts = [p for p in s.replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        d = int(parts[0]); return d, d, d
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise SystemExit("Usa --mnk N o --mnk M,N,K")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=str, default=None)
    parser.add_argument("--sizes", type=str, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    custom_tasks = None
    save = not args.no_save
    if args.mnk:
        M, N, K = _parse_mnk(args.mnk)
        custom_tasks = [("Custom", M, N, K)]
        save = False
    elif args.sizes:
        dims = [int(x) for x in args.sizes.split(",")]
        custom_tasks = ([("Square", d, d, d) for d in dims]
                        + [("Fixed_K", d, d, 8192) for d in dims])

    df = run_benchmarks(custom_tasks=custom_tasks)

    if save:
        filename = "../results/rtx4090_pytorch_eager.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Benchmark completado. Guardado en {filename}")
    else:
        print("\n💾 Resultados NO guardados (--mnk o --no-save).")

    if not df.empty:
        print("\n--- Resultados ---")
        print(df.to_markdown(index=False))