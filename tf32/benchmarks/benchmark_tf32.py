import torch
import pandas as pd
import itertools
import time
from tqdm import tqdm
import os
from pathlib import Path

print(f"Versión: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo: {torch.cuda.get_device_name(0)}")

# Requiere PyTorch 2.7.0 o superior
_major, _minor = (int(x) for x in torch.__version__.split(".")[:2])
if (_major, _minor) < (2, 7):
    raise RuntimeError(
        f"Se requiere PyTorch >= 2.7.0, pero se encontró {torch.__version__}"
    )

# Obtener la ruta absoluta del directorio donde está este script
script_dir = Path(__file__).parent.resolve()

# Cambiar el directorio de trabajo actual a la carpeta del script
os.chdir(script_dir)

# --- Configuración TF32 para RTX 4090 (Ada Lovelace) ---
# Entradas en FP32 pero los Tensor Cores hacen el matmul en TF32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # equivalente a TF32 en matmul fp32
torch.backends.cudnn.benchmark = True

device = torch.device("cuda")
torch._inductor.config.max_autotune_gemm_backends = "TRITON"

torch._logging.set_logs(output_code=True)


# 1. Definir la función base
def matmul_fn(a, b):
    return torch.matmul(a, b)


def _looks_like_oom(msg: str) -> bool:
    msg = msg.lower()
    return (
        "out of memory" in msg
        or "cudamalloc" in msg
        or "cuda error: out of memory" in msg
    )


def run_benchmarks(custom_tasks=None):
    results = []

    if custom_tasks is not None:
        all_tasks = custom_tasks
    else:
        # --- DEFINICIÓN DE CASOS ---
        # Nota: en FP32 cada matriz pesa 4x más que en FP16. 32768x32768 = 4 GiB
        # por tensor; en una RTX 4090 (24 GB) entra pero deja poco margen.
        dims_base = [1024, 2048, 4096, 8192, 16384, 32768]
        bench_1_combs = [("Square", d, d, d) for d in dims_base]
        K_fixed = 8192
        bench_2_combs = [("Fixed_K", i, i, K_fixed) for i in dims_base]
        all_tasks = bench_1_combs + bench_2_combs

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Ejecutando {len(all_tasks)} pruebas específicas (TF32)...")

    # Iteramos sobre la lista unificada
    for label, M, N, K in tqdm(all_tasks, desc="Benchmarking TF32"):
        dtype = torch.float32  # TF32 requiere entradas FP32
        try:
            # Limpieza de memoria para evitar fragmentación
            torch.cuda.empty_cache()

            # Pre-check: A + B + C + ~25% workspace
            itemsize = torch.empty((), dtype=dtype).element_size()
            needed = int((M * K + K * N + M * N) * itemsize * 1.25)
            free, _ = torch.cuda.mem_get_info()
            if needed > free:
                print(f"\n⚠️ Saltando {M}x{N}x{K}: requiere ~{needed/1e9:.1f} GB, libres {free/1e9:.1f} GB.")
                continue

            # Crear tensores
            a = torch.zeros((M, K), device=device, dtype=dtype)
            b = torch.zeros((K, N), device=device, dtype=dtype)

            # --- WARM-UP ---
            for _ in range(10):
                matmul_fn(a, b)
            torch.cuda.synchronize()

            # --- MEDICIÓN ---
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            iters = 100

            start.record()
            for _ in range(iters):
                matmul_fn(a, b)
            end.record()

            torch.cuda.synchronize()

            # Cálculos
            avg_time_ms = start.elapsed_time(end) / iters
            avg_time_sec = avg_time_ms / 1000.0

            # TFLOPS = 2 * M * N * K / Tiempo / 10^12
            flops = 2.0 * M * N * K
            tflops = flops / (avg_time_sec * 1e12)

            results.append({
                "Type": label,
                "M": M, "N": N, "K": K,
                "Time_ms": avg_time_ms,
                "TFLOPS": tflops,
            })

            del a, b

        except RuntimeError as e:
            if _looks_like_oom(str(e)):
                print(f"\n⚠️ OOM en {M}x{N}x{K}. Saltando.")
                torch.cuda.empty_cache()
            else:
                print(f"\n❌ Error en {M}x{N}x{K}: {e}")
        except Exception as e:
            if _looks_like_oom(str(e)):
                print(f"\n⚠️ OOM en {M}x{N}x{K}. Saltando.")
                torch.cuda.empty_cache()
            else:
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
    parser.add_argument("--mnk", type=str, default=None,
                        help="Tamano custom: 'N' o 'M,N,K'. NO guarda CSV.")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Lista de tamanos para el sweep, ej '1024,2048,4096'.")
    parser.add_argument("--no-save", action="store_true",
                        help="Ejecuta el sweep pero no escribe el CSV.")
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
        results_dir = Path("../results")
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = results_dir / "rtx4090_pytorch_eager_tf32.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Benchmark completado. Guardado en {filename}")
    else:
        print("\n💾 Resultados NO guardados (--mnk o --no-save).")

    if not df.empty:
        print("\n--- Resultados ---")
        print(df.to_markdown(index=False))
