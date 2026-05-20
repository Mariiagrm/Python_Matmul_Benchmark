import torch
import pandas as pd
import time
from tqdm import tqdm
import torch._inductor.config
import triton
import triton.language as tl
import os
from pathlib import Path

# Obtener la ruta absoluta del directorio donde está este script
script_dir = Path(__file__).parent.resolve()

# Cambiar el directorio de trabajo actual a la carpeta del script
os.chdir(script_dir)

# --- 1. CONFIGURACIÓN EXTREMA DEL ENTORNO Y HARDWARE ---
device = torch.device("cuda")

# Optimizaciones de backend para RTX 4090 (Ada Lovelace)
#torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_fp16_accumulation = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False # Forzamos FP16 puro según tus pruebas anteriores
torch.backends.cudnn.allow_tf32 = False

# Al forzar a Inductor a usar Triton y buscar exhaustivamente el mejor kernel da un rendimineto pesimo
#torch._inductor.config.max_autotune_gemm_backends = "TRITON"
#PyTorch probará ATen/cuBLAS vs Triton y elegirá el más rápido
torch._inductor.config.max_autotune = True


# Opcional: Descomenta esto solo si quieres ver el código generado. 
# Si buscas velocidad pura, imprimir en consola ralentiza Python.
# torch._logging.set_logs(output_code=True)

# 2. Definir la función base
def matmul_fn(a, b):
    return torch.matmul(a, b)

# --- 3. COMPILACIÓN: CERO ASESINOS DE RENDIMIENTO ---
# mode="max-autotune": Pruebas exhaustivas de kernels y tiles.
# fullgraph=True: Falla inmediatamente si hay una ruptura de grafo.
# dynamic=False: Asume tamaños estáticos. Evita recompilaciones sorpresa.
fast_matmul = torch.compile(
    matmul_fn, 
    mode="max-autotune", 
    fullgraph=True, 
    dynamic=False
)

#-------------Kernel de Triton------------------------------


# --- KERNEL TRITON SIMPLE ---
@triton.jit
def matmul_kernel_simple(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for _ in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

# --- WRAPPER DE PYTHON ---
def triton_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel_simple[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )
    return c

def run_benchmarks(custom_tasks=None):
    results = []

    if custom_tasks is not None:
        all_tasks = custom_tasks
    else:
        dims_base = [1024, 2046, 4096, 8192, 16384, 32768]
        bench_1_combs = [("Square", d, d, d) for d in dims_base]
        K_fixed = 8192
        bench_2_combs = [("Fixed_K", i, i, K_fixed) for i in dims_base]
        all_tasks = bench_1_combs + bench_2_combs
    
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 Modo: torch.compile(max-autotune, fullgraph=True, dynamic=False)")
    print(f"📊 Ejecutando {len(all_tasks)} pruebas específicas...")

    for label, M, N, K in tqdm(all_tasks, desc="Benchmarking"):
        dtype = torch.float16
        try:
            torch.cuda.empty_cache()

            # Usamos zeros para aislar el rendimiento puro sin overhead de generación aleatoria
            a = torch.zeros((M, K), device=device, dtype=dtype)
            b = torch.zeros((K, N), device=device, dtype=dtype)#.t()

            # Selecciona aqui que kernel medir.  El warmup y la medicion deben
            # usar EXACTAMENTE la misma funcion para no contaminar la medida con
            # compilacion JIT / autotune de Triton.
            bench_fn = fast_matmul
            #bench_fn = triton_matmul

            # --- 4. CALENTAMIENTO Y AUTO-TUNING ESTRICTO ---
            # Al cambiar de tamaño, Inductor detectará el cambio y recompilará.
            # Este bucle absorbe todo el tiempo de compilación y búsqueda de Triton.
            for _ in range(10):
                bench_fn(a, b)

            torch.cuda.synchronize()

            # --- 5. MEDICIÓN (Aislada del compilador) ---
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            iters = 100 #if (M*N*K) < (4096**3) else 20

            start.record()
            for _ in range(iters):
                bench_fn(a, b)

            end.record()

            torch.cuda.synchronize()
            
            avg_time_ms = start.elapsed_time(end) / iters
            avg_time_sec = avg_time_ms / 1000.0
            
            flops = 2.0 * M * N * K
            tflops = flops / (avg_time_sec * 1e12)

            results.append({
                "Type": label,
                "M": M, "N": N, "K": K,
                "Time_ms": avg_time_ms,
                "TFLOPS": tflops
            })

            del a, b

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ OOM en {M}x{N}x{K}. Saltando.")
            else:
                print(f"\n❌ Error Runtime en {M}x{N}x{K}: {e}")
        except Exception as e:
            # Captura posibles fallos de fullgraph=True
            print(f"\n❌ Error de compilación en {M}x{N}x{K}: {e}")

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
        filename = "../results/rtx4090_benchmark_jit.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Benchmark completado. Guardado en {filename}")
    else:
        print("\n💾 Resultados NO guardados (--mnk o --no-save).")

    if not df.empty:
        print("\n--- Resultados ---")
        print(df.to_markdown(index=False))