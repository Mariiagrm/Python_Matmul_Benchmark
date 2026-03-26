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


# --- KERNEL TRITON PERSONALIZADO ---
@triton.jit
def matmul_kernel_hilbert(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 1. IDENTIFICACIÓN DEL PROGRAMA (Scheduling)
    # Aquí es donde implementarías la lógica de la Curva de Hilbert.
    # Por simplicidad educativa, usamos un "Group Scheduling" que mejora L2
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Swizzling para maximizar L2 (Lógica de agrupación de bloques)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. OFFSETS Y PUNTEROS
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. BUCLE DE COMPUTACIÓN (Dot Product)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    #ERROR: AssertionError("Loop-carried variable accumulator has initial type <['128', '128'], fp16> but is re-assigned to <['128', '128'], fp32> in loop! Please make sure that the type stays consistent.")
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float16)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # 4. ESCRITURA DE RESULTADOS
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# --- WRAPPER DE PYTHON ---
def triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel_hilbert[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8, # Este parámetro controla el reuso de caché L2
        num_warps=8,
        num_stages=3
    )
    return c

def run_specific_benchmarks():
    results = []
    
    # --- DEFINICIÓN DE CASOS ---
    dims_base = [ 32768] 

    
    # Benchmark 1: Matrices Cuadradas
    bench_1_combs = [("Square", d, d, d) for d in dims_base]
    
    # Benchmark 2: K fijo (32768)
    K_fixed = 32768
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

            # --- 4. CALENTAMIENTO Y AUTO-TUNING ESTRICTO ---
            # Al cambiar de tamaño, Inductor detectará el cambio y recompilará.
            # Este bucle absorbe todo el tiempo de compilación y búsqueda de Triton.
            for _ in range(10):
                #fast_matmul(a, b)
                triton_matmul(a,b)
            
            # --- Para PERFILADO ---
            print("Capturando kernels...")
            torch.cuda.nvtx.range_push("profile_section")
            output = model_compiled(data)
            torch.cuda.synchronize() # Espera a que la GPU termine antes de cerrar el rango
            torch.cuda.nvtx.range_pop()

            # --- 5. MEDICIÓN (Aislada del compilador) ---
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            iters = 2 #if (M*N*K) < (4096**3) else 20
            
            start.record()
            for _ in range(iters):
                #fast_matmul(a, b)
                triton_matmul(a,b)

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

if __name__ == "__main__":
    df = run_specific_benchmarks()
    
    filename = "../unique_results/rtx4090_benchmark_jit_optimized.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Benchmark completado. Guardado en {filename}")
    
    print("\n--- Resultados: Matrices Cuadradas ---")
    print(df[df["Type"] == "Square"].to_markdown(index=False))
    
    print("\n--- Resultados: K Fijo (32768) ---")
    print(df[df["Type"] == "Fixed_K"].sort_values("TFLOPS", ascending=False).head(5).to_markdown(index=False))