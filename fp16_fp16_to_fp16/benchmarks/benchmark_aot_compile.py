import argparse
import tempfile
import torch
import pandas as pd
import time
import os
from tqdm import tqdm
from torch.export import export
import torch._inductor
from torch._inductor.package import load_package
import os
from pathlib import Path
#python benchmark_aot_compile.py --mnk 8192 --no-save
# Obtener la ruta absoluta del directorio donde está este script
script_dir = Path(__file__).parent.resolve()

# Cambiar el directorio de trabajo actual a la carpeta del script
os.chdir(script_dir)

# --- 1. CONFIGURACIÓN DE PRECISIÓN Y BACKEND ---
device = torch.device("cuda")
torch._logging.set_logs(output_code=True)

#Para forzar ALLOW_TF32=False en Triton, desactivamos TF32 a nivel global
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Forzamos acumulación estricta en FP16 (ACC_TYPE='tl.float16')
if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
    torch.backends.cuda.matmul.allow_fp16_accumulation = True

# Configuramos Inductor globalmente para el máximo autotuning durante la fase AOT
torch._inductor.config.max_autotune = True

# --- 2. PREPARACIÓN PARA AOTI ---
# AOTI / torch.export requiere que la función esté envuelta en un nn.Module

def load_aoti_model(package_path):
    # Intento 1: PyTorch 2.5 / 2.6+ (La más moderna para archivos .pt2)
    try:
        from torch._inductor import aoti_load_package
        return aoti_load_package(package_path)
    except ImportError:
        pass
    
    # Intento 2: PyTorch 2.4 / algunas versiones de 2.3
    try:
        from torch._export import aot_load
        return aot_load(package_path, "cuda")
    except ImportError:
        pass

    # Intento 3: PyTorch 2.2 / 2.3 (Versiones tempranas)
    try:
        from torch._export import aoti_load
        return aoti_load(package_path, "cuda")
    except ImportError:
        raise ImportError(f"No se encontró la función AOTI adecuada en tu versión de PyTorch ({torch.__version__}).")

class MatMulModel(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)

def _bench_one(model, label, M, N, K, dtype, package_dir):
    """Compila + mide un caso. Devuelve dict de resultado o None si falla."""
    try:
        torch.cuda.empty_cache()
        a = torch.zeros((M, K), device=device, dtype=dtype)
        b = torch.zeros((K, N), device=device, dtype=dtype)

        package_path = os.path.join(package_dir, f"matmul_{label}_{M}_{N}_{K}.pt2")
        ep = export(model, (a, b))
        torch._inductor.aoti_compile_and_package(ep, package_path=package_path)
        fast_matmul = load_aoti_model(package_path)

        for _ in range(10):
            fast_matmul(a, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 100 if (M*N*K) < (8192**3) else 10
        start.record()
        for _ in range(iters):
            fast_matmul(a, b)
        end.record()
        torch.cuda.synchronize()

        avg_time_ms = start.elapsed_time(end) / iters
        avg_time_sec = avg_time_ms / 1000.0
        flops = 2.0 * M * N * K
        tflops = flops / (avg_time_sec * 1e12)
        del a, b, fast_matmul
        return {"Type": label, "M": M, "N": N, "K": K,
                "Time_ms": avg_time_ms, "TFLOPS": tflops}
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n⚠️ OOM en {M}x{N}x{K}. Saltando.")
            torch.cuda.empty_cache()
        else:
            print(f"\n❌ Error de ejecución en {M}x{N}x{K}: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Error de AOTI Compilación en {M}x{N}x{K}: {e}")
        return None


def run_one_shot(M, N, K, dtype=torch.float16):
    """Ejecuta un unico caso (M,N,K) sin escribir CSV ni dejar .pt2 en disco."""
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 Modo: AOTI one-shot — {M}x{N}x{K} (sin guardar resultados)")
    model = MatMulModel().to(device)
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        res = _bench_one(model, "Custom", M, N, K, dtype, tmpdir)
    if res is not None:
        print(f"\n⏱️  Tiempo medio : {res['Time_ms']:.4f} ms")
        print(f"🔥 Rendimiento  : {res['TFLOPS']:.2f} TFLOPS")
    return res


def run_exhaustive_benchmark(custom_dims=None):
    # --- DEFINICIÓN DE CASOS ---
    dims_base = custom_dims if custom_dims else [1024, 2048, 4096, 8192, 16384, 32768]
    bench_1_combs = [("Square", d, d, d) for d in dims_base]

    K_fixed = 8192
    bench_2_combs = [("Fixed_K", i, i, K_fixed) for i in dims_base]

    all_tasks = bench_1_combs + bench_2_combs

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"🚀 Modo: AOTI (Ahead-Of-Time Compilation via torch.export)")
    print(f"📊 Ejecutando {len(all_tasks)} pruebas específicas...")

    results = []

    # Instanciamos el modelo base
    model = MatMulModel().to(device)
    model.eval()

    # Directorio para guardar los binarios AOTI generados
    os.makedirs("aoti_compiled_models", exist_ok=True)

    for label, M, N, K in tqdm(all_tasks, desc="Benchmarking"):
        res = _bench_one(model, label, M, N, K, torch.float16, "aoti_compiled_models")
        if res is not None:
            results.append(res)

    return pd.DataFrame(results)

def _parse_mnk(s):
    parts = [p for p in s.replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        d = int(parts[0]); return d, d, d
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise argparse.ArgumentTypeError("Usa --mnk N  o  --mnk M,N,K")


if __name__ == "__main__":
    #python benchmark_aot_compile.py --mnk 8192 --no-save
    parser = argparse.ArgumentParser(description="Benchmark AOTI matmul fp16")
    parser.add_argument("--mnk", type=_parse_mnk, default=None,
                        help="Tamano unico: '8192' (cuadrada) o 'M,N,K'. NO guarda resultados.")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Lista de tamanos para el sweep, ej: '1024,2048,4096'")
    parser.add_argument("--no-save", action="store_true",
                        help="Ejecuta el sweep pero no escribe el CSV.")
    args = parser.parse_args()

    try:
        import triton
        print(f"✅ Triton versión {triton.__version__} detectada.")
    except ImportError:
        print("⚠️ Advertencia: Triton no instalado.")

    # Modo one-shot: tamano arbitrario, sin escribir nada en disco
    if args.mnk is not None:
        M, N, K = args.mnk
        run_one_shot(M, N, K)
    else:
        custom_dims = ([int(x) for x in args.sizes.split(",")]
                       if args.sizes else None)
        df = run_exhaustive_benchmark(custom_dims=custom_dims)

        if args.no_save:
            print("\n💾 --no-save activo: no se escribe CSV.")
        else:
            filename = "../results/rtx4090_torch_aoti_benchmark.csv"
            df.to_csv(filename, index=False)
            print(f"\n✅ Benchmark completado. Guardado en {filename}")

        print("\n🔥 Top 5 Rendimiento (TFLOPS) AOTI:")
        print(df.sort_values(by="TFLOPS", ascending=False).head(5).to_markdown())