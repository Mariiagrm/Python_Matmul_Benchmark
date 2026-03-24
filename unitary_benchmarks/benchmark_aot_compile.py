import torch
import pandas as pd
import time
import os
from tqdm import tqdm
from torch.export import export
import torch._inductor
from torch._inductor.package import load_package

# --- 1. CONFIGURACIÓN DE PRECISIÓN Y BACKEND ---
device = torch.device("cuda")
torch._logging.set_logs(output_code=True)

# TODO Resuelto: Para forzar ALLOW_TF32=False en Triton, desactivamos TF32 a nivel global
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

def run_exhaustive_benchmark():
    # --- DEFINICIÓN DE CASOS ---
    dims_base = [ 8192] 
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
        dtype = torch.float16
        dtype_str = "fp16" 
            
        try:
            torch.cuda.empty_cache()

            # Crear tensores (usamos zeros para evitar NaN/Inf durante el profiling de Triton)
            a = torch.zeros((M, K), device=device, dtype=dtype)
            b = torch.zeros((K, N), device=device, dtype=dtype)

            # Nombre del archivo donde se guardará el modelo pre-compilado
            package_path = f"aoti_compiled_models/matmul_{label}_{M}_{N}_{K}.pt2"

            # ==========================================================
            # FASE AOTI (AHEAD-OF-TIME INDUCTION)
            # ==========================================================
            
            # 1. Exportar el modelo capturando su grafo (ExportedProgram)
            # Pasamos un ejemplo de los tensores para que el compilador fije las formas (static shapes)
            ep = export(model, (a, b))
            
          
            
            # 2. Compilar y empaquetar el grafo en un binario para el disco
            torch._inductor.aoti_compile_and_package(ep, package_path=package_path)
            
            # 3. Cargar la función/modelo compilado desde el disco
            # --- USAMOS LA FUNCIÓN DE CARGA ROBUSTA ---
            fast_matmul = load_aoti_model(package_path)
            
            # ==========================================================

            # --- WARM-UP ---
            for _ in range(10):
                fast_matmul(a, b)
            
            torch.cuda.synchronize()

            # --- MEDICIÓN ---
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Ajuste de iteraciones para no eternizar tamaños 32k
            iters = 100 if (M*N*K) < (8192**3) else 10
            
            start.record()
            for _ in range(iters):
                # Llamamos a la función cargada vía AOTI
                fast_matmul(a, b) 
            end.record()

            torch.cuda.synchronize()
            
            avg_time_ms = start.elapsed_time(end) / iters
            avg_time_sec = avg_time_ms / 1000.0
            
            flops = 2.0 * M * N * K
            tflops = flops / (avg_time_sec * 1e12)

            results.append({
                "Type": label,
                "Precision": dtype_str,
                "M": M, "N": N, "K": K,
                "Time_ms": avg_time_ms,
                "TFLOPS": tflops
            })

            # Limpiar referencias explícitamente
            del a, b, fast_matmul

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ OOM en {M}x{N}x{K}. Saltando.")
                torch.cuda.empty_cache()
            else:
                print(f"\n❌ Error de ejecución en {M}x{N}x{K}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ Error de AOTI Compilación en {M}x{N}x{K}: {e}")
            continue

    return pd.DataFrame(results)

if __name__ == "__main__":
    try:
        import triton
        print(f"✅ Triton versión {triton.__version__} detectada.")
    except ImportError:
        print("⚠️ Advertencia: Triton no instalado.")

    df = run_exhaustive_benchmark()
    
    filename = "../unique_results/rtx4090_torch_aoti_benchmark.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Benchmark completado. Guardado en {filename}")
    print("\n🔥 Top 5 Rendimiento (TFLOPS) AOTI:")
    print(df.sort_values(by="TFLOPS", ascending=False).head(5).to_markdown())