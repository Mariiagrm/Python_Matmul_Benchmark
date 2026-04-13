#!/bin/bash
# Uso: ./ejecutador_benchmarks.sh --a fp16 --profile 
#      ./ejecutador_benchmarks.sh --a fp32

# 1. Variables por defecto
DO_PROFILE=false
ARCH_MODE="fp16" # Valor por defecto
BENCHMAKS="true"
# Guardamos la ruta absoluta del directorio donde vive este script .sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. Procesamiento de argumentos
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --a)
            ARCH_MODE="$2"
            shift 2
            ;;
        --profile)
            DO_PROFILE=true
            shift
            ;;
        --sb)
            BENCHMAKS="false"
            shift
            ;;
        -h|--help)
            echo "Uso: $0 [--a fp16 | fp32] [--profile]"
            exit 0
            ;;
        *)
            echo "❌ Argumento desconocido: $1"
            echo "Uso: $0 [--a fp16 | fp32] [--profile] [--sb]"
            echo "  --a [fp16|fp32]   : Especifica el modo de arquitectura (predeterminado: fp16)"
            echo "  --profile         : Activa el modo de profiling con Nsight Systems y Compute"
            echo "  --sb              : Saltar ejecución de benchmarks (skip benchmarks)"
            echo "  -h, --help        : Muestra esta ayuda"
            exit 1
            ;;
    esac
done

# 3. Determinación del directorio raíz basado en los argumentos
if [ "$ARCH_MODE" == "fp16" ]; then
    ROOT_DIR="./fp16_fp16_to_fp16"
elif [ "$ARCH_MODE" == "fp32" ]; then
    ROOT_DIR="./fp16_fp16_to_fp32"
else
    echo "❌ Error: Modo de arquitectura '$ARCH_MODE' no reconocido."
    echo "Usa 'fp16' o 'fp32'."
    exit 1
fi

if [ "$DO_PROFILE" = true ]; then
    TARGET_DIR="$ROOT_DIR/unitary_benchmarks"
else
    TARGET_DIR="$ROOT_DIR/benchmarks"
fi

# 4. Validar existencia del directorio y entrar en él
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ Error: El directorio $TARGET_DIR no existe."
    exit 1
fi

echo "[INFO] Entrando al directorio de trabajo: $TARGET_DIR"
cd "$TARGET_DIR" || exit 1

# 5. Medida de seguridad y limpieza al salir
trap 'echo -e "\n[INFO] Restaurando frecuencias de la GPU...";  nvidia-smi --reset-gpu-clocks; exit' INT TERM EXIT

echo "Iniciando suite de benchmarks..."
if [ "$DO_PROFILE" = true ]; then
    echo "🔍 MODO PROFILING ACTIVADO (Nsight Systems y Compute)"
fi

# 6. Bloquear la frecuencia de la GPU
echo "[INFO] Fijando los relojes de la GPU a 2500 MHz..."
 nvidia-smi -pm 1
 nvidia-smi -lmc 10501
 nvidia-smi --lock-gpu-clocks=2500,2500

if [ $? -ne 0 ]; then
    echo "❌ Error al intentar bloquear los relojes. Asegúrate de tener permisos de ."
    exit 1
fi

echo "--------------------------------------------------------"

# 7. Ejecución de Benchmarks estándar
run_benchmarks() {
    if [ "$ARCH_MODE" == "fp16" ]; then
        echo "▶️  Ejecutando: FP16/FP16 Benchmark"
        python ./benchmark_fp16_fp16.py
        echo "✅ Completado: FP16/FP16"
        echo "--------------------------------------------------------"

        echo "▶️  Ejecutando: FP16/fp16 CUDA Benchmark"
        bash ./cuda_executor.sh
        echo "✅ Completado: cuda FP16/FP16"
        echo "--------------------------------------------------------"

    else
        echo "▶️  Ejecutando: FP16/FP32 Benchmark"
        python ./benchmark_fp16_fp32.py
        echo "✅ Completado: FP16/FP32"
        echo "--------------------------------------------------------"

        echo "▶️  Ejecutando: FP16/fp32 CUDA Benchmark"
        bash ./cuda_executor.sh
        echo "✅ Completado: cuda FP16/FP32"
        echo "--------------------------------------------------------"

        echo "▶️  Ejecutando: FP16/FP32 Matmul Benchmark (cuBLAS)"
        if [ -f "./mma-matmul/ejecutador.sh" ]; then
            cd "./mma-matmul" || exit 1
            bash "./ejecutador.sh"
            cd .. || exit 1
        else
            echo "⚠️  Advertencia: No se encontró ./mma-matmul/ejecutador.sh"
        fi
    fi
    echo "▶️  Ejecutando: AOT Compile Benchmark"
    python ./benchmark_aot_compile.py

    echo "▶️  Ejecutando: JIT Torch Compile Benchmark"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export TORCH_LOGS="graph_breaks,recompiles,perf_hints"
    python ./benchmark_torch_compile.py



    
}

if [ "$BENCHMAKS" == "true" ]; then
    run_benchmarks
else
    echo "⏭️  Saltando ejecución de benchmarks (modo --sb activado)."
fi

echo "--------------------------------------------------------"
if [ -f "$ROOT_DIR/utils/sortBenchmarks.py" ] && [ -f "$ROOT_DIR/utils/plotCreate.py" ]; then

    python "$ROOT_DIR/utils/sortBenchmarks.py"
    python "$ROOT_DIR/utils/plotCreate.py"
    echo "✅ Datos y Gráficas completados."
else
    echo "⚠️  Advertencia: No se encontraron los scripts en $ROOT_DIR/utils/. Saltando generación de gráficas."
fi



# 9. SECCIÓN DE PROFILING (Opcional)
if [ "$DO_PROFILE" = true ]; then
    echo "--------------------------------------------------------"
    echo "📊 Iniciando Captura de Nsight Systems (.nsys-rep)..."
    
    # Creamos la carpeta de resultados en la raíz del proyecto usando nuestra variable ancla
    NSIGHT_DIR="$ROOT_DIR/nvidia_nsight"
    mkdir -p "$NSIGHT_DIR"

    # Nsys (Timelines) - Como seguimos en TARGET_DIR, los ./benchmark_*.py funcionan perfectamente
    nsys profile -t cuda,osrt,nvtx --gpu-metrics-device=all -o "$NSIGHT_DIR/perfil_aot_nsys" -f python ./benchmark_aot_compile.py
    nsys profile -t cuda,osrt,nvtx --gpu-metrics-device=all -o "$NSIGHT_DIR/perfil_eager_nsys" -f python ./benchmark_fp16_fp32.py
    nsys profile -t cuda,osrt,nvtx --gpu-metrics-device=all -o "$NSIGHT_DIR/perfil_eager_nsys" -f python ./benchmark_fp16_fp16.py
    nsys profile -t cuda,osrt,nvtx --gpu-metrics-device=all -o "$NSIGHT_DIR/perfil_jit_nsys" -f python ./benchmark_torch_compile.py

    echo "🔬 Iniciando Captura de Nsight Compute (.ncu-rep)..."
    echo "[ADVERTENCIA] Esto puede tardar varios minutos por kernel."
    
    # NCU (Hardware Analysis)
    ncu --target-processes all --nvtx --set full -o "$NSIGHT_DIR/profile_report_aot" -f python ./benchmark_aot_compile.py
    ncu --target-processes all --nvtx --set full -o "$NSIGHT_DIR/profile_report_eager" -f python ./benchmark_fp16_fp32.py
    nsys profile -t cuda,osrt,nvtx --gpu-metrics-device=all -o "$NSIGHT_DIR/perfil_eager_nsys" -f python ./benchmark_fp16_fp16.py
    ncu --set full -c 5 -o "$NSIGHT_DIR/profile_report_jit" -f python ./benchmark_torch_compile.py
    
    echo "✅ Profiling completado."
else
    echo "--------------------------------------------------------"
    echo "⏭️  Saltando profiling (usa --profile para activarlo)."
fi

echo "🏁 Todo el flujo ha finalizado con éxito."