#!/bin/bash

# ==========================================
# Configuración
# ==========================================
CSV_FILE="benchmark_mma-matmul.csv"
RUNNER="./runner" # Ajusta la ruta a tu ejecutable

# Arrays de datos
dims_base=(1024 2048 4096 8192 16384 32768)
kernels=(32)

K_FIXED=8192

# Escribir la cabecera del CSV
echo "Kernel,Type,M,N,K,Time_ms,TFLOPS" > "$CSV_FILE"

# ==========================================
# 1. Configuración Hardware (con Trap de seguridad)
# ==========================================
# El 'trap' garantiza que si cancelas el script con Ctrl+C, los relojes se resetean.
trap 'echo -e "\n[INFO] Restaurando relojes...";  nvidia-smi --reset-gpu-clocks;  nvidia-smi --reset-memory-clocks;  nvidia-smi -pm DISABLED; exit' INT TERM EXIT

echo "[INFO] Bloqueando frecuencias de GPU..."
 nvidia-smi -pm ENABLED
 nvidia-smi --lock-gpu-clocks=2520,2520
 nvidia-smi --lock-memory-clocks=10501,10501

# ==========================================
# 2. Función de Extracción y Cálculo
# ==========================================
run_and_save() {
    local kernel=$1
    local type=$2
    local m=$3
    local n=$4
    local k=$5

    # -s 5: salta los warmups
    # -c 5: PERFILA SOLO 5 EJECUCIONES (si no pones -c, ncu perfilará las 100 repeticiones y tardará una eternidad)
    # 2>&1: Redirige todo para poder leerlo con awk
    # ncu --set empty --metrics gpu__time_duration.sum -s 10 -c 1 -k regex:'^(?!shmem*)' --clock-control none --print-summary per-gpu ./runner 31 4096 4096 4096
    #read unit max_val <<< $(echo "$OUTPUT" | awk '/^ *gpu__time_duration.sum / {print $2, $4}')

    local NCU_CMD="ncu --set empty --metrics gpu__time_duration.sum -s 10 -c 1 -k regex:'^(?!shmem*)' --clock-control none --print-summary per-gpu $RUNNER $kernel $m $n $k"    
    # Capturar la salida completa de NCU
    OUTPUT=$(eval $NCU_CMD 2>&1)

    # Buscar la fila "Duration", coger la Unidad (columna 2) y el Maximum (columna 4)
    read unit max_val <<< $(echo "$OUTPUT" | awk '/^ *gpu__time_duration.sum / {print $2, $4}')

    # Control de errores (por si hay OOM en matrices gigantes)
    if [ -z "$max_val" ]; then
        echo "      ❌ Error: No se extrajo la Duración (Posible Out Of Memory)."
        return
    fi

    # Convertir el tiempo extraído a milisegundos (ncu puede variar la unidad de salida)
    local time_ms
    if [ "$unit" == "usecond" ]; then
        time_ms=$(awk "BEGIN {print $max_val / 1000.0}")
    elif [ "$unit" == "msecond" ]; then
        time_ms=$max_val
    elif [ "$unit" == "second" ]; then
        time_ms=$(awk "BEGIN {print $max_val * 1000.0}")
    elif [ "$unit" == "nsecond" ]; then
        time_ms=$(awk "BEGIN {print $max_val / 1000000.0}")
    else
        time_ms=$max_val # Fallback
    fi

    # Calcular TFLOPS usando bc o awk (evitando desbordamiento)
    # Fórmula: (2 * 16384 * 16384 * 16384) / (30.39mseq * 10^9) =  tflops 
    local tflops
    tflops=$(awk "BEGIN {
        flops = 2.0 * $m * $n * $k;
        # Convertimos ms a segundos para el cálculo: ms * 10^9 no, es ms * 10^-3
        # TFLOPS = FLOPS / (segundos * 10^12)
        tflops = flops / ($time_ms / 1000.0) / 1000000000000.0;
        printf \"%.4f\", tflops;
    }")

    echo "      ✅ Result: ${time_ms} ms -> ${tflops} TFLOPS"
    
    # Guardar en CSV
    echo "$kernel,$type,$m,$n,$k,$time_ms,$tflops" >> "$CSV_FILE"
}

# ==========================================
# 3. Bucle Principal de Benchmarks
# ==========================================
for kernel in "${kernels[@]}"; do
    echo "=================================================="
    echo "🚀 PERFILANDO KERNEL: $kernel"
    echo "=================================================="

    # --- TAREA 1: Matrices Cuadradas ---
    echo "🔹 Procesando Tarea 1 (Cuadradas)..."
    for d in "${dims_base[@]}"; do
        echo "   -> Ejecutando M=$d, N=$d, K=$d"
        run_and_save "$kernel" "Square" $d $d $d
    done

    # --- TAREA 2: K Fijo ---
    echo "🔹 Procesando Tarea 2 (K=$K_FIXED)..."
    for i in "${dims_base[@]}"; do
        echo "   -> Ejecutando M=$i, N=$i, K=$K_FIXED"
        run_and_save "$kernel" "Fixed_K" $i $i $K_FIXED
    done
done

# El trap se encarga de soltar los relojes al llegar aquí.
echo -e "\n🏁 Benchmark completado. Datos guardados en $CSV_FILE"