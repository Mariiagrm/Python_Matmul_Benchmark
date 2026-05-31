#!/usr/bin/env bash
# Ejecuta los 3 benchmarks de Python (eager fp16, torch.compile y AOTI)
# para los tamanos pasados como parametro y agrupa la salida por tamano.
#
# Uso:
#   ./run_comparison.sh 1024 2048 4096
#   ./run_comparison.sh 8192 16384x16384x8192
#
# Cada argumento es un tamano para --mnk:
#   N            -> matmul cuadrada NxNxN
#   M,N,K  o MxNxK -> matmul rectangular

# # Resumen rápido (sections por defecto)
# ncu --target-processes all python3 ../benchmark_fp16_fp16.py --mnk 1024,1024,8192 --no-save

# # Set completo (ocupancia, memoria, instrucciones, roofline...)
# ncu --set full -o reporte python3 ../benchmark_fp16_fp16.py --mnk 1024,1024,8192 --no-save

# # Sólo un kernel concreto y limitando launches (mucho más rápido)
# ncu --kernel-name regex:matmul --launch-skip 10 --launch-count 1 --set full \
#     -o reporte python3 ../benchmark_fp16_fp16.py --mnk 1024,1024,8192 --no-save

# # Métricas concretas
# ncu --metrics sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
# sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed \
#     python3 ../benchmark_fp16_fp16.py --mnk 1024,1024,8192 --no-save

#
# Los resultados NO se guardan en CSV (modo one-shot con --mnk).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
# nsight python3 ../benchmark_fp16_fp16.py --mnk 1024,1024,8192 --no-save

SCRIPTS=(
    "benchmark_fp16_fp16.py:Eager FP16"
    "benchmark_torch_compile.py:torch.compile"
    "benchmark_aot_compile.py:AOTI"
)

if [ "$#" -lt 1 ]; then
    echo "Uso: $0 <tamano1> [tamano2 ...]"
    echo "Ejemplo: $0 1024 2048 4096"
    echo "         $0 8192 4096x4096x8192"
    exit 1
fi

SIZES=("$@")

LOG_DIR="$SCRIPT_DIR/log/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " Comparacion de benchmarks matmul FP16"
echo " Tamanos : ${SIZES[*]}"
echo " Logs    : $LOG_DIR"
echo "================================================================"

for size in "${SIZES[@]}"; do
    # Normaliza MxNxK -> M,N,K para --mnk
    mnk="${size//x/,}"
    safe="${size//,/x}"

    echo
    echo "################################################################"
    echo "#  Tamano: $size"
    echo "################################################################"

    for entry in "${SCRIPTS[@]}"; do
        script="${entry%%:*}"
        label="${entry##*:}"
        log_file="$LOG_DIR/${script%.py}_${safe}.log"

        echo
        echo "---- [$label] $script  --mnk $mnk ----"

        if [ ! -f "$script" ]; then
            echo "  (omitido: $script no existe)"
            continue
        fi

        start=$(date +%s)
        "$PYTHON" "$script" --mnk "$mnk" --no-save \
            > "$log_file" 2>&1
        rc=$?
        end=$(date +%s)
        dur=$((end - start))

        if [ $rc -ne 0 ]; then
            echo "  ERROR (rc=$rc, ${dur}s). Ultimas lineas:"
            tail -n 15 "$log_file" | sed 's/^/    /'
            continue
        fi

        # Extrae las metricas clave del log
        grep -E "Tiempo medio|Rendimiento|TFLOPS" "$log_file" | sed 's/^/  /' \
            || echo "  (sin metricas detectadas, ver $log_file)"
        echo "  duracion: ${dur}s   log: $log_file"
    done
done

echo
echo "================================================================"
echo " Listo. Logs completos en: $LOG_DIR"
echo "================================================================"
