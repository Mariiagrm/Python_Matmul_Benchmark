#!/bin/bash
# =====================================================================
#  ncu_executor.sh  —  Profiling de UN SOLO kernel con Nsight Compute
#  Target: benchmark_fp32 (cuBLAS GemmEx, FP16 in / FP32 acc, sm_89)
#
#  Uso:
#     bash ./ncu_executor.sh                # tamaño 4096, perfila 1 kernel
#     bash ./ncu_executor.sh 8192           # tamaño M=N=K=8192
#
#  Salida: ../../nvidia_nsight/ncu_fp32_cuda.ncu-rep
#  Abrir con:  ncu-ui ncu_fp32_cuda.ncu-rep   (o la GUI Nsight Compute)
# =====================================================================
set -e

# 1. Rutas CUDA (dentro del contenedor nvcr.io/nvidia/pytorch)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 2. Configuración
SIZE="${1:-4096}"            # M=N=K (4096 = compute-bound, rápido de perfilar)
BINARY="./benchmark_fp32"
OUT_DIR="../../nvidia_nsight"
OUT_NAME="ncu_fp32_cuda"

mkdir -p "$OUT_DIR"

# 3. Compilar si hace falta
if [ ! -f "$BINARY" ]; then
    echo "🔧 Compilando $BINARY ..."
    make
fi

# 4. Perfilar UN solo kernel
#    -s 10  -> salta los 10 lanzamientos de warmup
#    -c 1   -> captura solo 1 kernel (el primero en estado estacionario)
#    --set full -> todas las secciones (SOL, Compute, Memory, Occupancy...)
#    Nota: profiling con --set full reproduce el kernel muchas veces;
#          por eso limitamos a 1 kernel para que termine en segundos.
echo "🔬 Perfilando 1 kernel de cublasGemmEx (M=N=K=$SIZE)..."
echo "   Salida -> $OUT_DIR/$OUT_NAME.ncu-rep"

ncu \
    --set full \
    --skip 10 \
    --launch-count 1 \
    --target-processes all \
    -o "$OUT_DIR/$OUT_NAME" \
    -f \
    "$BINARY" "$SIZE"

echo ""
echo "✅ Listo. Reporte: $OUT_DIR/$OUT_NAME.ncu-rep"
echo "   Ver en consola:  ncu --import $OUT_DIR/$OUT_NAME.ncu-rep --page details | less"
echo "   Ver en GUI:      ncu-ui $OUT_DIR/$OUT_NAME.ncu-rep"
