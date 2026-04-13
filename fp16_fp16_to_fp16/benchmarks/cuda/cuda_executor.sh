#!/bin/bash

# 1. Configuración de rutas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Configuración de archivos
BINARY="./benchmark_fp16"
OUTPUT_FILE="../../results/rtx4090_benchmark_cuda.csv"

# Verificar binario
if [ ! -f "$BINARY" ]; then
    echo "❌ Error: No se encuentra $BINARY. Compila antes de ejecutar."
     nvcc -O3 -arch=sm_89 -lcublas benchmark_fp16.cu -o benchmark_fp16
    #exit 1
fi

# 2. Crear cabecera del CSV
echo "Type,M,N,K,Time_ms,TFLOPS" > "$OUTPUT_FILE"

# Lista de dimensiones
dims_base=(1024 2046 4096 8192 16384 32768)

echo "🚀 Iniciando Benchmark... Los datos se guardarán en $OUTPUT_FILE"

# Función interna para ejecutar y limpiar la salida
run_and_save() {
    local type=$1
    local m=$2
    local n=$3
    local k=$4

    # Ejecutar y capturar salida
    raw_output=$($BINARY $m $n $k)

    # Extraer solo los números usando awk
    #Tiempo promedio por iteración: 30.0801 ms
    time_ms=$(echo "$raw_output" | grep "Tiempo promedio" | awk '{print $5}')
    tflops=$(echo "$raw_output" | grep "Rendimiento estimado" | awk '{print $3}')

    # Guardar en CSV
    echo "$type,$m,$n,$k,$time_ms,$tflops" >> "$OUTPUT_FILE"
}

# --- TAREA 1: Matrices Cuadradas ---
echo "🔹 Procesando Tarea 1 (Cuadradas)..."
for d in "${dims_base[@]}"; do
    echo "   -> Ejecutando $d x $d x $d"
    run_and_save "Square" $d $d $d
done

# --- TAREA 2: K Fijo ---
K_FIXED=8192
echo -e "\n🔹 Procesando Tarea 2 (K=$K_FIXED)..."
for i in "${dims_base[@]}"; do
    echo "   -> Ejecutando M=$i, N=$i"
    run_and_save "Fixed_K" $i $i $K_FIXED
done

echo -e "\n✅ ¡Listo! Archivo '$OUTPUT_FILE' generado con éxito."