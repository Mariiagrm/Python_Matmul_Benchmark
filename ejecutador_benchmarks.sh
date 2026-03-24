#!/bin/bash

# ==========================================
# Script Automatizado de Benchmarks RTX 4090
# ==========================================

# 1. Medida de seguridad: Restaurar relojes pase lo que pase al salir
trap 'echo -e "\n[INFO] Restaurando frecuencias de la GPU a la normalidad...";  nvidia-smi --reset-gpu-clocks; exit' INT TERM EXIT

echo "🚀 Iniciando suite de benchmarks..."

# 2. Bloquear la frecuencia de la GPU (Prevenir Thermal Throttling)
echo "[INFO] Fijando los relojes de la GPU a 2500 MHz..."
 nvidia-smi -pm 1

# Fija la frecuencia del núcleo al máximo base/boost (ej. 2520 MHz para la 4090)

# Fija la frecuencia de la memoria al máximo
 nvidia-smi -lmc 10501
 nvidia-smi --lock-gpu-clocks=2500,2500

# Comprobar si el comando anterior funcionó
if [ $? -ne 0 ]; then
    echo "❌ Error al intentar bloquear los relojes. ¿Tienes permisos de sudo?"
    exit 1
fi

echo "[INFO] GPU bloqueada a 2500 MHz. Empezando pruebas..."
echo "--------------------------------------------------------"

# 3. Ejecutar los benchmarks secuencialmente
echo "▶️  Ejecutando: AOT Compile Benchmark"
python benchmarks/benchmark_aot_compile.py
echo "✅ Completado: AOT Compile"
echo "--------------------------------------------------------"

echo "▶️  Ejecutando: FP16 Benchmark"
python benchmarks/benchmark_fp16_fp16.py
echo "✅ Completado: FP16"
echo "--------------------------------------------------------"

echo "▶️  Ejecutando: JIT Torch Compile Benchmark"
# Permite a cuBLAS usar algoritmos más experimentales/agresivos
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_LOGDEST_DBG=stdout
export CUBLAS_LOGINFO_DBG=1
export TORCH_LOGS="graph_breaks,recompiles,perf_hints"
python benchmarks/benchmark_torch_compile.py
echo "✅ Completado: Torch JIT Compile"
echo "--------------------------------------------------------"

echo "▶️  Ejecutando: FP16 CUDA Benchmark"
bash benchmarks/ejecutador_cuda.sh
echo "✅ Completado: FP16 CUDA"
echo "--------------------------------------------------------"

# 4. Procesamiento de datos y visualización
echo "▶️  Ejecutando: Ordenación de Benchmarks"
python utils/ordenarBenchmark.py
echo "✅ Completado: Datos ordenados"
echo "--------------------------------------------------------"

echo "▶️  Ejecutando: Generación de Gráficas"
python utils/crearGrafica.py
echo "✅ Completado: Gráficas creadas"
echo "--------------------------------------------------------"

# 5. Fin del script (El 'trap' del inicio se encargará de resetear los relojes automáticamente)
echo " Todo el flujo (benchmarks + gráficas) ha finalizado con éxito."