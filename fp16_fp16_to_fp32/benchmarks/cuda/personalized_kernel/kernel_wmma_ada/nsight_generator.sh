#!/bin/bash
# 1) Recompila por si has tocado el .cu
make

# 2) Reporte binario (.ncu-rep) para abrir en Nsight Compute UI
ncu --set full \
    --kernel-name hgemm_wmma_kernel \
    --launch-skip 1 --launch-count 1 \
    -o report_wmma_4096 \
    -f ./hgemm_wmma_ada 4096 4096 4096

# 3) Mismo reporte en texto plano (para resultados_NSight.txt)
ncu --set full \
    --kernel-name hgemm_wmma_kernel \
    --launch-skip 1 --launch-count 1 \
    ./hgemm_wmma_ada 4096 4096 4096 \
    > resultados_NSight.txt

#ncu --import report_wmma_4096.ncu-rep --list-sections
