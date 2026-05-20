# =====================================================================
#  Makefile para los kernels GEMM del TFG (RTX 4090 / Ada Lovelace, sm_89)
#    - sgemm_ada        : FP32 sobre CUDA Cores  (sub-tile 8x8 por hilo)
#    - hgemm_wmma_ada   : FP16 sobre Tensor Cores (WMMA, fragmentos 16x16x16)
# =====================================================================

NVCC      ?= nvcc
ARCH      ?= sm_89
SIZE      ?= 4096

NVCCFLAGS := -O3 -arch=$(ARCH) -use_fast_math -std=c++17 \
             -lineinfo --ptxas-options=-v \
             -Xcompiler -Wall

LDFLAGS   := -lcublas

TARGETS   :=  hgemm_wmma_ada

.PHONY: all clean run run-fp32 run-wmma

all: $(TARGETS)


hgemm_wmma_ada: hgemm_wmma_ada.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: run-wmma


run-wmma: hgemm_wmma_ada
	./hgemm_wmma_ada $(SIZE) $(SIZE) $(SIZE)

clean:
	rm -f $(TARGETS)
