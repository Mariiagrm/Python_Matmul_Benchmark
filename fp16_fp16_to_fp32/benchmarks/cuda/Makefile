# =====================================================================
#  Makefile para benchmark_fp32 (RTX 4090 / Ada Lovelace, sm_89)
# =====================================================================

NVCC      ?= nvcc
ARCH      ?= sm_89
TARGET    := benchmark_fp32
SRC       := benchmark_fp32.cu

NVCCFLAGS := -O3 -arch=$(ARCH) -use_fast_math -std=c++17 \
             -lineinfo --ptxas-options=-v \
             -Xcompiler -Wall

LDFLAGS   := -lcublas

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) 4096 4096 4096

clean:
	rm -f $(TARGET)