# Makefile — gpu-batch-invariant study (SM120 / Blackwell only)
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_120

bik: cuda/bik.cu
	$(NVCC) -std=c++17 -O3 -arch=$(ARCH) cuda/bik.cu -o bik -lcublasLt -lcublas

clean:
	rm -f bik

.PHONY: clean