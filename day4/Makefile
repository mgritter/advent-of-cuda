CUDA_PATH = /usr/local/cuda-6.5
NVCC = $(CUDA_PATH)/bin/nvcc

# Hardcoded for ARM7, to compile on NVidia Jetson TK1
OS_SIZE = 32
OS_ARCH = armv7l
ARCH_FLAGS = -target-cpu-arch ARM
GENCODE_FLAGS = -gencode arch=compute_32,code=sm_32
LIBRARIES = -lcudart
CFLAGS = -O2
# CFLAGS = -g -G

all: day4

day4.o: day4.cu
	$(NVCC) $(CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

day4: day4.o
	$(NVCC) $(CFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
