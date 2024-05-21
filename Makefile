CC_FILES=$(shell find ./ -name "*.cu")
EXE_FILES=$(CC_FILES:.cu=)
TEST_FILE = $(shell find ./ -name "test_*.cu")
TEST_EXE = $(TEST_FILE:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O2 -arch=sm_86 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas

clean:
	rm -rf $(EXE_FILES)

test: $(TEST_EXE)

%:%.cu
	nvcc -o $@ $< -O2 -arch=sm_80 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublas

GEMM_V1_FILE = test_gemm_v1.cu
GEMM_V1_EXE = $(GEMM_V1_FILE:.cu=)
gemm_v1: $(GEMM_V1_EXE)

%:%.cu
	nvcc -o $@ $< -O2 -arch=sm_80 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublas

GEMM_V2_FILE = test_gemm_v2.cu
GEMM_V2_EXE = $(GEMM_V2_FILE:.cu=)
gemm_v2: $(GEMM_V2_EXE)

%:%.cu
	nvcc -o $@ $< -O2 -arch=sm_80 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublas