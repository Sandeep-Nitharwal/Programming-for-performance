CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-O2 -std=c++17
CPPFLAGS=
CUDAFLAGS=-O2 -ccbin=g++-9 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx

.DEFAULT_TARGET: all
# First target is the default target
all: problem1 problem2 problem3 problem4

# $@ is a variable that contains the target name.
problem1: problem1-v1 problem1-v2 problem1-v3 problem1-v4 problem1-v5
problem2: problem2t
problem3: problem3-v1 problem3-v2 problem3-v3 problem3-v4
problem4: problem4_2D problem4_3D problem4_2D_OPT problem4_3D_OPT

problem1-v1: problem1/problem1-v1.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem1-v2: problem1/problem1-v2.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem1-v3: problem1/problem1-v3.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem1-v4: problem1/problem1-v4.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem1-v5: problem1/problem1-v5.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem2t: problem2/problem2.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v1: problem3/problem3-v1.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v2: problem3/problem3-v2.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v3: problem3/problem3-v3.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v4: problem3/problem3-v4.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem4_2D: problem4/problem4_2D.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem4_3D: problem4/problem4_3D.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem4_2D_OPT: problem4/problem4_2D_OPT.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem4_3D_OPT: problem4/problem4_3D_OPT.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

# The target is not a filename
clean:
	rm */*.out */compile_commands.json
