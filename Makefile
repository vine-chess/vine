EXE = vine
FILES = $(shell find src -name '*.cpp' ! -name 'cuda_test.cpp')
CUDA_TEST_FILES = src/eval/value_network.cu src/eval/policy_network.cu

OBJS = $(FILES:.cpp=.o)
TEST_EXE = cuda_test
TEST_FILES = $(filter-out src/main.cpp,$(FILES)) src/tests/cuda_test.cpp
TEST_OBJS = $(TEST_FILES:.cpp=.cuda_test.o)
DATAGEN_OBJS = $(FILES:.cpp=.datagen.o)
DEPS = $(OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(DATAGEN_OBJS:.o=.d)

OPTIMIZE ?= -O3 -flto

FLAGS = -std=c++20 -fconstexpr-steps=100000000
FLAGS += $(EXTRA_FLAGS)
FLAGS += $(OPTIMIZE)
DEPFLAGS = -MMD -MP
TEST_FLAGS = $(filter-out -flto,$(FLAGS))
DATAGEN_FLAGS = $(filter-out -flto,$(FLAGS)) -DDATAGEN -DDATAGEN_CUDA
TEST_CXX = clang++
NVCC ?= nvcc
NVCCFLAGS = -std=c++20 -O3

DOWNLOAD_NETS = no

ifdef EVALFILE
	FLAGS += -DEVALFILE=\"$(EVALFILE)\"
else ifdef VALUEFILE
	ifdef POLICYFILE
		FLAGS += -DVALUEFILE=\"$(VALUEFILE)\" -DPOLICYFILE=\"$(POLICYFILE)\"
	else
		$(error POLICYFILE must be defined alongside VALUEFILE)
	endif
else
	VALUEFILE = net58.vn
	POLICYFILE = net21.pn
	FLAGS += -DVALUEFILE=\"$(VALUEFILE)\" -DPOLICYFILE=\"$(POLICYFILE)\"
	DOWNLOAD_NETS = yes
endif

CC ?= clang
CXX ?= clang++

ifeq ($(OS),Windows_NT)
	FLAGS += -static
endif

build ?= native

M64     = -m64 -mpopcnt
MSSE2   = $(M64) -msse -msse2
MSSSE3  = $(MSSE2) -mssse3
MAVX2   = $(MSSSE3) -msse4.1 -mbmi -mfma -mavx2
MAVX512 = $(MAVX2) -mavx512f -mavx512bw

ifeq ($(build), native)
	FLAGS += -march=native
else ifeq ($(findstring sse2, $(build)), sse2)
	FLAGS += $(MSSE2)
else ifeq ($(findstring ssse3, $(build)), ssse3)
	FLAGS += $(MSSSE3)
else ifeq ($(findstring avx2, $(build)), avx2)
	FLAGS += $(MAVX2)
else ifeq ($(findstring avx512, $(build)), avx512)
	FLAGS += $(MAVX512)
endif

.DEFAULT_GOAL := all 

.PHONY: nets compile-commands clean
nets:
ifeq ($(DOWNLOAD_NETS),yes)
	@if [ ! -f $(VALUEFILE) ]; then \
		curl -sOL https://github.com/vine-chess/vine-networks/raw/refs/heads/main/value/$(VALUEFILE); \
	fi
	@if [ ! -f $(POLICYFILE) ]; then \
		curl -sOL https://github.com/vine-chess/vine-networks/raw/refs/heads/main/policy/$(POLICYFILE); \
	fi
endif

%.o: %.cpp
	$(CXX) $(FLAGS) $(DEPFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(FLAGS) $(DEPFLAGS) -c $< -o $@

%.cuda_test.o: %.cpp
	$(TEST_CXX) $(TEST_FLAGS) $(DEPFLAGS) -c $< -o $@

%.datagen.o: %.cpp
	$(TEST_CXX) $(DATAGEN_FLAGS) $(DEPFLAGS) -c $< -o $@

all: nets $(OBJS)
	$(CXX) $(FLAGS) $(OBJS) -o $(EXE)

datagen: nets $(DATAGEN_OBJS)
	$(NVCC) $(NVCCFLAGS) $(DATAGEN_OBJS) $(CUDA_TEST_FILES) -o $(EXE)

cuda-test: nets $(TEST_OBJS)
	$(NVCC) $(NVCCFLAGS) $(TEST_OBJS) $(CUDA_TEST_FILES) -o $(TEST_EXE)

compile-commands:
	bear --output compile_commands.json -- $(MAKE) -B cuda-test

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(DATAGEN_OBJS) $(DEPS) $(TEST_EXE)

-include $(DEPS)
