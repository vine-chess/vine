EXE = vine
FILES = $(shell find src -name '*.cpp')

OBJS = $(FILES:.cpp=.o)

OPTIMIZE ?= -O3 -flto

FLAGS = -std=c++20 -fconstexpr-steps=100000000
FLAGS += $(EXTRA_FLAGS)
FLAGS += $(OPTIMIZE)

ifdef EVALFILE
	FLAGS += -DEVALFILE=\"$(EVALFILE)\"
else
	FLAGS += -DVALUEFILE=\"$(VALUEFILE)\" -DPOLICYFILE=\"$(POLICYFILE)\"
endif

CC ?= gcc
CXX ?= g++

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

ifeq ($(MAKECMDGOALS),datagen)
	FLAGS += -DDATAGEN
endif

datagen: all

%.o: %.cpp
	$(CXX) $(FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@

all: $(OBJS)
	$(CXX) $(FLAGS) $(OBJS) -o $(EXE)

clean:
	rm -f $(OBJS)