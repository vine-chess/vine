EXE = vine
FILES = $(shell find src -name '*.cpp')

OBJS = $(FILES:.cpp=.o)

OPTIMIZE ?= -O3 -flto

FLAGS = -std=c++20 -fconstexpr-steps=100000000
FLAGS += $(EXTRA_FLAGS)
FLAGS += $(OPTIMIZE)

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
	VALUEFILE = net24.vn
	POLICYFILE = net3.pn
	FLAGS += -DVALUEFILE=\"$(VALUEFILE)\" -DPOLICYFILE=\"$(POLICYFILE)\"
	DOWNLOAD_NETS = yes
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


.PHONY: nets
nets:
ifeq ($(DOWNLOAD_NETS),yes)
	@if [ ! -f $(VALUEFILE) ]; then \
		curl -sOL https://github.com/vine-chess/vine-networks/raw/refs/heads/main/value/$(VALUEFILE); \
	fi
	@if [ ! -f $(POLICYFILE) ]; then \
		curl -sOL https://github.com/vine-chess/vine-networks/raw/refs/heads/main/policy/$(POLICYFILE); \
	fi
endif

datagen: all

%.o: %.cpp
	$(CXX) $(FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@

all: nets $(OBJS)
	$(CXX) $(FLAGS) $(OBJS) -o $(EXE)

clean:
	rm -f $(OBJS)
