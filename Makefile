#DB Add MaCh3 methodology of finding CUDA version and compute capability
CUDAVER=$(shell nvcc --version | grep -o 'V[0-9].*' |  cut -d. -f1 |sed 's:V::')

$(info $$CUDAVER is ${CUDAVER})

ifeq ($(shell expr $(CUDAVER) \< 9), 1)
ARCH=	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_32,code=sm_32 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_35,code=compute_35
else
ifeq ($(shell expr $(CUDAVER) \>= 11), 1)
ARCH= 	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_86,code=compute_86
else
ARCH=	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_32,code=sm_32 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_35,code=compute_35
endif
endif

ROOTLIBS := $(shell root-config --libs)
ROOTINCLUDES := $(shell root-config --incdir)
ROOTCFLAGS := $(shell root-config --cflags)

INCDIR = -I. -I..
LIB_OBJECTS += $(ROOTLIBS)

SYSLIB     = -lm
LINK_ARGS_BIN = $(SYSLIB) $(ROOTLIBS)
CXXFLAGS   = -Wall -O3 -g -fPIC $(ROOTCFLAGS) $(INCDIR) -Werror -std=c++11

CXXFLAGS += -fopenmp

AR=ar
ARFLAGS=rcsv

LD_SHARED=g++
SOFLAGS= -shared $(ROOTCFLAGS)

atmoscudapropagator.o:
	nvcc -g -O2 -x cu $(ARCH) -lineinfo -std=c++11 -Xcompiler="-fopenmp -Wall" -I. -c atmoscudapropagator.cuh

cuda_unique.o:
	nvcc -g -O2 -x cu $(ARCH) -lineinfo -std=c++11 -Xcompiler="-fopenmp -Wall" -I. -c cuda_unique.cuh

hpc_helpers.o:
	nvcc -g -O2 -x cu $(ARCH) -lineinfo -std=c++11 -Xcompiler="-fopenmp -Wall" -I. -c hpc_helpers.cuh

cpupropagator.o:
	g++ -c $(LINK_ARGS_BIN) $(CXXFLAGS) $(OMP_DEFINES) cpupropagator.hpp -o cpupropagator.o

propagator.o:
	g++ -c $(LINK_ARGS_BIN) $(CXXFLAGS) $(OMP_DEFINES) propagator.hpp -o propagator.o

libCUDAProb3.a: propagator.o cpupropagator.o cudapropagator.o cuda_unique.o hpc_helpers.o
	$(AR) $(ARFLAGS) $@ $^

libCUDAProb3.so: libCUDAProb3.a
	$(LD_SHARED) $(SOFLAGS) $(LIB_OBJECTS) $(ROOTLIBS) libCUDAProb3.a -o libCUDAProb3.so 

TestScript: libCUDAProb3.so
	nvcc -g -O2 -x cu $(ARCH) -lineinfo -std=c++11 -Xcompiler="-fopenmp -Wall -I${ROOTINCLUDES} $(ROOTLIBS)" -I. TestScript.cpp -o Exec

all: libCUDAProb3.so TestScript

clean:
	rm *.o *.a *.so
