# Looks like it works fine on old Compute Capability 5 cards
ARCH=	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_62,code=sm_62 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_70,code=compute_70

# Had to modify ptx assembler option
all:
	nvcc -g -O2 -x cu $(ARCH) -lineinfo -std=c++14 -Xcompiler="-fopenmp -Wall" -I.. main.cpp -o maingpu

cpu:
	g++ -g -O2 -std=c++14 -fopenmp -Wall -I.. main.cpp -o maincpu

clean:
	rm main
