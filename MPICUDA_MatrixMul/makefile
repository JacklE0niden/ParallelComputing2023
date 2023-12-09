CC = mpicc
NVCC = /usr/bin/nvcc
CFLAGS = -fopenmp -g -w -O4 -I..
NVCCFLAGS = -Xcompiler -fopenmp -g -w -O4 -I..  
MPI = -I /usr/include/x86_64-linux-gnu/mpich
LIB = -L /usr/lib/ -lmpich -lopa -lmpl -lpthread




all: 
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} matmul_row.cu -o matmul_row
	${NVCC} ${NVCCFLAGS} ${MPI} ${LIB} matmul_col.cu -o matmul_col

clean:
	-rm matmul_row matmul_col