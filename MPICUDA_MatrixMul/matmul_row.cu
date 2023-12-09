#include <stdio.h>
#include <cuda.h>
#include "mpi.h"

#define BLOCKSIZE 16

int IntializingMatrixVectors(float **, float **, float **, int , int , int , int);
int CheckDevice(int );
void printResultMatrix(float *, int , int, const char*);
float abs(float, float);
#define SAFE(call)                                                         \
            do{                                                                      \
                 cudaError_t err = call;                                             \
                 if(err != cudaSuccess)                                              \
                 {                                                                   \
                        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                         __FILE__, __LINE__, cudaGetErrorString( err) );             \
                         exit(1);                                                    \
                 }                                                                   \
               } while (0)                                                           \

// __global__ void MatrixMultiplication(float *Matrix, float *Vector, float *Solution, int r1, int c1, int r2, int c2, int VectorLength, int ScatterSize, int ThreadDim, int MyRank, int NumberofProcessors)
// {
// 	int tidx = threadIdx.x;// 获取当前线程在块内的索引
// 	// 计算当前线程处理的元素索引
// 	int RowStart = blockIdx.x * blockDim.x * ScatterSize;
// 	int RowEnd = RowStart + ScatterSize;	
// 	// 以下循环用于计算矩阵乘法
// 	for (int i = RowStart + tidx; i < RowEnd; i += blockDim.x) {
// 		// 内循环迭代矩阵的列
// 		for (int j = 0; j < c2; j++) {
// 			float sum = 0.0;
// 			for (int k = 0; k < r2; k++)
// 				sum += Matrix[i * c1 + k] * Vector[k * c2 + j];	
// 			// 将累加结果存储在输出矩阵 Solution 中
// 			Solution[i * c2 + j] = sum;
// 		}
// 	}
// 	__syncthreads();
// }

__global__ void MatrixMultiplication(float *MA, float *MB, float *Res, int r1, int c1, int c2, int ScatterSize, int myid) {
	//单线程矩阵乘法（一个元素）的核函数
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	// 取出矩阵的行和列（在网格中）
    if (row < ScatterSize && col < c2) { //单线程进行矩阵乘法
        float sum = 0.0;
        for (int k = 0; k < c1; ++k) {
            sum += MA[row * c1 + k] * MB[k * c2 + col];
        }
        Res[row * c2 + col] = sum;
    }
}

int main(int argc, char **argv)
{
	int myid, numprocs;
	int Root = 0, Index, Status = 1;
	float *MA, *ResultM, *MB, *ResultMatrix;
	float *MyMA, *MyResultMatrix;//每个进程负责的矩阵和计算结果
	float *DeviceMA, *DeviceMyResultM, *DeviceVectorB, *DeviceMB, *CPUResultM;
	int RowsNo, ColsNo, RowsNo2, ColsNo2, VectorSize, ScatterSize, IndexCol, IndexValue, DeviceStatus;
	int bsize, pinned;
	int print =0;
	int verify =0;
	double start_time,end_time ;
	//MPI Intialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); //当前进程的rank
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs); //获取进程的总数
	
	//Checking if valid number of arguements have been passed
	if(argc < 6)
	{
		if(myid == Root)
			printf("Usage:< mpirun >< -n >< Number of processors >< ./Program Name >< Number of Rows of Matri x>< Number of Columns of Matrix >< Rows of Matrix 2 > <Coloumns of Matrix 1>  <-v if verification is required>  <-p if print is required>\n");
		MPI_Finalize();
		exit(-1);
	}
	if ((argc >= 6 && strcmp(argv[5],"-v") == 0)){
		verify=1;
		printf("v=1");
	}

	if ((argc ==7 && strcmp(argv[6],"-p") == 0) || (argc == 6 && strcmp(argv[5],"-p")==0)){
		print=1;
	}
	//Assigning values to RowsNo, ColsNo, VectorSize from the arguements passed
	RowsNo = atoi( argv[1] ); //矩阵1的行数
	ColsNo = atoi( argv[2] ); //矩阵1的列数
	RowsNo2= atoi( argv[3] ); //矩阵2的行数
	ColsNo2= atoi( argv[4] ); //矩阵2的列数
	// pinned = atoi( argv[5]);

	bsize=RowsNo2*ColsNo2; //矩阵2的元素个数
	if (myid==0)
		printf("Resultant Matrix Number of Elements is %d\n\n\n",bsize);
	
	int elements;
	// 处理错误的输入
	if( ColsNo != RowsNo2)
	{
		if(myid == Root)
			printf("Entered wrong input, Number of columns of matrix 1 should be equal to number of rows of matrix 2 \n");
		MPI_Finalize();
		exit(-1);
	}

	if(RowsNo < numprocs)
	{
		if(myid == Root)
			printf("Given number of Rows of the matrix should be more than number of processors \n");
		MPI_Finalize();
		exit(-1);
	}

	//Checking if Matrix can be distributed evenly to all the nodes
	if(RowsNo % numprocs != 0)
	{
		if(myid == Root)
			printf("The Rows of the matrix can not be distributed evenly among processors \n");
		MPI_Finalize();
		exit(-1);
	}

	// 根进程调用随机化初始矩阵的方法
    if(myid == Root)
        Status = IntializingMatrixVectors(&MA, &MB, &ResultM, RowsNo, ColsNo, RowsNo2, ColsNo2);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&Status, 1, MPI_INT, Root, MPI_COMM_WORLD);
	//状态为0说明分配失败

	//Allocating memory for the Vector by all nodes expect root node
	if(myid != Root)
		MB = (float *)malloc(bsize * sizeof(float));

	//Broad casting the Vector to all the nodes from root node
	MPI_Bcast(MB, bsize, MPI_FLOAT, Root, MPI_COMM_WORLD);

	//计算分割大小以及结果矩阵的大小
	ScatterSize = RowsNo / numprocs;
	elements = (RowsNo*ColsNo2)/numprocs;

	MyMA = (float *)malloc(ScatterSize * ColsNo * sizeof(float));
	if(MyMA == NULL)
		Status = 0;
	
	MyResultMatrix = (float *)malloc(elements* sizeof(float));
	if(MyResultMatrix == NULL)
		Status = 0;

	MPI_Scatter(MA, ScatterSize * ColsNo, MPI_FLOAT, MyMA, ScatterSize * ColsNo, MPI_FLOAT, Root, MPI_COMM_WORLD);

	// printf("Process %d received:\n", myid);
	// for (int i = 0; i < ScatterSize; i++) {
	// 	for (int j = 0; j < ColsNo; j++) {
	// 		printf("%f ", MyMA[i * ColsNo + j]);
	// 	}
	// 	printf("\n");
	// }

	DeviceStatus = CheckDevice(myid);
	
    if(DeviceStatus == 0)
    {
		printf("cuda is fucked away\n");
        printf("Processor with rank %d doing partial product of matrix on CPU \n",myid);
		//TODO:增加计时点
		for(Index = 0 ; Index < ScatterSize ; Index++) 
		{
    		MyResultMatrix[Index] =0;
    		IndexValue = Index * ColsNo;
    		for(IndexCol = 0; IndexCol < ColsNo; IndexCol++) 
				MyResultMatrix[Index] += (MyMA[IndexValue++] * MB[IndexCol]);
    	}
	}
	else
	{
		cudaSetDevice(myid);
		SAFE( cudaMalloc( (void **)&DeviceMA, ScatterSize * ColsNo * sizeof(float) ) );
		SAFE( cudaMalloc( (void **)&DeviceMB, bsize*sizeof(float) ) );
		SAFE( cudaMalloc( (void **)&DeviceMyResultM, elements * sizeof(float) ) );
		cudaMemcpy( (void *)DeviceMA, (void *)MyMA, ScatterSize * ColsNo * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( (void *)DeviceMB, (void *)MB,  bsize*sizeof(float), cudaMemcpyHostToDevice );
		start_time = MPI_Wtime();
    	// dim3 gridSize((ColsNo2 + blockSize.x - 1) / (8*blockSize.x), (RowsNo + blockSize.y - 1) / (8*blockSize.y));
		dim3 blockSize(16, 16);
		dim3 gridSize((ColsNo2 + blockSize.x - 1) / blockSize.x, (ScatterSize + blockSize.y - 1) / blockSize.y);
		MatrixMultiplication<<<gridSize, blockSize>>>(DeviceMA, DeviceMB, DeviceMyResultM, RowsNo, ColsNo, ColsNo2, ScatterSize, myid);
		cudaMemcpy( (void *)MyResultMatrix, (void *)DeviceMyResultM, elements * sizeof(float), cudaMemcpyDeviceToHost );
		end_time = MPI_Wtime();
	}        
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(MyResultMatrix,elements, MPI_FLOAT, ResultM, elements, MPI_FLOAT, Root, MPI_COMM_WORLD);//收集到跟进程的ResultM中
	
	//验证
	int valid = 1;
	if (myid == Root){	
		
		CPUResultM = (float *)malloc(RowsNo * ColsNo2 * sizeof(float));
		double start_time_cpu = MPI_Wtime();
		for (int i = 0; i < RowsNo; i++) {
				for (int j = 0; j < ColsNo2; j++) 
				{
					float sum = 0.0;
					for (int k = 0; k < RowsNo2; k++)
						sum = sum + MA[i * ColsNo + k] * MB[k * ColsNo2 + j];
			
					CPUResultM[i * ColsNo2 + j] = sum;
				}
		}
		double end_time_cpu = MPI_Wtime();
		if(verify==1)
		{	
			printf("\n\n-------------------------Validate Result-------------------------\n\n");
			for(Index = 0; Index < ColsNo2 * RowsNo; Index++)
			{
				float a = ResultM[Index];
				float b = CPUResultM[Index];
				if (abs(a,b) >=  0.01 * a)
				{	
					valid = 0;
					printf("Unmatch element in element [%d, %d], gpu_value: %f, cpu_value: %f\n", \
					Index/ColsNo2,Index % ColsNo2, ResultM[Index], CPUResultM[Index]);
				}
			}
			if (valid == 1) {
				printf("Good! Result is valid.\n");
			} else {
				printf("Result is invalid!!!\n");
			}
		}
		printf("\nGPU Work took %f seconds\n\n", end_time - start_time);
		printf("\nCPU Work took %f seconds\n\n", end_time_cpu - start_time_cpu);
	}
	//Root processor printing the resultant vector if print specified
	if(myid == Root && print==1)
	{	
		printResultMatrix_on_csv(ResultM, RowsNo, ColsNo2, "output_row.csv");
		free(MA);
		free(ResultM);
	}
	free(MyMA);
	free(MB);
	free(MyResultMatrix);
	/*//Freeing the device memory
	SAFE( cudaFree( DeviceMA ) );
	SAFE( cudaFree( DeviceMB ) );
	SAFE( cudaFree( DeviceMyResultM ) );*/
	MPI_Finalize();
	return(0);
}
int IntializingMatrixVectors(float **MA, float **MB, float **ResultM, int RowsNo, int ColsNo, int RowsNo2, int ColsNo2)
{
    float *TempMA, *TempVectorB, *TempResultM, *TempMB;
    int Status = 1;  // 初始设为 1，表示初始化成功
    int Index;

    // 分配内存
    TempMA = (float *)malloc(RowsNo * ColsNo * sizeof(float));
    if(TempMA == NULL)
        Status = 0;  // 内存分配失败

    TempMB = (float *)malloc(RowsNo2 * ColsNo2 * sizeof(float));
    if(TempMB == NULL)
        Status = 0;  // 内存分配失败

    TempResultM = (float *)malloc(RowsNo * ColsNo2 * sizeof(float));
    if(TempResultM == NULL)
        Status = 0;  // 内存分配失败

    // 初始化矩阵和向量
    int a = 10;
    for(Index = 0; Index < RowsNo * ColsNo; Index++)
        TempMA[Index] = (float)rand() / (float)(RAND_MAX / a);

    for(Index = 0; Index < RowsNo2 * ColsNo2; Index++)
        TempMB[Index] = (float)rand() / (float)(RAND_MAX / a);

    for(Index = 0; Index < ColsNo2 * RowsNo; Index++)
        TempResultM[Index] = 0.0f;
	// int count = 1;
    // for (Index = 0; Index < RowsNo * ColsNo; Index++) {
    //     TempMA[Index] = (float)count++;
    // }

    // for (Index = 0; Index < RowsNo2 * ColsNo2; Index++) {
    //     TempMB[Index] = (float)count++;
    // }

    // for (Index = 0; Index < ColsNo2 * RowsNo; Index++) {
    //     TempResultM[Index] = 0.0f;
    // }
    // 将临时指针赋值给传入的指针
    *MA = TempMA;
    *MB = TempMB;
    *ResultM = TempResultM;

    return Status;
}

int CheckDevice(int myid)
{
        int DeviceCount, Device;
        struct cudaDeviceProp Properties;

        cudaGetDeviceCount(&DeviceCount);
        if(DeviceCount >= 1)
        {
                cudaGetDevice(&Device);
                cudaGetDeviceProperties(&Properties, Device);
                printf("Processor with  rank %d has the Device by name %s and computation is done on this device \n",myid, Properties.name);
        }
        return(DeviceCount);
}

void printResultMatrix(float *ResultM, int RowsNo, int ColsNo2, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < RowsNo; i++) {
        for (int j = 0; j < ColsNo2; j++) {
            fprintf(file, "%f", ResultM[i * ColsNo2 + j]);
            if (j < ColsNo2 - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}
void printResultMatrix_on_csv(float *ResultVector, int RowsNo, int ColsNo2, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < RowsNo; i++) {
        for (int j = 0; j < ColsNo2; j++) {
            fprintf(file, "%f", ResultVector[i * ColsNo2 + j]);
            if (j < ColsNo2 - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

float abs(float a, float b)
{
	if(a>=b)
		return a-b;
	else 
		return b-a;
}