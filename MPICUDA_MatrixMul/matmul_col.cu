#include <stdio.h>
#include <cuda.h>
#include "mpi.h"

#define BLOCKSIZE 16

int IntializingMatrixVectors(float **, float **, float **, int , int , int , int);
int CheckDevice(int );
void printResultMatrix(float *, int , int); 
void printResultMatrix_on_csv(float *, int , int, const char*); 
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

//Kernel that performs Matrix Vector Multiplication
// //用一维的方式存储矩阵
// __global__ void MatrixVectorMultiplication(float *MA, float *MB, float *Solution, int r1, int c1, int r2, int c2, int VectorLength, int ScatterSize, int ThreadDim, int MyRank, int NumberofProcessors)
// {
// 	int tidx = threadIdx.x;// 获取当前线程在块内的索引
// 	// 计算当前线程处理的元素索引
// 	int RowStart = blockIdx.x * blockDim.x * ScatterSize;
// 	int RowEnd = RowStart + ScatterSize;	
// 	// 以下循环用于计算矩阵乘法
// 	for (int i = RowStart + tidx; i < RowEnd; i += blockDim.x) {
// 		// 内循环迭代矩阵的列
// 		for (int j = 0; j < c2; j++) {
// 			float sum = 0.0;// 初始化局部变量 sum 用于存储矩阵乘法的累加结果	
// 			// 执行矩阵乘法的累加步骤，计算矩阵元素相乘的和
// 			for (int k = 0; k < r2; k++)
// 				sum += MA[i * c1 + k] * MB[k * c2 + j];	
// 			// 将累加结果存储在输出矩阵 Solution 中
// 			Solution[i * c2 + j] = sum;
// 		}
// 	}
// 	__syncthreads();
// }
__global__ void MatrixMultiplication(float *MA, float *MB, float *Res, int r1, int ScatterSize, int c2) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < r1 && col < c2) {
        float sum = 0.0;
        for (int k = 0; k < ScatterSize; ++k) {
            sum += MA[row * ScatterSize + k] * MB[k * c2 + col];
        }
        Res[row * c2 + col] = sum;
    }
}

int main(int argc, char **argv)
{
	int myid, numprocs;
	int Root = 0, Index, Status = 1;
	float *MatrixA, *ResultVector, *MatrixB, *ResultMatrix;
	float *MyMatrixA, *MyMatrixB, *MyResultMatrix;//每个进程负责的矩阵和计算结果
	float *DeviceMyMatrixA, *DeviceMyMatrixB, *DeviceMyResultVector, *DeviceVectorB, *DeviceMatrixB, *CPUResultVector;
	int RowsNo, ColsNo, RowsNo2, ColsNo2, VectorSize, ScatterSize, IndexCol, IndexValue, DeviceStatus;
	int bsize;
	int print =0;
	int verify =0;
	double start_time, end_time, start_time_cpu, end_time_cpu;
	//MPI Intialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); //当前进程的rank
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs); //获取进程的总数

	//Checking if valid number of arguements have been passed
	if(argc < 6)
	{
		if(myid == Root)
			printf("Usage:< mpirun >< -n >< Number of processors >< ./Program Name >< Number of Rows of Matrix >< Number of Columns of Matrix >< Rows of Matrix 2 > <Coloumns of Matrix 1>  <-v if verification is required>  <-p if print is required>\n");
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

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&Status, 1, MPI_INT, Root, MPI_COMM_WORLD);
	//状态为0说明分配失败

	//Allocating memory for the Vector by all nodes expect root node
	if(myid != Root)
		MatrixB = (float *)malloc(bsize * sizeof(float));

	Status = IntializingMatrixVectors(&MatrixA, &MatrixB, &ResultVector, RowsNo, ColsNo, RowsNo2, ColsNo2);
	// //Broad casting the Vector to all the nodes from root node
	// MPI_Bcast(MatrixB, bsize, MPI_FLOAT, Root, MPI_COMM_WORLD);
	//计算分割大小以及结果矩阵的大小
	ScatterSize = ColsNo / numprocs; //10
	elements = (RowsNo*ColsNo2);
	MyMatrixB = (float *)malloc(ScatterSize * ColsNo2 * sizeof(float));
	MPI_Scatter(MatrixB, ScatterSize * ColsNo2, MPI_FLOAT, MyMatrixB, ScatterSize * ColsNo2, MPI_FLOAT, Root, MPI_COMM_WORLD);
	
	MyMatrixA = (float *)malloc(ScatterSize * RowsNo * sizeof(float));
	if(MyMatrixA == NULL)
		Status = 0;

	MyResultMatrix = (float *)malloc(elements* sizeof(float)); // 存储结果矩阵
	if(MyResultMatrix == NULL)
		Status = 0;
	
	int *sendcounts = (int *)malloc(numprocs * sizeof(int));
	int *displs = (int *)malloc(numprocs * sizeof(int));
	
	if (myid == 0) {
        // Calculate sendcounts and displs for each process
        for (int i = 0; i < numprocs; i++) {
            sendcounts[i] = ScatterSize;
            displs[i] = i * ScatterSize;
        }
    }
	// 切分，分发
	for(int j = 0; j < RowsNo; j++)
	{
		MPI_Scatterv(MatrixA+j*ColsNo, sendcounts, displs, MPI_FLOAT, MyMatrixA+j*ScatterSize, ScatterSize*numprocs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	
	// MPI_Scatter(MatrixA, ScatterSize * ColsNo, MPI_FLOAT, MyMatrixA, ScatterSize * ColsNo, MPI_FLOAT, Root, MPI_COMM_WORLD);

	printf("Process %d received:\n", myid);
	for (int i = 0; i < RowsNo; i++) {
		for (int j = 0; j < ScatterSize; j++) {
			printf("%f ", MyMatrixA[i * ScatterSize + j]);
		}
		printf("\n");
	}

	DeviceStatus = CheckDevice(myid);

    if(DeviceStatus == 0)
    {
		printf("cuda is fucked away\n");exit(1);
	}
	else // GPU可用，在GPU上运行矩阵乘法
	{
		cudaSetDevice(myid);
		// printf("Unpinned mode\n");
		SAFE( cudaMalloc( (void **)&DeviceMyMatrixA, ScatterSize * RowsNo * sizeof(float) ) );
		SAFE( cudaMalloc( (void **)&DeviceMyMatrixB, ScatterSize * ColsNo2 * sizeof(float) ) );
		SAFE( cudaMalloc( (void **)&DeviceMyResultVector, elements * sizeof(float) ) );
		// 将数据从主机内存同步地传输到设备内存
		cudaMemcpy( (void *)DeviceMyMatrixA, (void *)MyMatrixA, ScatterSize * RowsNo * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( (void *)DeviceMyMatrixB, (void *)MyMatrixB, ScatterSize * ColsNo2 * sizeof(float), cudaMemcpyHostToDevice );
		cudaSetDevice(myid);

		start_time = MPI_Wtime();
		// 定义线程块和网格的大小
   	 	dim3 blockSize(16, 16);  // 16x16 线程块
    	dim3 gridSize((ColsNo2 + blockSize.x - 1) / blockSize.x, (RowsNo + blockSize.y - 1) / blockSize.y);    	// 调用核函数
    	MatrixMultiplication<<<gridSize, blockSize>>>(DeviceMyMatrixA, DeviceMyMatrixB, DeviceMyResultVector, RowsNo, ScatterSize, ColsNo2);
		cudaMemcpy( (void *)MyResultMatrix, (void *)DeviceMyResultVector, elements * sizeof(float), cudaMemcpyDeviceToHost );
			
	}        

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(MyResultMatrix, ResultVector, elements, MPI_FLOAT, MPI_SUM, Root, MPI_COMM_WORLD);
	end_time = MPI_Wtime();

	//验证
	int valid = 1;
	if (myid == Root && verify==1){
		printf("\n\n-------------------------Validate Result-------------------------\n\n");
		CPUResultVector = (float *)malloc(RowsNo * ColsNo2 * sizeof(float));
		start_time_cpu = MPI_Wtime();
		for (int i = 0; i < RowsNo; i++) {
				for (int j = 0; j < ColsNo2; j++) 
				{
					float sum = 0.0;
					for (int k = 0; k < RowsNo2; k++)
						sum = sum + MatrixA[i * ColsNo + k] * MatrixB[k * ColsNo2 + j];
					// A[i, k]*B[k,j]
					CPUResultVector[i * ColsNo2 + j] = sum;
				}
		}
		end_time_cpu = MPI_Wtime();
		for(Index = 0; Index < ColsNo2 * RowsNo; Index++)
		{
			double a = ResultVector[Index];
			double b = CPUResultVector[Index];
			if ( (abs(a,b)) >= (0.01*a) )
			{
				valid = 0;
				printf("Unmatch element in element [%d, %d], gpu_value: %f, cpu_value: %f\n", \
				Index/ColsNo2,Index % ColsNo2, ResultVector[Index], CPUResultVector[Index]);
			}
		}
		printf("\nGPU Work took %f seconds\n\n", end_time - start_time);
		printf("\nCPU Work took %f seconds\n\n", end_time_cpu - start_time_cpu);
		// if (valid == 1) {
		// 	printf("Good! Result is valid.\n");
		// } else {
		// 	printf("Result is invalid!!!\n");
		// }
	}
	
	//Root processor printing the resultant vector if print specified
	if(myid == Root && print==1)
	{	
		printResultMatrix_on_csv(ResultVector, RowsNo, ColsNo2, "output_col.csv");
		// printf("The resultant vector with size %d  is \n",RowsNo*ColsNo2);
		// for(Index = 0; Index < ColsNo2 * RowsNo; Index++)
		// 	printf(" %f \n", ResultVector[Index]);
		// freeing the Vectors allocated by the root node
		free(MatrixA);
		free(ResultVector);
	}

	if(myid == 0)
	{
		if (valid == 1) {
        	printf("Good! Result is valid.\n");
		} else {
			printf("Result is invalid!!!\n");
		}
	}
	free(MyMatrixA);
	free(MatrixB);
	free(MyResultMatrix);
	
	/*//Freeing the device memory
	CUDA_SAFE_CALL( cudaFree( DeviceMyMatrixA ) );
	CUDA_SAFE_CALL( cudaFree( DeviceMatrixB ) );
	CUDA_SAFE_CALL( cudaFree( DeviceMyResultVector ) );*/
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

void printResultMatrix(float *ResultVector, int RowsNo, int ColsNo2) {
    for (int i = 0; i < RowsNo; i++) {
        for (int j = 0; j < ColsNo2; j++) {
            printf("%f", ResultVector[i * ColsNo2 + j]);
            if (j < ColsNo2 - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
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