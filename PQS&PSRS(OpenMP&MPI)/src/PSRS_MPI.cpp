#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <cstring> 
#include <ctime>   

// #define SIZE 500

// TODO： 完成非整除的形式

// 1 无序序列的划分及局部排序
// 根据数据块的划分方法 将无序序列划分成p部分 每个处理器对其中的一部分进行串行快速排序 这样每个处理器就会拥有一个局部有序序列。

// 2 正则采样
// 每个处理器从局部有序序列中选取第w,2w,...,(p-1)w共p-1个代表元素。其中w = n/p^2。

// 3 确定主元
// 每个处理器都将自己选取好的代表元素发送给处理器p0。p0对这p段有序序列做多路归并排序 再从这排序后的序列中选取第p-1,2(p-1), ...,(p-1)(p-1)共p-1个元素作为主元。

// 4 分发主元
// p0将这p-1个主元分发给各个处理器。

// 5 局部有序序列划分
// 每个处理器在接收到主元后 根据主元将自己的局部有序序列划分成p段。

// 6 p段有序序列的分发
// 每个处理器将自己的第i段发送给第i个处理器 是处理器i都拥有所有处理器的第i段。

// 7 多路排序
// 每个处理器将上一步得到的p段有序序列做多路归并。

// 经过这7步后 一次将每个处理器的数据取出 这些数据是有序的。

int cmp(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// 归并排序的合并操作
void merge(int a[], int p, int q, int r) {
    int n1 = q - p + 1;
    int n2 = r - q;
    int *b = (int *)malloc((n1 + 1) * sizeof(int));
    int *c = (int *)malloc((n2 + 1) * sizeof(int));
    memcpy(b, a + p, n1 * sizeof(int));
    memcpy(c, a + q + 1, n2 * sizeof(int));
    b[n1] = INT32_MAX;
    c[n2] = INT32_MAX;
    int i = 0, j = 0;
    for (int k = p; k <= r; k++) {
        if (b[i] <= c[j]) {
            a[k] = b[i];
            i++;
        } else {
            a[k] = c[j];
            j++;
        }
    }
    free(b);
    free(c);
}

// 归并排序
void mergesort(int a[], int p, int r, int pos[], int size[]) {
    if (p < r) {
        int q = (p + r) / 2;
        mergesort(a, p, q, pos, size);
        mergesort(a, q + 1, r, pos, size);
        merge(a, pos[p], pos[q] + size[q] - 1, pos[r] + size[r] - 1);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

void swap(int* a, int* b) {

    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void serialQuickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        serialQuickSort(arr, low, pi - 1);
        serialQuickSort(arr, pi + 1, high);
    }
}

void serialSRS(int arr[], int SIZE)
{
    
}

int main(int argc, char *argv[]) {
    if (argc != 2){
        // Print an error message to stderr
        fprintf(stderr, "Usage: %s <desire_size_of_array>\n", argv[0]);
        // Return an error code and exit the program
        exit(1);
    }

    int SIZE = atoi(argv[1]);  // 将第二个命令行参数转换为整数
    srand((unsigned int)time(NULL));

    int numprocs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    // printf("myid:%d numprocs:%d\n", myid, numprocs);

    int parallelArr[SIZE];
    int serialArr[SIZE];

    if (myid == 0) {
        printf("                            MPI PSRS \n\n");

        printf("-------------------------Generate array-------------------------");
        printf("\nArray Size: %d\n\n", SIZE);
    }
    for (int i = 0; i < SIZE; i++) {
        parallelArr[i] = (rand() % 1000) + 1;
        serialArr[i] = parallelArr[i];
    }
    int n = sizeof(parallelArr) / sizeof(parallelArr[0]);
    // printf("n:%d\n",n);
    if (myid == 0) {
        printf("Array Before Sorted:\n");
        printArray(parallelArr, SIZE);
    }
    //加入起始计时点
    double start_time = MPI_Wtime();

    // 阶段1：无序序列的划分及局部排序
    // Scatter data to all processes
    int localSize = SIZE / numprocs;
    int *localArr = (int *)malloc(sizeof(int) * localSize);
    MPI_Scatter(parallelArr, localSize, MPI_INT, localArr, localSize, MPI_INT, 0, MPI_COMM_WORLD);
    qsort(localArr, localSize, sizeof(int), cmp);
    MPI_Gather(localArr, localSize, MPI_INT, parallelArr, localSize, MPI_INT, 0, MPI_COMM_WORLD);

    // 阶段2：正则采样
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(parallelArr, SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    int *samples = (int*)malloc(numprocs*numprocs*sizeof(int));
    for(int i = 0; i < numprocs; i++){
        // printf("i:%d, index: %d, value:%d\n",i,n/numprocs*myid+n*i/(numprocs*numprocs),parallelArr[n/numprocs*myid+n*i/(numprocs*numprocs)]);
        samples[numprocs*myid + i] = parallelArr[n/numprocs*myid+n*i/(numprocs*numprocs)];
    }
    for(int i = 0; i < numprocs; i++){
        MPI_Bcast(samples+numprocs*i, numprocs*sizeof(int), MPI_BYTE, i, MPI_COMM_WORLD);
    }
    //阶段3 采样排序、确定主元
    //先对采样数组进行排序；
    if(myid == 0)
        qsort(samples, numprocs*numprocs, sizeof(int), cmp);
    // 选取主元
    int *main_val = (int*)malloc((numprocs - 1)*sizeof(int));
    if(myid == 0){
        for(int i = 0; i < numprocs - 1; i++){
            main_val[i] = samples[numprocs*(i + 1)];
        }
    }

    //阶段4 分发这p-1个主元
    MPI_Bcast(main_val, (numprocs-1)*sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);

    //阶段5 根据p-1个主元将有序序列划分成p段
    //每个线程都有一个这个数组，只需要记录每个线程的块大小数组即可，这也就是通讯的量
    //blocksizes[i]表示 第i//numproc号线程 需要给到 第i%numproc号线程 的元素个数
    int *blocksize = (int*)malloc(numprocs*sizeof(int));
    memset(blocksize, 0, numprocs*sizeof(int));
    int index = 0;
    for(int i = 0; i < n/numprocs; i++){
        while(parallelArr[n/numprocs*myid+i] > main_val[index]){ // 很巧妙的一个设计
            index += 1;
            if(index == numprocs - 1){
                blocksize[numprocs - 1] = n/numprocs - i;
                break;
            }
        }
        if(index == numprocs - 1){
            blocksize[numprocs - 1] = n/numprocs - i;
            break;
        }
        blocksize[index]++;
    }
    int *blocksizes = (int*)malloc(numprocs*numprocs*sizeof(int));
    for(int i = 0; i < numprocs; i++){
        blocksizes[numprocs*myid + i] = blocksize[i];
    }
    for(int i = 0; i < numprocs; i++){
        MPI_Bcast(blocksizes+numprocs*i, numprocs*sizeof(int), MPI_BYTE, i, MPI_COMM_WORLD);
    }
    //阶段6 全局交换
    //blocksizes[i]表示 第i//numproc号进程 需要给到 第i%numproc号进程 的元素个数
    // 各进程统计要接收的块的大小
    int* recv_size = (int*)malloc(numprocs*sizeof(int));
    MPI_Alltoall(blocksize, 1, MPI_INT, recv_size, 1, MPI_INT, MPI_COMM_WORLD);
    int totalSize = 0;
    for(int i = 0; i < numprocs; i++){
        totalSize += blocksizes[myid + i*numprocs];
    }
    // 全局交换
    int *array_exchanged = (int*)malloc(totalSize*sizeof(int));
    int *sendPos = (int*)malloc(numprocs*sizeof(int));
    int *recvPos = (int*)malloc(numprocs*sizeof(int));
    sendPos[0] = 0;
    recvPos[0] = 0;    
    for(int i = 1; i < numprocs; i++){
        sendPos[i] = sendPos[i - 1] + blocksize[i - 1];
        recvPos[i] = recvPos[i - 1] + recv_size[i - 1];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Alltoallv(parallelArr+n/numprocs*myid, blocksize, sendPos, MPI_INT, array_exchanged, recv_size, recvPos, MPI_INT, MPI_COMM_WORLD);

    //阶段7 归并排序 发送到进程0
    mergesort(array_exchanged, 0, numprocs - 1, recvPos, recv_size);
    //下面的数组存放每个进程的totalsize大小，方便确定后面接受的位置
    int *listSize = (int*)malloc(numprocs*sizeof(int));
    MPI_Gather(&totalSize, 1, MPI_INT, listSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myid == 0){
        recvPos[0] = 0;
        for(int i = 1; i < numprocs; i++){
            recvPos[i] = recvPos[i - 1] + listSize[i - 1];
        }
    }
    MPI_Gatherv(array_exchanged, totalSize, MPI_INT, parallelArr, listSize, recvPos, MPI_INT, 0, MPI_COMM_WORLD);
    //阶段7 成果展示
    if (myid != 0) {
        MPI_Finalize();  // 在0号线程中调用MPI_Finalize
        return 0;       
    }
    double end_time = MPI_Wtime();
    if (myid == 0) {
        printf("Number of threads: %d\n", numprocs);
        printf("Exucution Completed!\n\n");
        printf("-------------------------Result-------------------------\n\n");
        printf("Sorted array: \n");
        // printf("Array Aftor sort and merge epoch 7:\n");
        printArray(parallelArr, SIZE);
        printf("\nWork took %f seconds\n\n", end_time - start_time);
    }

    // Clean up
    free(localArr);
    free(samples);
    free(blocksize);
    free(blocksizes);
    free(recv_size);
    free(array_exchanged);
    free(sendPos);
    free(recvPos);
    free(listSize);


    if (myid == 0) {
    // 仅在 rank 0 执行的代码
        double start_time_serial = MPI_Wtime();
        serialQuickSort(serialArr, 0, n - 1);
        double end_time_serial = MPI_Wtime();
        printf("\nSerial Work took %f seconds\n\n", end_time_serial - start_time_serial);
    }
    
    printf("\n\n-------------------------Validate Result-------------------------\n\n");

    int valid = 1;
    for (int i = 0; i < SIZE; i++) {
        if (serialArr[i] != parallelArr[i]) {
            printf("Unmatch element in element %d  value: %d\n", i, serialArr[i]);
            valid = 0;
        }
    }

    if (valid == 1) {
        printf("Good! Result is valid.\n");
    } else {
        printf("Result is invalid!!!\n");
    }

    printf("\n\n\n\n\n");
    MPI_Finalize();
    return 0;
}

