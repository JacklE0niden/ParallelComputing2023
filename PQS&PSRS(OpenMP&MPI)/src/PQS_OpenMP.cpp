#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

// #define SIZE 500

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

void quickSort_parallel_internal(int arr[], int low, int high, int cutoff) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (high - low < cutoff) {
            quickSort_parallel_internal(arr, low, pi - 1, cutoff);
            quickSort_parallel_internal(arr, pi + 1, high, cutoff);
        } else {
#pragma omp task
            {
                quickSort_parallel_internal(arr, low, pi - 1, cutoff);
            }
#pragma omp task
            {
                quickSort_parallel_internal(arr, pi + 1, high, cutoff);
            }
        }
    }
}

void quickSort_parallel(int arr[], int low, int high, int cutoff) {
#pragma omp parallel
#pragma omp single
    {
        quickSort_parallel_internal(arr, low, high, cutoff);
#pragma omp taskwait
    }
}

void serialQuickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        serialQuickSort(arr, low, pi - 1);
        serialQuickSort(arr, pi + 1, high);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 3){
        fprintf(stderr, "Usage: %s <desired_num_threads> <desire_size_of_array>\n", argv[0]);
        exit(1);
    }
    srand((unsigned int)time(NULL));
    int SIZE = atoi(argv[2]);  // 将第二个命令行参数转换为整数
    int parallelArr[SIZE];
    int serialArr[SIZE];
    printf("                           OpenMP PQS \n\n");
    printf("-------------------------Generate array-------------------------");
    printf("\nArray Size: %d\n\n", SIZE);
    for (int i = 0; i < SIZE; i++) {
        parallelArr[i] = (rand() % 1000) + 1;
        serialArr[i] = parallelArr[i];
    }
    printf("Array Before Sorted:\n");
    printArray(parallelArr, SIZE);
    int n = sizeof(parallelArr) / sizeof(parallelArr[0]);

    int desired_num_threads = atoi(argv[1]);
    omp_set_num_threads(desired_num_threads);
    double start_time = omp_get_wtime();

    int cutoff = 100;
    #pragma omp parallel
    {
        #pragma omp single
        {
            int num_threads = omp_get_num_threads();
            printf("Number of threads in parallel region: %d\n", num_threads);
        }
    }
    quickSort_parallel(parallelArr, 0, n - 1, cutoff);
    double end_time = omp_get_wtime();

    printf("Exucution Completed!\n\n");
    printf("-------------------------Result-------------------------\n\n");
    printf("Sorted array: \n");
    printArray(parallelArr, n);

    printf("\nWork took %f seconds\n\n", end_time - start_time);

    printf("\n\n-------------------------Validate Result-------------------------\n\n");
    double start_time_serial = omp_get_wtime();
    serialQuickSort(serialArr, 0, n - 1);
    double end_time_serial = omp_get_wtime();

    printf("\nSerial Work took %f seconds\n\n", end_time_serial - start_time_serial);
    int valid = 1;
    for (int i = 0; i < SIZE; i++) {
        if (serialArr[i] != parallelArr[i]) {
            printf("Unmatch element in element %d  value: %d\n", i, serialArr[i]);
            valid = 0;
        }
    }

    if (valid == 1) {
        printf("Good! Result is valid.");
    } else {
        printf("Result invalid!!!");
    }

    printf("\n\n\n\n\n");

    return 0;
}