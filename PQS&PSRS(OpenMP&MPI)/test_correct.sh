#!/bin/bash

# Run PQS_OpenMP
./build/PQS_OpenMP 10 1000 > output/output_PQS_OpenMP.txt

# Run PSRS_MPI
mpirun -n 10 ./build/PSRS_MPI 1000 > output/output_PSRS_MPI.txt

# Run tests
output_PQS_OpenMP=$(./build/PQS_OpenMP 10 1000)
output_PSRS_MPI=$(mpirun -n 10 ./build/PSRS_MPI 1000)

# Check if both outputs contain the expected message
if [[ $output_PQS_OpenMP == *"Good! Result is valid."* && $output_PSRS_MPI == *"Good! Result is valid."* ]]; then
    echo "Test passed: Results match expected output."
else
    echo "Test failed: Results do not match expected output."
fi