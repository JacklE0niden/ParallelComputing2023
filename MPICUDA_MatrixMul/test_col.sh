# mpirun -n 1 ./matvec_col 64 64 64 64 -v -p
# mpirun -n 2 ./matvec_col 64 64 64 64 -v -p
# mpirun -n 4 ./matvec_col 64 64 64 64 -v -p


# mpirun -n 1 ./matvec_col 128 128 128 128 -v -p
# mpirun -n 2 ./matvec_col 128 128 128 128 -v -p
# mpirun -n 4 ./matvec_col 128 128 128 128 -v -p


# mpirun -n 1 ./matvec_col 256 256 256 256 -v -p
# mpirun -n 2 ./matvec_col 256 256 256 256 -v -p
# mpirun -n 4 ./matvec_col 256 256 256 256 -v -p


# mpirun -n 1 ./matvec_col 512 512 512 512 -v -p
mpirun -n 2 ./matvec_col 512 512 512 512 -v -p
# mpirun -n 4 ./matvec_col 512 512 512 512 -v -p


# mpirun -n 1 ./matvec_col 1024 1024 1024 1024 -v -p
# mpirun -n 2 ./matvec_col 1024 1024 1024 1024 -v -p
# mpirun -n 4 ./matvec_col 1024 1024 1024 1024 -v -p


# python3 ./csv_to_xlsx.py