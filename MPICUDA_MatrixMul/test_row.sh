# mpirun -n 1 ./matvec_row 64 64 64 64 -v -p
# mpirun -n 2 ./matvec_row 64 64 64 64 -v -p
# mpirun -n 4 ./matvec_row 64 64 64 64 -v -p
# mpirun -n 8 ./matvec_row 64 64 64 64 -v -p

# mpirun -n 1 ./matvec_row 128 128 128 128 -v -p
# mpirun -n 2 ./matvec_row 128 128 128 128 -v -p
# mpirun -n 4 ./matvec_row 128 128 128 128 -v -p
# mpirun -n 8 ./matvec_row 128 128 128 128 -v -p

# mpirun -n 1 ./matvec_row 256 256 256 256 -v -p
# mpirun -n 2 ./matvec_row 256 256 256 256 -v -p
# mpirun -n 4 ./matvec_row 256 256 256 256 -v -p
# mpirun -n 8 ./matvec_row 256 256 256 256 -v -p

# mpirun -n 1 ./matvec_row 512 512 512 512 -v -p
mpirun -n 2 ./matvec_row 512 512 512 512 -v -p
# mpirun -n 4 ./matvec_row 512 512 512 512 -v -p
# mpirun -n 8 ./matvec_row 512 512 512 512 -v -p

# mpirun -n 1 ./matvec_row 1024 1024 1024 1024 -v -p
# mpirun -n 2 ./matvec_row 1024 1024 1024 1024 -v -p
# mpirun -n 4 ./matvec_row 1024 1024 1024 1024 -v -p
# mpirun -n 8 ./matvec_row 1024 1024 1024 1024 -v -p

# mpirun -n 1 ./matvec_row 2048 2048 2048 2048 -v -p
# mpirun -n 2 ./matvec_row 2048 2048 2048 2048 -v -p
# mpirun -n 4 ./matvec_row 2048 2048 2048 2048 -v -p
# mpirun -n 8 ./matvec_row 2048 2048 2048 2048 -v -p

# python3 ./csv_to_xlsx.py


