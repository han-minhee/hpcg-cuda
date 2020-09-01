./configure CUDA_MPI_OMP
make -j
cd bin
mpirun -n 2 ./xhpcg 64 64 64