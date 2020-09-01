./configure CUDA_MPI_OMP
make -j
cd bin
mpirun -n 2 ./xhpcg 104 104 104