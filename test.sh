./configure CUDA_MPI_OMP
make clean
make -j
cd bin
./xhpcg 64 64 64