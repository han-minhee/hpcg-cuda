# -*- Makefile -*-

arch = CUDA_MPI_OMP
setup_file = setup/Make.$(arch)

include $(setup_file)


HPCG_DEPS = src/CG.o \
	    src/CG_ref.o \
	    src/TestCG.o \
	    src/ComputeResidual.o \
	    src/ExchangeHalo.o \
	    src/GenerateGeometry.o \
	    src/GenerateProblem.o \
	    src/GenerateProblem_ref.o \
	    src/CheckProblem.o \
	    src/MixedBaseCounter.o \
	    src/OptimizeProblem.o \
	    src/ReadHpcgDat.o \
	    src/ReportResults.o \
	    src/SetupHalo.o \
	    src/SetupHalo_ref.o \
	    src/TestSymmetry.o \
	    src/TestNorms.o \
	    src/WriteProblem.o \
	    src/YAML_Doc.o \
	    src/YAML_Element.o \
	    src/ComputeDotProduct.o \
	    src/ComputeDotProduct_ref.o \
	    src/mytimer.o \
	    src/ComputeOptimalShapeXYZ.o \
	    src/ComputeSPMV.o \
	    src/ComputeSPMV_ref.o \
	    src/ComputeSYMGS.o \
	    src/ComputeSYMGS_ref.o \
	    src/ComputeWAXPBY.o \
	    src/ComputeWAXPBY_ref.o \
	    src/ComputeMG_ref.o \
	    src/ComputeMG.o \
	    src/ComputeProlongation_ref.o \
	    src/ComputeRestriction_ref.o \
	    src/CheckAspectRatio.o \
	    src/OutputFile.o \
	    src/GenerateCoarseProblem.o \
	    src/init.o \
	    src/finalize.o

CUDA_DEPS = cuda-src/ComputeDotProductInside.o \
			cuda-src/ComputeMGInside.o \
			cuda-src/ComputeProlongationInside.o \
			cuda-src/ComputeResidualInside.o \
			cuda-src/ComputeRestrictionInside.o \
			cuda-src/ComputeSPMVInside.o \
			cuda-src/ComputeWAXPBYInside.o \
			cuda-src/ExchangeHaloInside.o \
			cuda-src/finalizeInside.o \
			cuda-src/GenerateCoarseProblemInside.o \
			cuda-src/GenerateProblemInside.o \
			cuda-src/InitInside.o \
			cuda-src/MultiColoring.o \
			cuda-src/Permute.o \
			cuda-src/SetupHaloInside.o \
			cuda-src/SparseMatrixInside.o \
			cuda-src/TestCGInside.o \
			cuda-src/TestSymmetryInside.o \
			cuda-src/Utils.o \
			cuda-src/VectorInside.o \
			cuda-src/ComputeSYMGSInside.o

# These header files are included in many source files, so we recompile every file if one or more of these header is modified.
PRIMARY_HEADERS = ./src/Geometry.hpp ./src/SparseMatrix.hpp ./src/Vector.hpp ./src/CGData.hpp \
                  ./src/MGData.hpp ./src/hpcg.hpp 
				  # ./src/SparseMatrixOp.hpp

all: bin/xhpcg

bin/xhpcg: src/main.o $(HPCG_DEPS) $(CUDA_DEPS)
	$(LINKER) $(LINKFLAGS) src/main.o $(HPCG_OPTS) $(HPCG_DEPS) $(CUDA_DEPS) $(HPCG_LIBS) $(NVCC_LIBS) -o bin/xhpcg

clean:
	rm -f src/*.o cuda-src/*.o bin/xhpcg

.PHONY: all clean

src/main.o: ./src/main.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/CG.o: ./src/CG.cpp ./src/CG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/CG_ref.o: ./src/CG_ref.cpp ./src/CG_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/TestCG.o: ./src/TestCG.cpp ./src/TestCG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeResidual.o: ./src/ComputeResidual.cpp ./src/ComputeResidual.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ExchangeHalo.o: ./src/ExchangeHalo.cpp ./src/ExchangeHalo.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/GenerateGeometry.o: ./src/GenerateGeometry.cpp ./src/GenerateGeometry.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/GenerateProblem.o: ./src/GenerateProblem.cpp ./src/GenerateProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/GenerateProblem_ref.o: ./src/GenerateProblem_ref.cpp ./src/GenerateProblem_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/CheckProblem.o: ./src/CheckProblem.cpp ./src/CheckProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/MixedBaseCounter.o: ./src/MixedBaseCounter.cpp ./src/MixedBaseCounter.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/OptimizeProblem.o: ./src/OptimizeProblem.cpp ./src/OptimizeProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ReadHpcgDat.o: ./src/ReadHpcgDat.cpp ./src/ReadHpcgDat.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ReportResults.o: ./src/ReportResults.cpp ./src/ReportResults.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/SetupHalo.o: ./src/SetupHalo.cpp ./src/SetupHalo.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/SetupHalo_ref.o: ./src/SetupHalo_ref.cpp ./src/SetupHalo_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/TestSymmetry.o: ./src/TestSymmetry.cpp ./src/TestSymmetry.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/TestNorms.o: ./src/TestNorms.cpp ./src/TestNorms.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/WriteProblem.o: ./src/WriteProblem.cpp ./src/WriteProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/YAML_Doc.o: ./src/YAML_Doc.cpp ./src/YAML_Doc.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/YAML_Element.o: ./src/YAML_Element.cpp ./src/YAML_Element.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeDotProduct.o: ./src/ComputeDotProduct.cpp ./src/ComputeDotProduct.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeDotProduct_ref.o: ./src/ComputeDotProduct_ref.cpp ./src/ComputeDotProduct_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/finalize.o: ./src/finalize.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/init.o: ./src/Init.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/mytimer.o: ./src/mytimer.cpp ./src/mytimer.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeOptimalShapeXYZ.o: ./src/ComputeOptimalShapeXYZ.cpp ./src/ComputeOptimalShapeXYZ.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeSPMV.o: ./src/ComputeSPMV.cpp ./src/ComputeSPMV.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeSPMV_ref.o: ./src/ComputeSPMV_ref.cpp ./src/ComputeSPMV_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeSYMGS.o: ./src/ComputeSYMGS.cpp ./src/ComputeSYMGS.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeSYMGS_ref.o: ./src/ComputeSYMGS_ref.cpp ./src/ComputeSYMGS_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeWAXPBY.o: ./src/ComputeWAXPBY.cpp ./src/ComputeWAXPBY.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeWAXPBY_ref.o: ./src/ComputeWAXPBY_ref.cpp ./src/ComputeWAXPBY_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeMG_ref.o: ./src/ComputeMG_ref.cpp ./src/ComputeMG_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeMG.o: ./src/ComputeMG.cpp ./src/ComputeMG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeProlongation_ref.o: ./src/ComputeProlongation_ref.cpp ./src/ComputeProlongation_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/ComputeRestriction_ref.o: ./src/ComputeRestriction_ref.cpp ./src/ComputeRestriction_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/GenerateCoarseProblem.o: ./src/GenerateCoarseProblem.cpp ./src/GenerateCoarseProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/CheckAspectRatio.o: ./src/CheckAspectRatio.cpp ./src/CheckAspectRatio.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/OutputFile.o: ./src/OutputFile.cpp ./src/OutputFile.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

src/SparseMatrixOp.o: ./src/SparseMatrixOp.cpp ./src/SparseMatrixOp.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

# CUDA Implementation from Here

cuda-src/Utils.o: ./cuda-src/Utils.cu ./cuda-src/Utils.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

cuda-src/MultiColoring.o: ./cuda-src/MultiColoring.cu ./cuda-src/MultiColoring.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

cuda-src/Permute.o: ./cuda-src/Permute.cu ./cuda-src/Permute.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src $< -o $@

cuda-src/SparseMatrixInside.o: ./cuda-src/SparseMatrixInside.cu ./cuda-src/SparseMatrixInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeDotProductInside.o: ./cuda-src/ComputeDotProductInside.cu ./cuda-src/ComputeDotProductInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeMGInside.o: ./cuda-src/ComputeMGInside.cu ./cuda-src/ComputeMGInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeProlongationInside.o: ./cuda-src/ComputeProlongationInside.cu ./cuda-src/ComputeProlongationInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeResidualInside.o: ./cuda-src/ComputeResidualInside.cu ./cuda-src/ComputeResidualInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeRestrictionInside.o: ./cuda-src/ComputeRestrictionInside.cu ./cuda-src/ComputeRestrictionInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeSPMVInside.o: ./cuda-src/ComputeSPMVInside.cu ./cuda-src/ComputeSPMVInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeSYMGSInside.o: ./cuda-src/ComputeSYMGSInside.cu ./cuda-src/ComputeSYMGSInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ComputeWAXPBYInside.o: ./cuda-src/ComputeWAXPBYInside.cu ./cuda-src/ComputeWAXPBYInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/ExchangeHaloInside.o: ./cuda-src/ExchangeHaloInside.cu ./cuda-src/ExchangeHaloInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/finalizeInside.o: ./cuda-src/finalizeInside.cu ./cuda-src/finalizeInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/GenerateCoarseProblemInside.o: ./cuda-src/GenerateCoarseProblemInside.cu ./cuda-src/GenerateCoarseProblemInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/GenerateProblemInside.o: ./cuda-src/GenerateProblemInside.cu ./cuda-src/GenerateProblemInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/InitInside.o: ./cuda-src/InitInside.cu ./cuda-src/InitInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/SetupHaloInside.o: ./cuda-src/SetupHaloInside.cu ./cuda-src/SetupHaloInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/TestCGInside.o: ./cuda-src/TestCGInside.cu ./cuda-src/TestCGInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/TestSymmetryInside.o: ./cuda-src/TestSymmetryInside.cu ./cuda-src/TestSymmetryInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@

cuda-src/VectorInside.o: ./cuda-src/VectorInside.cu ./cuda-src/VectorInside.cuh $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I./src -I./cuda-src $< -o $@
