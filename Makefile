CUDA_INSTALL_PATH=/usr/local/cuda-12.0
NVCCFLAGS=-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -Xcompiler -fpermissive
LDFLAGS=-lcudart  -lcublas -lcusparse -L $(CUDA_INSTALL_PATH)/lib64 -g 
INCLUDES = -I $(CUDA_INSTALL_PATH)/include
.PHONY :all bicg_cusparse cg_cusparse pre memory_all
all:
	nvcc Mille_cg_ad.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille_cg_ad -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc Mille_bicg_ad.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o Mille_bicg_ad -Xcompiler -fopenmp -O3 -maxrregcount=32
	nvcc bicg_cusparse.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o bicg_cusparse -Xcompiler -fopenmp -O3
	nvcc cg_cusparse.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o cg_cusparse -Xcompiler -fopenmp -O3
	nvcc pre_time.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o pre_time -Xcompiler -fopenmp -O3
	nvcc memory_all.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o memory_all -Xcompiler -fopenmp -O3
bicg_cusparse:
	nvcc bicg_cusparse.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o bicg_cusparse -Xcompiler -fopenmp -O3
cg_cusparse:
	nvcc cg_cusparse.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o cg_cusparse -Xcompiler -fopenmp -O3
pre:
	nvcc pre_time.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o pre_time -Xcompiler -fopenmp -O3
memory_all:
	nvcc memory_all.cu $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o memory_all -Xcompiler -fopenmp -O3
clean:
	rm Mille_cg_ad
	rm Mille_bicg_ad
	rm bicg_cusparse
	rm cg_cusparse
	rm pre_time
	rm memory_all