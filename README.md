# Mille-feuille: High Performance CG and BiCGSTAB Solver

## Overview

Mille-feuille is an efficient high-performance solver for Conjugate Gradient (CG) and BiConjugate Gradient Stabilized (BiCGSTAB) methods that implements novel techniques for sparse matrix storage and computation on GPU architectures.

## Main Contributions

- **C1**: Develops a tiled sparse format to store sparse matrices in tiles with different initial precisions
- **C2**: Leverages atomic operations to enable the entire solving procedure within a single GPU kernel
- **C3**: Enables tile-wise on-chip dynamic precision conversion at runtime within the single kernel

## Artifact Availability

**Computational Artifact**: [https://doi.org/10.5281/zenodo.12589363](https://doi.org/10.5281/zenodo.12589363)

## Performance Results

Experimental results demonstrate that Mille-feuille significantly outperforms existing solvers:

### CG Solver Performance
- **3.03x/2.68x** faster than cuSPARSE/hipSPARSE
- **5.37x** faster than PETSc  
- **4.36x** faster than Ginkgo

### BiCGSTAB Solver Performance
- **2.65x/2.32x** faster than cuSPARSE/hipSPARSE
- **3.57x** faster than PETSc
- **3.78x** faster than Ginkgo

## System Requirements

### Hardware
- **GPU**: NVIDIA A100 GPU (PCIe, 80GB, 1.94TB/s)
- **Disk Space**: At least 60 GB for experiment datasets

### Software Dependencies
- Mpich v3.4.1 or above
- NVIDIA CUDA Toolkit v12.0
- cuSPARSE v12.0
- cuBLAS v12.0
- GCC v11.3.0 or above
- Python 3.9 or above
- CMake v3.26 or above
- Python packages: Pandas v2.2.2, matplotlib v3.9.0, seaborn v0.13.2

## Environment setup

refer to `./环境配置说明.md`：

``` bash
# 0. Download and Extract
unzip Mille-feuille.zip
cd Mille-feuille
# 1. 创建环境
bash ./setup_environment.sh
# 2. 配置环境
source ~/.mille_feuille_env.sh
# 3. 检查环境
bash ./check_environment.sh
```

## Installation & Setup

```bash
# 1. Compile Mille-feuille
# If needed, update CUDA path in Mille-feuille/Makefile: CUDA_INSTALL_PATH = /Your/CUDA/path
make

# 2. Dataset Preparation
# Full dataset (~22GB, ~1 hour): 230 matrices for CG + 686 matrices for BiCGSTAB
python3 matrix.py

# OR Simplified dataset (~45 minutes): 570 matrices
# python3 matrix_simple.py

# 3. Comparative Methods Setup

# PETSc
unzip petsc-main.zip
cd ./petsc-main/
bash compile_PETSc.sh
cd ..

# Ginkgo
unzip ginkgo-develop.zip
cd ./ginkgo-develop/
bash compile_Ginkgo.sh
cd ..
```


## Execution
``` bash
# Performance Tests (5+ hours)
bash test_performance.sh

# Memory Tests (2+ hours) 
bash test_memory.sh

# Preprocessing Tests (1+ hour)
bash test_preprocess.sh

# Generate All Figures
bash all_figures.sh
```

## Expected Outputs
- test_performance.sh: Figures 8, 9, 10, 11 (speedup comparisons, precision ratios)

- test_memory.sh: Figure 12 (memory cost comparison)

- test_preprocess.sh: Figure 13 (cost breakdown)

- all_figures.sh: All figures (8-13) from the paper

## Expected Reproduction Time

- Dataset Download: 1 hour

- Compilation: 1 hour

- Computation: 6 hours

- Total: ~8 hours

## Input Parameters

- Matrix Format: Matrix Market format (*.mtx)

- Dataset Size: 230 matrices for CG, 686 matrices for BiCGSTAB

- Repetitions: 100 iterations per test

