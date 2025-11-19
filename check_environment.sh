#!/bin/bash
# 快速检查环境配置脚本

echo "=========================================="
echo "环境检查"
echo "=========================================="

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查GCC
echo -n "GCC: "
if command -v gcc &> /dev/null; then
    GCC_VER=$(gcc --version | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1)
    if [ "$(printf '%s\n' "11.3.0" "$GCC_VER" | sort -V | head -n1)" = "11.3.0" ]; then
        echo -e "${GREEN}$(gcc --version | head -n1)${NC}"
    else
        echo -e "${RED}$(gcc --version | head -n1) (需要 >= 11.3.0)${NC}"
    fi
else
    echo -e "${RED}未找到${NC}"
fi

# 检查CMake
echo -n "CMake: "
if command -v cmake &> /dev/null; then
    CMAKE_VER=$(cmake --version 2>/dev/null | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1)
    if [ "$(printf '%s\n' "3.26.0" "$CMAKE_VER" | sort -V | head -n1)" = "3.26.0" ]; then
        echo -e "${GREEN}$(cmake --version | head -n1)${NC}"
    else
        echo -e "${RED}$(cmake --version | head -n1) (需要 >= 3.26)${NC}"
    fi
else
    echo -e "${RED}未找到${NC}"
fi

# 检查CUDA
echo -n "CUDA: "
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -n1)
    if [ -n "$CUDA_VER" ]; then
        if [ "$(printf '%s\n' "12.0" "$CUDA_VER" | sort -V | head -n1)" = "12.0" ]; then
            echo -e "${GREEN}$(nvcc --version | grep 'release' | head -n1)${NC}"
        else
            echo -e "${YELLOW}$(nvcc --version | grep 'release' | head -n1) (推荐 12.0)${NC}"
        fi
    else
        echo -e "${YELLOW}nvcc找到但无法获取版本${NC}"
    fi
else
    echo -e "${RED}未找到 (检查CUDA_HOME环境变量)${NC}"
fi

# 检查MPICH
echo -n "MPICH: "
if command -v mpicc &> /dev/null; then
    MPICH_VER=$(mpicc --version 2>/dev/null | grep -oP 'version \K[\d.]+' | head -n1)
    if [ -n "$MPICH_VER" ]; then
        if [ "$(printf '%s\n' "3.4.1" "$MPICH_VER" | sort -V | head -n1)" = "3.4.1" ]; then
            echo -e "${GREEN}$(mpicc --version | head -n1)${NC}"
        else
            echo -e "${YELLOW}$(mpicc --version | head -n1) (需要 >= 3.4.1)${NC}"
        fi
    else
        echo -e "${YELLOW}mpicc找到但无法获取版本${NC}"
    fi
else
    echo -e "${YELLOW}未找到 (可选)${NC}"
fi

# 检查Python
echo -n "Python: "
if command -v python3 &> /dev/null; then
    PYTHON_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -n1)
    if [ "$(printf '%s\n' "3.9" "$PYTHON_VER" | sort -V | head -n1)" = "3.9" ]; then
        echo -e "${GREEN}$(python3 --version)${NC}"
    else
        echo -e "${RED}$(python3 --version) (需要 >= 3.9)${NC}"
    fi
else
    echo -e "${RED}未找到${NC}"
fi

# 检查Python包
echo ""
echo "Python包:"
python3 -c "
try:
    import pandas
    print(f'  ${GREEN}✓${NC} pandas: {pandas.__version__}')
except ImportError:
    print(f'  ${RED}✗${NC} pandas: 未安装')

try:
    import matplotlib
    print(f'  ${GREEN}✓${NC} matplotlib: {matplotlib.__version__}')
except ImportError:
    print(f'  ${RED}✗${NC} matplotlib: 未安装')

try:
    import seaborn
    print(f'  ${GREEN}✓${NC} seaborn: {seaborn.__version__}')
except ImportError:
    print(f'  ${RED}✗${NC} seaborn: 未安装')
" 2>&1 || echo "  Python包检查失败"

echo ""
echo "=========================================="

