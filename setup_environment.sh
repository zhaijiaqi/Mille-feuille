#!/bin/bash
# 环境配置脚本 - 根据软件版本需求配置开发环境
# 
# 需求软件版本：
# - Mpich v3.4.1 or above
# - NVIDIA CUDA Toolkit v12.0
# - cuSPARSE v12.0
# - cuBLAS v12.0
# - GCC v11.3.0 or above
# - Python 3.9 or above
# - cmake v3.26 or above
# - Pandas v2.2.2
# - matplotlib v3.9.0
# - seaborn v0.13.2

# 注意：不使用 set -e，因为某些检测命令（如which）可能返回非零退出码，但不应该导致脚本退出

echo "=========================================="
echo "环境配置脚本开始执行"
echo "=========================================="

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_version() {
    local name=$1
    local current=$2
    local required=$3
    local comparison=$4  # "ge" for >=, "eq" for ==
    
    if [ "$comparison" = "ge" ]; then
        if [ "$(printf '%s\n' "$required" "$current" | sort -V | head -n1)" = "$required" ]; then
            echo -e "${GREEN}✓${NC} $name: $current (>= $required)"
            return 0
        else
            echo -e "${RED}✗${NC} $name: $current (需要 >= $required)"
            return 1
        fi
    elif [ "$comparison" = "eq" ]; then
        if [ "$current" = "$required" ]; then
            echo -e "${GREEN}✓${NC} $name: $current (== $required)"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} $name: $current (推荐 == $required，但可能兼容)"
            return 0
        fi
    fi
}

# 1. 配置模块系统
echo ""
echo "1. 配置模块系统..."
if command -v module &> /dev/null; then
    echo "模块系统已找到"
    # 初始化模块系统（如果需要）
    if [ -z "$MODULEPATH" ]; then
        source /etc/profile.d/modules.sh 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}警告: 未找到module命令，将尝试直接使用系统软件${NC}"
fi

# 2. 配置GCC 11.3.0+
echo ""
echo "2. 配置GCC编译器..."
module unload compilers/gcc
if command -v module &> /dev/null; then
    if module avail compilers/gcc/11.3.0 &> /dev/null; then
        echo "加载GCC 11.3.0模块..."
        module load compilers/gcc/11.3.0 2>/dev/null || true
    elif module avail gcc/11.3.0 &> /dev/null; then
        echo "加载GCC 11.3.0模块..."
        module load gcc/11.3.0 2>/dev/null || true
    fi
fi

GCC_VERSION=$(gcc --version | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1)
if check_version "GCC" "$GCC_VERSION" "11.3.0" "ge"; then
    export CC=gcc
    export CXX=g++
else
    echo -e "${RED}错误: GCC版本不符合要求${NC}"
    exit 1
fi

# 3. 配置cmake 3.26+
echo ""
echo "3. 配置CMake..."
if command -v module &> /dev/null; then
    if module avail cmake/3.26.3 &> /dev/null; then
        echo "加载cmake 3.26.3模块..."
        module load cmake/3.26.3 2>/dev/null || true
    elif module avail cmake/3.31.6 &> /dev/null; then
        echo "加载cmake 3.31.6模块..."
        module load cmake/3.31.6 2>/dev/null || true
    fi
fi

CMAKE_VERSION=$(cmake --version 2>/dev/null | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1)
if check_version "CMake" "$CMAKE_VERSION" "3.26.0" "ge"; then
    echo "CMake配置成功"
else
    echo -e "${RED}错误: CMake版本不符合要求${NC}"
    exit 1
fi

# 4. 配置LLVM/clangd（修复 clangd 10.0.1 的 bug）
echo ""
echo "4. 配置LLVM/clangd..."
if command -v module &> /dev/null; then
    if module avail llvm/clang_llvm-11.1.0 &> /dev/null; then
        echo "加载LLVM 11.1.0模块（修复clangd错误）..."
        module load llvm/clang_llvm-11.1.0 2>/dev/null || true
        if command -v clangd &> /dev/null && ! clangd --version 2>&1 | grep -q "CommandLine Error"; then
            echo -e "${GREEN}✓${NC} clangd配置成功: $(clangd --version 2>&1 | head -n1)"
        fi
    fi
fi

# 5. 配置CUDA 12.0
echo ""
echo "5. 配置CUDA Toolkit..."

# 自动检测CUDA路径
DETECTED_CUDA_PATH=""
# 可能的CUDA路径列表（按优先级排序）
POSSIBLE_CUDA_PATHS=(
    "/usr/local/cuda-12.0"
    "/usr/local/cuda"
)

# 从环境变量中获取
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    DETECTED_CUDA_PATH="$CUDA_HOME"
    echo "从CUDA_HOME环境变量找到CUDA: $DETECTED_CUDA_PATH"
else
    # 尝试从PATH中找到nvcc
    NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "$NVCC_PATH" ]; then
        DETECTED_CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
        echo "从PATH中的nvcc找到CUDA: $DETECTED_CUDA_PATH"
    else
        # 检查常见路径
        for path in "${POSSIBLE_CUDA_PATHS[@]}"; do
            if [ -n "$path" ] && [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
                DETECTED_CUDA_PATH="$path"
                echo "找到CUDA安装路径: $DETECTED_CUDA_PATH"
                break
            fi
        done
    fi
fi

# 如果找到了CUDA路径
if [ -n "$DETECTED_CUDA_PATH" ] && [ -d "$DETECTED_CUDA_PATH" ]; then
    export CUDA_HOME="$DETECTED_CUDA_PATH"
    export PATH="$DETECTED_CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$DETECTED_CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    
    # 检查nvcc
    if [ -f "$DETECTED_CUDA_PATH/bin/nvcc" ]; then
        CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -n1)
        if [ -n "$CUDA_VERSION" ]; then
            if [ "$(printf '%s\n' "12.0" "$CUDA_VERSION" | sort -V | head -n1)" = "12.0" ]; then
                echo -e "${GREEN}✓${NC} CUDA Toolkit: $CUDA_VERSION (>= 12.0)"
            else
                echo -e "${YELLOW}⚠${NC} CUDA Toolkit: $CUDA_VERSION (推荐 12.0，但可能兼容)"
            fi
        fi
        
        # 检查cuSPARSE和cuBLAS
        if [ -f "$DETECTED_CUDA_PATH/lib64/libcusparse.so" ] || [ -f "$DETECTED_CUDA_PATH/lib64/libcusparse.so.12" ] || [ -f "$DETECTED_CUDA_PATH/lib64/libcusparse.so.11" ]; then
            echo -e "${GREEN}✓${NC} cuSPARSE库已找到"
        else
            echo -e "${YELLOW}⚠${NC} cuSPARSE库未找到，但CUDA通常包含它"
        fi
        
        if [ -f "$DETECTED_CUDA_PATH/lib64/libcublas.so" ] || [ -f "$DETECTED_CUDA_PATH/lib64/libcublas.so.12" ] || [ -f "$DETECTED_CUDA_PATH/lib64/libcublas.so.11" ]; then
            echo -e "${GREEN}✓${NC} cuBLAS库已找到"
        else
            echo -e "${YELLOW}⚠${NC} cuBLAS库未找到，但CUDA通常包含它"
        fi
    else
        echo -e "${YELLOW}⚠${NC} CUDA路径存在但nvcc未找到，请检查CUDA安装"
    fi
else
    echo -e "${YELLOW}⚠${NC} 未找到CUDA安装路径"
    echo "已检查以下路径:"
    for path in "${POSSIBLE_CUDA_PATHS[@]}"; do
        if [ -n "$path" ]; then
            echo "  - $path"
        fi
    done
    echo ""
    echo "如果CUDA已安装在其他位置，请设置环境变量:"
    echo "  export CUDA_HOME=/path/to/cuda"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo -e "${YELLOW}继续执行，但CUDA功能可能不可用${NC}"
    # 不退出，允许继续执行其他配置
fi

# 6. 配置MPICH 3.4.1+
echo ""
echo "6. 配置MPICH..."
if command -v mpicc &> /dev/null; then
    MPICH_VERSION=$(mpicc --version 2>/dev/null | grep -oP 'version \K[\d.]+' | head -n1)
    if [ -n "$MPICH_VERSION" ]; then
        if check_version "MPICH" "$MPICH_VERSION" "3.4.1" "ge"; then
            echo "MPICH配置成功"
        else
            echo -e "${YELLOW}警告: MPICH版本 $MPICH_VERSION 低于要求，但继续执行${NC}"
        fi
    fi
else
    echo -e "${YELLOW}警告: MPICH未找到，如果需要MPI功能，请安装MPICH 3.4.1+${NC}"
    echo "可以通过以下方式安装："
    echo "  - 使用module系统: module avail mpich"
    echo "  - 或从源码编译安装"
fi

# 6. 检查Python版本
echo ""
echo "6. 检查Python版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -n1)
if check_version "Python" "$PYTHON_VERSION" "3.9" "ge"; then
    echo "Python版本符合要求"
else
    echo -e "${RED}错误: Python版本不符合要求${NC}"
    exit 1
fi

# 7. 创建conda环境
echo ""
echo "7. 激活conda环境 'mille'..."

if conda info --envs | grep -q '^mille\s'; then
    echo "已检测到mille环境，激活..."
else
    echo "未检测到mille环境，创建中..."
    conda create -n mille python=3.9 -y
fi

# 需要在脚本中用 source 激活（影响后续指令）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mille

pip install -r requirements.txt


# 验证Python包
echo ""
echo "验证Python包安装..."
python3 << 'PYTHON_EOF'
import sys
try:
    import pandas
    print(f'✓ pandas: {pandas.__version__}')
except ImportError:
    print('✗ pandas: 未安装')
    sys.exit(1)

try:
    import matplotlib
    print(f'✓ matplotlib: {matplotlib.__version__}')
except ImportError:
    print('✗ matplotlib: 未安装')
    sys.exit(1)

try:
    import seaborn
    print(f'✓ seaborn: {seaborn.__version__}')
except ImportError:
    print('✗ seaborn: 未安装')
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}所有Python包验证成功${NC}"
else
    echo -e "${YELLOW}警告: 部分Python包验证失败，请检查安装${NC}"
fi

# 8. 生成环境配置文件
echo ""
echo "8. 生成环境配置文件..."
ENV_FILE="$HOME/.mille_feuille_env.sh"
cat > "$ENV_FILE" << 'EOF'
# Mille-feuille项目环境配置
# 使用方式: source ~/.mille_feuille_env.sh

# 加载模块（如果可用）
module unload compilers/gcc
if command -v module &> /dev/null; then
    module load compilers/gcc/11.3.0 2>/dev/null || module load gcc/11.3.0 2>/dev/null || true
    module load cmake/3.26.3 2>/dev/null || module load cmake/3.31.6 2>/dev/null || true
    # 加载LLVM 11.1.0以修复clangd错误
    module load llvm/clang_llvm-11.1.0 2>/dev/null || true
fi

# CUDA配置（自动检测）
# 如果CUDA_HOME未设置，脚本会尝试自动检测
if [ -z "$CUDA_HOME" ]; then
    # 尝试从PATH中找到nvcc
    NVCC_PATH=$(which nvcc 2>/dev/null)
    if [ -n "$NVCC_PATH" ]; then
        export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
    elif [ -d "/usr/local/cuda-12.0" ]; then
        export CUDA_HOME=/usr/local/cuda-12.0
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    fi
fi

if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# 编译器
export CC=gcc
export CXX=g++

# Python路径（如果使用--user安装）
export PATH=$HOME/.local/bin:$PATH

# 激活conda环境 'mille'
conda activate mille
EOF

echo -e "${GREEN}✓${NC} 环境配置文件已生成: $ENV_FILE"
echo "  使用方式: source $ENV_FILE"

# 9. 总结
echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "当前环境配置："
echo "  GCC: $(gcc --version | head -n1)"
echo "  CMake: $(cmake --version | head -n1)"
if command -v nvcc &> /dev/null; then
    echo "  CUDA: $(nvcc --version | grep 'release' | head -n1)"
fi
echo "  Python: $(python3 --version)"
if command -v mpicc &> /dev/null; then
    echo "  MPICH: $(mpicc --version | head -n1)"
fi
echo ""
echo "下次使用时，请运行:"
echo "  source $ENV_FILE"
echo ""

