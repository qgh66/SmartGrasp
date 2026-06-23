#!/usr/bin/env bash
# ============================================================================
# SmartGrasp macOS 环境一键安装脚本
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo " SmartGrasp macOS 环境安装"
echo "========================================="

# 1. 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 conda，请先安装 Miniforge3："
    echo "   brew install miniforge"
    echo "   或从 https://github.com/conda-forge/miniforge 下载"
    exit 1
fi

echo "✓ conda 已安装: $(conda --version)"

# 2. 创建 conda 环境
ENV_NAME="smartgrasp-mac"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 ${ENV_NAME} 已存在，跳过创建。"
    echo "   如需重建请先运行: conda env remove -n ${ENV_NAME}"
else
    echo ""
    echo "📦 创建 conda 环境: ${ENV_NAME} ..."
    conda env create -f environment_mac.yml
    echo "✓ 环境创建完成"
fi

# 3. 激活环境并安装额外依赖
echo ""
echo "📦 安装 SAM2 + LangSAM ..."
conda run -n ${ENV_NAME} pip install --quiet \
    git+https://github.com/facebookresearch/sam2.git \
    git+https://github.com/luca-medeiros/lang-segment-anything.git \
    2>&1 | tail -5

echo ""
echo "========================================="
echo " ✅ 安装完成！"
echo ""
echo "激活环境："
echo "   conda activate ${ENV_NAME}"
echo ""
echo "验证安装："
echo "   python -c 'import torch; print(\"PyTorch:\", torch.__version__); print(\"MPS available:\", torch.backends.mps.is_available())'"
echo "   python -c 'import lang_sam; print(\"LangSAM: OK\")'"
echo ""
echo "设置环境变量："
echo "   export SMARTGRASP_DATA_DIR=/path/to/FreeGraspData"
echo "   export OPENAI_API_KEY='your-api-key'"
echo "   export OPENAI_BASE_URL='https://your-api-endpoint/v1'"
echo ""
echo "运行感知管线："
echo "   python perception/perception.py --mode vlm --scene-id 184"
echo "   python perception/perception.py --mode gt  --scene-id 184"
echo "========================================="
