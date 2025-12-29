#!/bin/bash

# RunPod 完整环境设置脚本
# 功能：设置 Git、安装 uv、安装依赖、配置 GitHub
# 使用方法：
#   curl -fsSL https://raw.githubusercontent.com/SecondGithub12138/cs336-assignment2-systems/main/setup_runpod.sh | bash
# 或者：
#   git clone https://github.com/SecondGithub12138/cs336-assignment2-systems.git
#   cd cs336-assignment2-systems
#   bash setup_runpod.sh

set -e  # 遇到错误立即退出

echo "================================"
echo "RunPod 完整环境设置"
echo "================================"

# ========================================
# 1. 检查 GPU
# ========================================
echo -e "\n[1/8] 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "⚠️  未检测到 GPU，将使用 CPU 模式"
fi

# ========================================
# 2. 配置 Git
# ========================================
echo -e "\n[2/8] 配置 Git..."

# Git 提交者信息（显示在 commit 记录中）
GIT_USER_NAME="Sean from RunPod"
GIT_USER_EMAIL="chenzhuofusean@gmail.com"

# 检查是否已配置
CURRENT_NAME=$(git config --global user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [ -n "$CURRENT_NAME" ]; then
    echo "Git 用户名已配置: $CURRENT_NAME"
else
    git config --global user.name "$GIT_USER_NAME"
    echo "✅ 设置 Git 用户名: $GIT_USER_NAME"
fi

if [ -n "$CURRENT_EMAIL" ]; then
    echo "Git 邮箱已配置: $CURRENT_EMAIL"
else
    git config --global user.email "$GIT_USER_EMAIL"
    echo "✅ 设置 Git 邮箱: $GIT_USER_EMAIL"
fi

# 设置默认分支名称
git config --global init.defaultBranch main

# 配置 credential helper（避免每次都输入密码）
git config --global credential.helper store

echo "✅ Git 配置完成"

# ========================================
# 3. 设置 SSH 密钥（可选）
# ========================================
echo -e "\n[3/8] 检查 SSH 密钥..."

if [ -f "$HOME/.ssh/id_ed25519" ]; then
    echo "SSH 密钥已存在"
    echo "公钥:"
    cat "$HOME/.ssh/id_ed25519.pub"
    echo ""
    echo "💡 如需 SSH 连接 GitHub: https://github.com/settings/keys"
else
    echo "未找到 SSH 密钥（可使用 HTTPS 方式推送）"
fi

# ========================================
# 4. 确保在正确的目录
# ========================================
echo -e "\n[4/8] 确认项目目录..."

if [ ! -f "pyproject.toml" ]; then
    if [ -d "cs336-assignment2-systems" ]; then
        cd cs336-assignment2-systems
        echo "✅ 进入: cs336-assignment2-systems"
    else
        echo "❌ 错误：未找到项目目录"
        echo "请先运行: git clone https://github.com/SecondGithub12138/cs336-assignment2-systems.git"
        exit 1
    fi
else
    echo "✅ 已在项目目录"
fi

echo "当前: $(pwd)"

# ========================================
# 5. 安装 uv
# ========================================
echo -e "\n[5/8] 安装 uv..."

if command -v uv &> /dev/null; then
    echo "uv 已安装: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
        echo "✅ uv -> $HOME/.cargo/bin"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo "✅ uv -> $HOME/.local/bin"
    fi
fi

UV_PATH=$(which uv 2>/dev/null || echo "$HOME/.cargo/bin/uv")

# ========================================
# 6. 安装依赖
# ========================================
echo -e "\n[6/8] 安装依赖..."
$UV_PATH sync

# ========================================
# 7. 测试环境
# ========================================
echo -e "\n[7/8] 测试环境..."
$UV_PATH run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# ========================================
# 8. 完成
# ========================================
echo -e "\n[8/8] ✅ 完成！耗时: ${SECONDS}s"
echo "================================"
echo ""
echo "📋 快速开始："
echo ""
echo "  # 运行 benchmark"
echo "  bash run_all_benchmarks.sh"
echo ""
echo "  # Git 操作"
echo "  git status"
echo "  git add ."
echo "  git commit -m 'Results from RunPod'"
echo "  git push  # 首次需要 GitHub Token"
echo ""
echo "  # 一键运行+关机"
echo "  bash run_all_benchmarks.sh && sudo poweroff"
echo ""
echo "💡 GitHub Token: https://github.com/settings/tokens"
echo "   权限: repo"
echo "   用户名: SecondGithub12138"
echo ""
echo "⚠️  完成后停止 Pod！"
echo "================================"
