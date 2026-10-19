#!/bin/bash
#SBATCH --gpus=A100-SXM4-80GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=ebd_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -e
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
echo "Node: $(hostname)"
nvidia-smi -L

# Use a project-local pip cache to avoid home directory quota limits.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT/.pip_cache}"
mkdir -p "$PIP_CACHE_DIR"

PYTHON=$(which python3.12 2>/dev/null || which python3.11 2>/dev/null || which python3)
echo "Python: $PYTHON ($($PYTHON --version))"

rm -rf gpu_env
$PYTHON -m venv gpu_env
gpu_env/bin/pip install --upgrade pip setuptools wheel -q

echo "=== Installing vllm (includes torch) ==="
gpu_env/bin/pip install vllm 2>&1 | tail -10
gpu_env/bin/python -c "import vllm; print('vllm', vllm.__version__)"
gpu_env/bin/python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo "=== Other deps ==="
gpu_env/bin/pip install openai transformers huggingface_hub \
    datasets pytest pytest-timeout scikit-learn numpy tqdm -q

echo "DONE: $(date)"
