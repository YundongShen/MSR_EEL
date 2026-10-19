#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=ebd_install
#SBATCH --output=logs/install_%j.out
#SBATCH --error=logs/install_%j.err

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
echo "Node: $(hostname)"
nvidia-smi -L

# Step 1: torch with CUDA 12.1 pre-built wheel
echo "=== Installing torch (CUDA 12.1) ==="
gpu_env/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
gpu_env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Step 2: vllm pre-built wheel
echo "=== Installing vllm ==="
gpu_env/bin/pip install vllm==0.6.6.post1 -q
gpu_env/bin/python -c "import vllm; print(vllm.__version__)"

echo "Done: $(date)"
