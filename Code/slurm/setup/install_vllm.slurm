#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=install_vllm
#SBATCH --output=logs/install_vllm_%j.out
#SBATCH --error=logs/install_vllm_%j.err

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
echo "Node: $(hostname)"
nvidia-smi -L

echo "=== Installing torch (CUDA 12.1) ==="
env/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "=== Installing vllm ==="
env/bin/pip install vllm==0.6.6.post1 -q
env/bin/python -c "import vllm; print(vllm.__version__)"

echo "Done: $(date)"
