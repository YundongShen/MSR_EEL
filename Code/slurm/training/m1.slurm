#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --job-name=m1
#SBATCH --output=logs/m1_%j.out
#SBATCH --error=logs/m1_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# M1: pure positive-pair InfoNCE, no hard negatives, no tier weighting
python train.py \
    --instances data/processed/instances_full.jsonl \
    --encoder   microsoft/unixcoder-base \
    --epochs    10 \
    --checkpoint-dir checkpoints/m1 \
    --pair-types req_test req_hunk orig_hunk req_orig

echo "Done: $(date)"
