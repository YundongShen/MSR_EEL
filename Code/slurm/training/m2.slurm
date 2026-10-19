#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --job-name=m2
#SBATCH --output=logs/m2_%j.out
#SBATCH --error=logs/m2_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# M2: M1 + Tier-3 same-instance hard negatives
python train.py \
    --instances data/processed/instances_full.jsonl \
    --encoder   microsoft/unixcoder-base \
    --epochs    10 \
    --checkpoint-dir checkpoints/m2 \
    --pair-types req_test req_hunk orig_hunk req_orig \
    --tier3     data/cache/tier3_hunks.jsonl

echo "Done: $(date)"
