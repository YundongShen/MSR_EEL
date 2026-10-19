#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --job-name=m4
#SBATCH --output=logs/m4_%j.out
#SBATCH --error=logs/m4_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# M4: M3 + Tier-aware loss weighting (Tier-1 w=1.0, Tier-2 w=0.67)
python train.py \
    --instances data/processed/instances_full.jsonl \
    --encoder   microsoft/unixcoder-base \
    --epochs    10 \
    --checkpoint-dir checkpoints/m4 \
    --pair-types req_test req_hunk orig_hunk req_orig \
    --tier3     data/cache/tier3_hunks.jsonl \
    --same-repo \
    --tier-aware-loss

echo "Done: $(date)"
