#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_m3
#SBATCH --output=logs/eval_m3_%j.out
#SBATCH --error=logs/eval_m3_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# M3: M2 + same-repo cross-issue hard negatives
python evaluate.py \
    --checkpoint checkpoints/m3/best.pt \
    --instances  data/processed/instances_full.jsonl \
    --tier3      data/cache/tier3_hunks.jsonl \
    --exp        all \
    --umap \
    --repo-pool

echo "Done: $(date)"
