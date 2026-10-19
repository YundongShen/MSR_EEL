#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_m2_geo
#SBATCH --output=logs/eval_m2_geometry_%j.out
#SBATCH --error=logs/eval_m2_geometry_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Re-run geometry only to rebuild umap_embs_m2.npz with per-hunk traceability metadata
python evaluate.py \
    --checkpoint checkpoints/m2/best.pt \
    --instances  data/processed/instances_full.jsonl \
    --tier3      data/cache/tier3_hunks.jsonl \
    --exp        geometry \
    --umap

echo "Done: $(date)"
