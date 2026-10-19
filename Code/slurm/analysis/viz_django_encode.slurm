#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=viz_encode
#SBATCH --output=logs/viz_encode_%j.out
#SBATCH --error=logs/viz_encode_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python analysis/viz_django.py \
    --checkpoint-m2 checkpoints/m2/best.pt \
    --instances      data/processed/instances_full.jsonl \
    --tier3          data/cache/tier3_hunks.jsonl \
    --out-prefix     logs/django_viz \
    --encode-only

echo "Done: $(date)"
