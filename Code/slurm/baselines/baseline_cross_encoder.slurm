#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=ebd_cross_encoder
#SBATCH --output=logs/baseline_cross_encoder_%j.out
#SBATCH --error=logs/baseline_cross_encoder_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

MODEL=${MODEL:-"checkpoints/cross_encoder"}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Model: $MODEL"

python baselines/baselines_extra.py \
    --mode      cross_encoder \
    --model     "$MODEL" \
    --instances data/processed/instances_full.jsonl \
    --tier3     data/cache/tier3_hunks.jsonl \
    --llm-t12   data/cache/llm_t12_hunks.jsonl \
    --repo-pool \
    --output    logs/baseline_cross_encoder.json

echo "Done: $(date)"
