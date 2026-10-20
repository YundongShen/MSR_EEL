#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=baseline_pretrained
#SBATCH --output=logs/baseline_pretrained_%j.out
#SBATCH --error=logs/baseline_pretrained_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

MODEL=${MODEL:-"microsoft/graphcodebert-base"}

TAG=$(echo "$MODEL" | tr '/' '_')

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Model: $MODEL"

python baselines/baselines_pretrained.py \
    --model     "$MODEL" \
    --instances data/processed/instances_full.jsonl \
    --tier3     data/cache/tier3_hunks.jsonl \
    --repo-pool \
    --output    "logs/baseline_pretrained_${TAG}.json"

echo "Done: $(date)"
