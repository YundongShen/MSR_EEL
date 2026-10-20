#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_m2_abl_no_test
#SBATCH --output=logs/eval_m2_abl_no_test_%j.out
#SBATCH --error=logs/eval_m2_abl_no_test_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python evaluate.py \
    --checkpoint checkpoints/m2_abl_no_test/best.pt \
    --llm-t12 data/cache/llm_t12_hunks.jsonl \
    --instances  data/processed/instances_full.jsonl \
    --tier3      data/cache/tier3_hunks.jsonl \
    --exp        retrieval \
    --repo-pool \
    --score-alpha 1.0 \
    --score-beta  0.0 \
    --score-gamma 0.5

echo "Done: $(date)"
