#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_exp4_lso
#SBATCH --output=logs/eval_exp4_lso_%j.out
#SBATCH --error=logs/eval_exp4_lso_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

python evaluate.py \
    --exp retrieval \
    --checkpoint checkpoints/exp4_lso/best.pt \
    --instances data/processed/instances_sympy_only.jsonl \
    --tier3 data/cache/tier3_hunks.jsonl \
    --llm-t12 data/cache/llm_t12_hunks.jsonl \
    --score-alpha 1.0 --score-beta 0.5 --score-gamma 0.5 \
    --all-instances \
    --repo-pool
