#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --job-name=exp4_lso
#SBATCH --output=logs/train_exp4_lso_%j.out
#SBATCH --error=logs/train_exp4_lso_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

python train.py \
    --instances data/processed/instances_non_sympy.jsonl \
    --encoder microsoft/unixcoder-base \
    --epochs 10 \
    --batch-size 32 \
    --tier3 data/cache/tier3_hunks.jsonl \
    --llm-t12 data/cache/llm_t12_hunks.jsonl \
    --pair-types req_hunk req_test orig_hunk req_orig \
    --checkpoint-dir checkpoints/exp4_lso
