#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_orig_only
#SBATCH --output=logs/eval_orig_only_%j.out
#SBATCH --error=logs/eval_orig_only_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python evaluate.py \
    --exp          retrieval \
    --checkpoint   checkpoints/m2_abl_no_req/best.pt \
    --llm-t12 data/cache/llm_t12_hunks.jsonl \
    --instances    data/processed/instances_full.jsonl \
    --tier3        data/cache/tier3_hunks.jsonl \
    --repo-pool \
    --score-alpha  0.0 \
    --score-beta   0.0 \
    --score-gamma  1.0

echo "Done: $(date)"
