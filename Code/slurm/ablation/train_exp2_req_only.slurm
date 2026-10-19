#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=train_exp2_req_only
#SBATCH --output=logs/train_exp2_req_only_%j.out
#SBATCH --error=logs/train_exp2_req_only_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Exp2 ablation: REQ-only signal
# Only (REQ, HUNK) pairs — no TEST, no ORIG signal in training
# At inference: alpha=1, beta=0, gamma=0
python train.py \
    --instances      data/processed/instances_full.jsonl \
    --encoder        microsoft/unixcoder-base \
    --epochs         10 \
    --batch-size     32 \
    --tier3          data/cache/tier3_hunks.jsonl \
    --llm-t12        data/cache/llm_t12_hunks.jsonl \
    --pair-types     req_hunk \
    --checkpoint-dir checkpoints/exp2_req_only

echo "Done: $(date)"
