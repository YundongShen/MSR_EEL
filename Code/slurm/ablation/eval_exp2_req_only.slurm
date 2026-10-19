#!/bin/bash
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_exp2_req_only
#SBATCH --output=logs/eval_exp2_req_only_%j.out
#SBATCH --error=logs/eval_exp2_req_only_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# REQ-only single-view model: trained on req_hunk only (train_exp2_req_only.slurm), scored by REQ only
python evaluate.py \
    --exp          retrieval \
    --checkpoint   checkpoints/exp2_req_only/best.pt \
    --instances    data/processed/instances_full.jsonl \
    --tier3        data/cache/tier3_hunks.jsonl \
    --repo-pool \
    --score-alpha  1.0 \
    --score-beta   0.0 \
    --score-gamma  0.0

echo "Done: $(date)"
