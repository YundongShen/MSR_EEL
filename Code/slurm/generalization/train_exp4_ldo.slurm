#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=exp4_ldo
#SBATCH --output=logs/train_exp4_ldo_%j.out
#SBATCH --error=logs/train_exp4_ldo_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Exp4: Leave-Django-Out — same M2 setup, trained on 1442 non-Django instances only.
# splits.json still used; _subset() filters to non-Django train/val IDs automatically.
python train.py \
    --instances      data/processed/instances_non_django.jsonl \
    --encoder        microsoft/unixcoder-base \
    --epochs         10 \
    --batch-size     32 \
    --tier3          data/cache/tier3_hunks.jsonl \
    --llm-t12        data/cache/llm_t12_hunks.jsonl \
    --pair-types     req_hunk req_test orig_hunk req_orig \
    --checkpoint-dir checkpoints/exp4_ldo

echo "Done: $(date)"
echo "Best checkpoint: $(ls -lh checkpoints/exp4_ldo/best.pt 2>/dev/null || echo 'not found')"
