#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=ebd_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

ENCODER=${ENCODER:-"microsoft/unixcoder-base"}

echo "Encoder: $ENCODER"
echo "Instances: data/processed/instances_full.jsonl"

python train.py \
    --instances data/processed/instances_full.jsonl \
    --encoder   "$ENCODER" \
    --epochs    10 \
    --batch-size 32 \
    --resume    checkpoints/best.pt

echo "Done: $(date)"
echo "Best checkpoint: $(ls -lh checkpoints/best.pt 2>/dev/null || echo 'not found')"
