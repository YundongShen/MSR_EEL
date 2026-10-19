#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=ebd_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

CHECKPOINT=${CHECKPOINT:-"checkpoints/best.pt"}
ENCODER=${ENCODER:-"microsoft/unixcoder-base"}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Checkpoint: $CHECKPOINT"
echo "Encoder: $ENCODER"

python evaluate.py \
    --exp        all \
    --checkpoint "$CHECKPOINT" \
    --encoder    "$ENCODER" \
    --instances  data/processed/instances_full.jsonl \
    --tier3      data/cache/tier3_hunks.jsonl

echo "Done: $(date)"
