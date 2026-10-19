#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --job-name=exp2_hunk_scores
#SBATCH --output=logs/exp2_hunk_scores_%j.out
#SBATCH --error=logs/exp2_hunk_scores_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Save per-hunk component scores for M2 (geometry experiment, test split only)
python evaluate.py \
    --checkpoint       checkpoints/m2/best.pt \
    --instances        data/processed/instances_full.jsonl \
    --tier3            data/cache/tier3_hunks.jsonl \
    --exp              geometry \
    --save-hunk-scores logs/exp2_hunk_scores_m2.jsonl

echo "Done: $(date)"
