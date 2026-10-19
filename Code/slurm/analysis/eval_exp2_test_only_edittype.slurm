#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=exp2_test_only_edittype
#SBATCH --output=logs/exp2_test_only_edittype_%j.out
#SBATCH --error=logs/exp2_test_only_edittype_%j.err

# TEST-view edit-type analysis:
#   Step 1 — geometry eval with exp2_test_only/best.pt + beta=1 to save per-hunk scores
#   Step 2 — analysis/exp2_analysis.py joins scores with hunk_edit_types.jsonl labels

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

echo "=== Step 1: geometry eval (alpha=0 beta=1 gamma=0) ==="
python evaluate.py \
    --checkpoint       checkpoints/exp2_test_only/best.pt \
    --instances        data/processed/instances_full.jsonl \
    --tier3            data/cache/tier3_hunks.jsonl \
    --exp              geometry \
    --score-alpha      0.0 \
    --score-beta       1.0 \
    --score-gamma      0.0 \
    --save-hunk-scores logs/exp2_hunk_scores_test_only_b1.jsonl

echo "=== Step 2: edit-type analysis ==="
python analysis/exp2_analysis.py \
    --scores logs/exp2_hunk_scores_test_only_b1.jsonl \
    --labels data/processed/hunk_edit_types.jsonl \
    --output logs/exp2_analysis_test_only_b1.json

echo "Done: $(date)"
