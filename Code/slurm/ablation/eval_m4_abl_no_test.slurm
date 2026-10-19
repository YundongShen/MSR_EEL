#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_abl_no_test
#SBATCH --output=logs/eval_m4_abl_no_test_%j.out
#SBATCH --error=logs/eval_m4_abl_no_test_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# E5 ablation: M4 trained without TEST entity (no req_test / test_hunk pairs).
# Evaluate with beta=0 (matched score: no TEST term) for fairest comparison.
python evaluate.py \
    --checkpoint checkpoints/m2_abl_no_test/best.pt \
    --instances  data/processed/instances_full.jsonl \
    --tier3      data/cache/tier3_hunks.jsonl \
    --exp        all \
    --repo-pool \
    --score-alpha 1.0 \
    --score-beta  0.0 \
    --score-gamma 0.5

echo "Done: $(date)"
