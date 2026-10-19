#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --job-name=m2_abl_no_orig
#SBATCH --output=logs/m2_abl_no_orig_%j.out
#SBATCH --error=logs/m2_abl_no_orig_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# E5 ablation: M2 without ORIG entity (remove orig_hunk + req_orig positive pairs)
python train.py \
    --instances data/processed/instances_full.jsonl \
    --encoder   microsoft/unixcoder-base \
    --epochs    10 \
    --checkpoint-dir checkpoints/m2_abl_no_orig \
    --pair-types req_test req_hunk \
    --tier3     data/cache/tier3_hunks.jsonl

echo "Done: $(date)"
