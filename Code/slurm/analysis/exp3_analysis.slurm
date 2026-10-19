#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=exp3_analysis
#SBATCH --output=logs/exp3_analysis_%j.out
#SBATCH --error=logs/exp3_analysis_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python analysis/exp3_analysis.py \
    --m1        checkpoints/m1/best.pt \
    --m2        checkpoints/m2/best.pt \
    --instances data/processed/instances_full.jsonl \
    --tier3     data/cache/tier3_hunks.jsonl \
    --output    logs/exp3_analysis.json \
    --analysis  all

echo "Done: $(date)"
echo "Output: $(ls -lh logs/exp3_analysis.json 2>/dev/null || echo 'not found')"
