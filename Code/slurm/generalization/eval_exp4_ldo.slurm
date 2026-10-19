#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_exp4_ldo
#SBATCH --output=logs/eval_exp4_ldo_%j.out
#SBATCH --error=logs/eval_exp4_ldo_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Exp4: Evaluate LDO model on ALL 849 Django instances.
# --all-instances bypasses splits.json filtering, using every instance in the file.
python evaluate.py \
    --exp          retrieval \
    --checkpoint   checkpoints/exp4_ldo/best.pt \
    --instances    data/processed/instances_django_only.jsonl \
    --tier3        data/cache/tier3_hunks.jsonl \
    --all-instances \
    --repo-pool

echo "Done: $(date)"
