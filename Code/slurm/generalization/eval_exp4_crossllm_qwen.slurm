#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_exp4_qwen_qwen
#SBATCH --output=logs/eval_exp4_crossllm_qwen_%j.out
#SBATCH --error=logs/eval_exp4_crossllm_qwen_%j.err

# EXP4: Cross-LLM generalization.
# Evaluates M2 model on test split with Qwen-generated T3 hunks.
# Baseline for comparison: eval with Claude Haiku T3 (logs/eval_m2*.err).
# Run AFTER generate_candidates_qwen.slurm has finished.

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python evaluate.py \
    --exp        retrieval \
    --checkpoint checkpoints/m2/best.pt \
    --instances  data/processed/instances_full.jsonl \
    --llm-t12   data/cache/llm_t12_hunks_qwen.jsonl \
    --tier3     data/cache/tier3_hunks_qwen.jsonl \
    --repo-pool

echo "Done: $(date)"
