#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_gemini
#SBATCH --output=logs/eval_gemini_%j.out
#SBATCH --error=logs/eval_gemini_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python data_pipeline/prepare_rq4.py --dataset crossllm_gemini
python evaluate.py \
    --exp retrieval \
    --checkpoint checkpoints/m2/best.pt \
    --instances data/cache/rq4/crossllm_gemini/instances.jsonl \
    --llm-t12 data/cache/rq4/crossllm_gemini/llm_t12_hunks.jsonl \
    --tier3 data/cache/rq4/crossllm_gemini/tier3_hunks.jsonl \
    --score-alpha 1.0 --score-beta 0.5 --score-gamma 0.5 \
    --all-instances \
    --repo-pool
