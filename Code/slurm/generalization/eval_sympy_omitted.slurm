#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_sympy_omitted
#SBATCH --output=logs/eval_sympy_omitted_%j.out
#SBATCH --error=logs/eval_sympy_omitted_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

python data_pipeline/prepare_rq4.py --dataset sympy_1cdf6f2
python evaluate.py \
    --exp retrieval \
    --checkpoint checkpoints/exp4_lso/best.pt \
    --instances data/cache/rq4/sympy_1cdf6f2/instances.jsonl \
    --llm-t12 data/cache/rq4/sympy_1cdf6f2/llm_t12_hunks.jsonl \
    --tier3 data/cache/rq4/sympy_1cdf6f2/tier3_hunks.jsonl \
    --score-alpha 1.0 --score-beta 0.5 --score-gamma 0.5 \
    --all-instances \
    --repo-pool
