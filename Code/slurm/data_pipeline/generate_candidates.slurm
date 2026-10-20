#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output=logs/generate_candidates_%j.out
#SBATCH --error=logs/generate_candidates_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set. Submit with:"
    echo "  sbatch --export=ANTHROPIC_API_KEY=\$ANTHROPIC_API_KEY slurm/data_pipeline/generate_candidates.slurm"
    exit 1
fi

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Model: claude-haiku-4-5-20251001"

python data_pipeline/generate_candidates.py \
    --input  data/raw/swebench_full_instances.jsonl \
    --generations data/cache/haiku_generations.jsonl \
    --model  claude-haiku-4-5-20251001

echo "Done: $(date)"
python data_pipeline/build_dataset.py
