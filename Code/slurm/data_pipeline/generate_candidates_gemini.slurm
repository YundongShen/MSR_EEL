#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=generate_candidates_gemini
#SBATCH --output=logs/generate_candidates_gemini_%j.out
#SBATCH --error=logs/generate_candidates_gemini_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL="models/gemini-2.5-flash-lite"
export OPENAI_API_KEY="${GEMINI_API_KEY:-}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set."
    echo "  sbatch --export=GEMINI_API_KEY=<key> slurm/data_pipeline/generate_candidates_gemini.slurm"
    exit 1
fi

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Model: ${GEMINI_MODEL}  API: ${GEMINI_API_BASE}"

python data_pipeline/generate_candidates.py \
    --input    data/raw/swebench_full_instances.jsonl \
    --generations   data/cache/gemini_generations.jsonl \
    --provider openai \
    --api-base "${GEMINI_API_BASE}" \
    --model    "${GEMINI_MODEL}" \
    --split    test \
    --sleep    0.5

echo "Done: $(date)"
python data_pipeline/build_dataset.py \
    --generations data/cache/gemini_generations.jsonl \
    --retained-out data/cache/llm_t12_hunks_gemini.jsonl \
    --tier3-out data/cache/tier3_hunks_gemini.jsonl \
    --manifest data/cache/dataset_manifest_gemini.json
