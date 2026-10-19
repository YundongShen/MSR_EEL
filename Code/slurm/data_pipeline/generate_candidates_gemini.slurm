#!/bin/bash
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=ebd_candidates_gemini
#SBATCH --output=logs/tier3_gemini_%j.out
#SBATCH --error=logs/tier3_gemini_%j.err

# Cross-LLM generalization experiment (EXP4).
# Generates candidate patches with Gemini (OpenAI-compatible API) on the 229 test instances only.
# Retained / Tier-3 hunks are built from these generations by data_pipeline/build_dataset.py (run below).
#
# Submit with:
#   sbatch --export=GEMINI_API_KEY=<your-key> slurm/data_pipeline/generate_candidates_gemini.slurm

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

# --- Gemini config (fill these in) ---
GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL="models/gemini-2.5-flash-lite"
# Set GEMINI_API_KEY via --export when submitting, or export it here.
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
