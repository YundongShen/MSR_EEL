#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=ebd_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source gpu_env/bin/activate

REPOS_DIR="${TMPDIR}/repos"
mkdir -p "$REPOS_DIR"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "TMPDIR repos: $REPOS_DIR"

python data_pipeline/download_swebench.py \
    --dataset princeton-nlp/SWE-bench \
    --output data/raw/swebench_full_instances.jsonl \
    --repos-dir "$REPOS_DIR"

echo "Done: $(date)"
echo "Total instances written: $(wc -l < data/raw/swebench_full_instances.jsonl)"
