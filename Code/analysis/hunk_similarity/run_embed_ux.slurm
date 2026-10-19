#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --job-name=ux_sim
#SBATCH --output=logs/hunk_similarity/ux_%j.out
#SBATCH --error=logs/hunk_similarity/ux_%j.err
set -euo pipefail
cd "${EEL_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
"${EEL_PYTHON_BIN:-python}" analysis/hunk_similarity/embed_ux.py "$@"
