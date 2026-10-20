#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=plot_scatter_only
#SBATCH --output=logs/plot_scatter_only_%j.out
#SBATCH --error=logs/plot_scatter_only_%j.err

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

python analysis/plot_django_regions.py \
    --embs       logs/django_viz_embs.npz \
    --out-prefix logs/django \
    --scatter-only

echo "Done: $(date)"
