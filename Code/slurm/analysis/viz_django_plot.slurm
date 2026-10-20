#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=viz_django_plot
#SBATCH --output=logs/viz_django_plot_%j.out
#SBATCH --error=logs/viz_django_plot_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"

python analysis/viz_django.py \
    --out-prefix logs/django_viz \
    --plot-only

echo "3D/2D viz done: $(date)"

python analysis/plot_django_regions.py \
    --embs       logs/django_viz_embs.npz \
    --out-prefix logs/django

echo "Combined figure done: $(date)"
