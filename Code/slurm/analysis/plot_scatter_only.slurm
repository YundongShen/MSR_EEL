#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=scatter_only
#SBATCH --output=logs/scatter_only_%j.out
#SBATCH --error=logs/scatter_only_%j.err

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

python analysis/plot_django_regions.py \
    --embs       logs/django_viz_embs.npz \
    --out-prefix logs/django \
    --scatter-only

echo "Done: $(date)"
