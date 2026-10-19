#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --job-name=ebd_parse
#SBATCH --output=logs/parse_%j.out
#SBATCH --error=logs/parse_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
source gpu_env/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"

python data_pipeline/parse_instances.py \
    --input  data/raw/swebench_full_instances.jsonl \
    --output data/processed/instances_full.jsonl

echo "Done: $(date)"
echo "Instances written: $(wc -l < data/processed/instances_full.jsonl)"
