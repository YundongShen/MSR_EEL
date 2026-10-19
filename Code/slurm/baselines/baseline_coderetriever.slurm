#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=ebd_coderetriever
#SBATCH --output=logs/baseline_coderetriever_%j.out
#SBATCH --error=logs/baseline_coderetriever_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

MODEL=${MODEL:-"microsoft/graphcodebert-base"}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Model: $MODEL"

echo "==> Downloading model if not cached ..."
python - <<'PYEOF'
from transformers import AutoTokenizer, AutoModel
import os
m = os.environ.get("MODEL", "microsoft/graphcodebert-base")
AutoTokenizer.from_pretrained(m)
AutoModel.from_pretrained(m)
print(f"{m}  OK")
PYEOF

python baselines/baselines_extra.py \
    --mode      coderetriever \
    --model     "$MODEL" \
    --instances data/processed/instances_full.jsonl \
    --tier3     data/cache/tier3_hunks.jsonl \
    --repo-pool \
    --output    logs/baseline_coderetriever.json

echo "Done: $(date)"
