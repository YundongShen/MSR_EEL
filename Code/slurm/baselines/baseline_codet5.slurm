#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=ebd_codet5
#SBATCH --output=logs/baseline_codet5_%j.out
#SBATCH --error=logs/baseline_codet5_%j.err

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

MODEL=${MODEL:-"Salesforce/codet5p-110m-embedding"}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Model: $MODEL"

echo "==> Downloading model if not cached ..."
python - <<'PYEOF'
from transformers import AutoTokenizer, AutoModel
import os
m = os.environ.get("MODEL", "Salesforce/codet5p-110m-embedding")
AutoTokenizer.from_pretrained(m, trust_remote_code=True)
AutoModel.from_pretrained(m, trust_remote_code=True)
print(f"{m}  OK")
PYEOF

python baselines/baselines_extra.py \
    --mode      codet5 \
    --model     "$MODEL" \
    --instances data/processed/instances_full.jsonl \
    --tier3     data/cache/tier3_hunks.jsonl \
    --repo-pool \
    --output    logs/baseline_codet5.json

echo "Done: $(date)"
