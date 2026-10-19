#!/bin/bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_score_abl
#SBATCH --output=logs/eval_score_ablation_%j.out
#SBATCH --error=logs/eval_score_ablation_%j.err

set -euo pipefail
export PYTHONHASHSEED=42

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

CHECKPOINT="checkpoints/m4/best.pt"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint: $CHECKPOINT"

# E4: Score ablation — ablate which entities contribute to EntailmentScore.
# Default weights: alpha=1.0, beta=0.5, gamma=0.5.
# When an entity is removed, its weight is set to 0 (the other kept at default).
for label_alpha_beta_gamma in \
    "REQ_only         1.0 0.0 0.0" \
    "REQ+TEST_no_orig 1.0 0.5 0.0" \
    "REQ+ORIG_no_test 1.0 0.0 0.5" \
    "Full_default     1.0 0.5 0.5"; do
    read -r label alpha beta gamma <<< "$label_alpha_beta_gamma"
    echo ""
    echo "=== $label  alpha=$alpha  beta=$beta  gamma=$gamma ==="
    python evaluate.py \
        --exp        retrieval \
        --checkpoint "$CHECKPOINT" \
        --instances  data/processed/instances_full.jsonl \
        --tier3      data/cache/tier3_hunks.jsonl \
        --score-alpha "$alpha" \
        --score-beta  "$beta" \
        --score-gamma "$gamma"
done

echo ""
echo "Done: $(date)"
