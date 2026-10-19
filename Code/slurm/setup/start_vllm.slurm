#!/bin/bash
#SBATCH --gpus=A100-SXM4-80GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --job-name=ebd_vllm
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

# -----------------------------------------------------------------------
# Start a vLLM OpenAI-compatible server for Qwen2.5-Coder-32B-Instruct.
#
# After this job starts, submit collect_pairs.slurm pointing to this node:
#   --llm-api-base http://<node>:8000/v1
#
# The node name is printed to logs/vllm_<jobid>.out at startup.
# -----------------------------------------------------------------------

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
MODEL="${EEL_VLLM_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"

cd "$PROJECT"
source gpu_env/bin/activate

mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model: $MODEL"

# Write node name to file so collect job can find it.
echo "$(hostname)" > logs/vllm_node.txt

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 12288 \
    --tensor-parallel-size 1 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --served-model-name Qwen2.5-Coder-32B-Instruct
