#!/bin/bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=ebd_candidates_deepseek
#SBATCH --output=logs/tier3_deepseek_%j.out
#SBATCH --error=logs/tier3_deepseek_%j.err

# EXP4 Cross-LLM: Generate candidate patches with DeepSeek-Coder-7B-Instruct via local vLLM server.

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0
source gpu_env/bin/activate

export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_CACHE=$HF_HOME/hub

MODEL="deepseek-ai/deepseek-coder-7b-instruct-v1.5"
PORT=8101

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

echo "Starting vLLM server on port $PORT ..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i} * 5 seconds"
        break
    fi
    sleep 5
done

if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 10 minutes"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "Generating T3 hunks with $MODEL ..."
python data_pipeline/generate_candidates.py \
    --input       data/raw/swebench_full_instances.jsonl \
    --generations      data/cache/deepseek_generations.jsonl \
    --provider    openai \
    --api-base    "http://localhost:${PORT}/v1" \
    --model       "$MODEL" \
    --split       test \
    --max-tokens  1024 \
    --code-budget 6000 \
    --sleep       0.2

echo "Generation done: $(date)"
python data_pipeline/build_dataset.py \
    --generations data/cache/deepseek_generations.jsonl \
    --retained-out data/cache/llm_t12_hunks_deepseek.jsonl \
    --tier3-out data/cache/tier3_hunks_deepseek.jsonl \
    --manifest data/cache/dataset_manifest_deepseek.json

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done: $(date)"
