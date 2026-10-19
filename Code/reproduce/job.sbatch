#!/bin/bash
# Submit from Code after setup: sbatch reproduce/job.sbatch python train.py ...
# Supply the cluster account, partition and GPU type through sbatch options.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch reproduce/job.sbatch command [arguments...]" >&2
    exit 2
fi
CODE_ROOT=${EEL_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ -d "$CODE_ROOT/Code/reproduce" ]]; then
    CODE_ROOT="$CODE_ROOT/Code"
fi
cd "$CODE_ROOT"
source env/bin/activate
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "Node: $(hostname)"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
exec "$@"
