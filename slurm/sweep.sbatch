#!/bin/bash
# Run one experiment config. Default: one node, all tasks in parallel
# (--workers = allocated CPUs). Submitted with --array, it instead runs one task
# per array index (for configs too large for one job). The simulator is
# CPU-only; GPUs are needed only for the optional prefill-cost calibration.
#SBATCH --account=def-CHANGEME
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
##SBATCH --gpus-per-node=1        # only for GPU calibration jobs
set -euo pipefail
module load python/3.11
source "$HOME/envs/kvsim/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python -m kvsim.run "$1"                       # one task; merge afterwards
else
  python -m kvsim.run "$1" --workers "$SLURM_CPUS_PER_TASK"
fi
