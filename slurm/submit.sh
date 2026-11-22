#!/bin/bash
# Usage: slurm/submit.sh configs/e1_capacity.yaml [more configs...]
# Submits one single-node job per config (all tasks run in parallel inside it).
# For very large configs use slurm/submit_array.sh instead.
set -euo pipefail
mkdir -p logs
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  sbatch -J "$name" slurm/sweep.sbatch "$cfg"
done
