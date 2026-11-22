#!/bin/bash
# Usage: slurm/submit_array.sh configs/big.yaml
# One array task per grid point (1 CPU each), then a merge job.
set -euo pipefail
cfg="$1"
module load python/3.11
source "$HOME/envs/kvsim/bin/activate"
export PYTHONPATH="$PWD/src"
n=$(python -m kvsim.run "$cfg" --count)
name=$(basename "$cfg" .yaml)
account=$(grep -oP '(?<=--account=)\S+' slurm/sweep.sbatch)
mkdir -p logs
jid=$(sbatch --parsable --array=0-$((n - 1)) --cpus-per-task=1 --mem=4G --time=00:30:00 \
  --output="logs/%x_%A_%a.out" -J "$name" slurm/sweep.sbatch "$cfg")
sbatch --dependency=afterok:"$jid" -J "${name}_merge" --account="$account" \
  --time=00:10:00 --mem=2G --output="logs/${name}_merge_%j.out" \
  --wrap "module load python/3.11 && source $HOME/envs/kvsim/bin/activate && PYTHONPATH=$PWD/src python -m kvsim.run $cfg --merge"
echo "submitted $n array tasks as job $jid"
