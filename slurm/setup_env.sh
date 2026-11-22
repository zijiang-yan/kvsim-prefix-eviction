#!/bin/bash
# One-time setup on an Alliance login node. Uses the Alliance wheelhouse
# (--no-index) as recommended for their clusters; no conda.
set -euo pipefail
module load python/3.11
virtualenv --no-download "$HOME/envs/kvsim"
source "$HOME/envs/kvsim/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy pyyaml pandas matplotlib
echo "ok: $HOME/envs/kvsim"
