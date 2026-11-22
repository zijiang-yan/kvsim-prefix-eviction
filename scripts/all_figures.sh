#!/bin/bash
# Rebuild every paper figure from results/ (run from sim/).
set -euo pipefail
P="uv run --extra plot python scripts/plot_capacity.py"
$P results/e1_capacity.csv --policies lfu leaf_lfu pgdsf ours lru leaf_lru
$P results/t1_squad_capacity.csv --facet wl_order --logx --policies lru leaf_lfu pgdsf ours
( cd ../6ae7f1760f7f8025d12aecc2/images && pdflatex -interaction=nonstopmode orphan.tex >/dev/null && rm -f orphan.aux orphan.log && echo "wrote orphan.pdf" )
