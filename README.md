# kvsim

Trace-driven simulator for prefix-aware KV-cache eviction in RAG serving.
Kept outside the Overleaf folder (`../6ae7f1760f7f8025d12aecc2`); figure scripts
write PDFs into that folder's `images/`.

## Local

```bash
uv sync                      # add --extra plot for figures
uv run pytest -q
uv run kvsim configs/smoke.yaml                 # ~10 s, writes results/smoke.csv
uv run --extra plot python scripts/plot_capacity.py results/smoke.csv --out results/smoke.pdf
```

## SQuAD retrieval trace

```bash
mkdir -p data/raw
curl -L -o data/raw/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
uv run --extra trace python scripts/build_squad_trace.py data/raw/train-v1.1.json traces/squad_train.json.gz
uv run kvsim configs/t1_squad_capacity.yaml --workers 16
```

87,599 questions, 18,896 paragraphs, TF-IDF top-10 retrieval (recall@5 0.76),
tokens estimated as 1.3 x words. Configs select it with `workload.trace`.

## Alliance (Slurm)

```bash
bash slurm/setup_env.sh                          # once, on a login node
# edit --account in slurm/sweep.sbatch
bash slurm/submit.sh configs/e*.yaml             # one 32-CPU job per config
bash slurm/submit_array.sh configs/big.yaml      # very large configs: array + merge
```

A 50k-request run takes ~3-5 s per policy, so E1-E5 (1,440 runs) also fit on a
multi-core workstation: `uv run kvsim configs/e1_capacity.yaml --workers 16`.

Results land in `results/<name>.csv`. The simulator is CPU-only; request GPUs
only for the optional prefill-cost calibration.

## Layout

| Path | Contents |
|---|---|
| `src/kvsim/cache.py` | prefix tree, byte accounting |
| `src/kvsim/policies.py` | `lru`, `leaf_lru` (+`lookahead`), `lfu`, `leaf_lfu`, `pgdsf`, `ours`, `ours_nodep` |
| `src/kvsim/costs.py` | analytical prefill cost $c(n,p)$, `l0` |
| `src/kvsim/workload.py` | synthetic RAG generator (Zipf, hotspot shifts) |
| `src/kvsim/simulate.py` | replay + metrics |
| `src/kvsim/run.py` | config -> tasks -> CSV (`--task-id`, `--merge`, `--workers`) |
| `configs/` | one YAML per experiment |
