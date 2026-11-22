"""Expand an experiment config into tasks and run them.

Config (YAML):
    name: e1_capacity
    kv_bytes_per_token: 131072      # Llama-3-8B: 2 * 32 layers * 8 heads * 128 dim * 2 B
    l0: 26700                        # attention crossover length (costs.py); .inf = tokens only
    warmup_frac: 0.1
    seeds: [0, 1, 2, 3, 4]
    workload: {n_requests: 20000, zipf_alpha: 0.9, ...}
    sweep:                           # crossed; "workload.x" overrides workload keys
      budget_frac: [0.05, 0.1, 0.2]
    policies:
      - {name: leaf_lru}
      - {name: ours, k: 8, decay: 0.001}
      - {name: ours, k: 8, decay: 0.0, label: ours_nodecay}

Task = one (sweep point, policy, seed). Run everything locally, or one task per
Slurm array index with --task-id (defaults to $SLURM_ARRAY_TASK_ID).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import yaml

from .costs import LLAMA3_8B_L0
from .policies import make_policy
from .simulate import simulate
from .workload import make_workload, unique_tokens


def expand(cfg: dict) -> list[dict]:
    sweep = cfg.get("sweep", {})
    keys = list(sweep)
    tasks = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        point = dict(zip(keys, values))
        for pol in cfg["policies"]:
            for seed in cfg.get("seeds", [0]):
                tasks.append({"point": point, "policy": pol, "seed": seed})
    return tasks


@lru_cache(maxsize=8)
def _trace(wl_items: tuple):
    wl = make_workload(dict(wl_items))
    trace = wl.generate()
    return trace, unique_tokens(trace)


def run_task(cfg: dict, task: dict) -> dict:
    wl = dict(cfg.get("workload", {}))
    budget_frac = None
    l0 = float(cfg.get("l0", LLAMA3_8B_L0))
    for k, v in task["point"].items():
        if k.startswith("workload."):
            wl[k.split(".", 1)[1]] = v
        elif k == "budget_frac":
            budget_frac = v
        elif k == "l0":
            l0 = float(v)
        else:
            raise ValueError(f"unknown sweep key {k!r}")
    wl["seed"] = task["seed"]
    trace, uniq = _trace(tuple(sorted(wl.items())))

    kvb = cfg.get("kv_bytes_per_token", 131072)
    frac = budget_frac if budget_frac is not None else cfg.get("budget_frac", 0.1)
    capacity = int(frac * uniq * kvb)

    pol_cfg = dict(task["policy"])
    label = pol_cfg.pop("label", pol_cfg["name"])
    policy = make_policy(pol_cfg.pop("name"), **pol_cfg)
    metrics = simulate(trace, policy, capacity, kvb, cfg.get("warmup_frac", 0.1), l0)

    row = {"experiment": cfg["name"], "policy": label, "seed": task["seed"],
           "budget_frac": frac, "l0": l0, "working_set_tokens": uniq}
    row.update({k: v for k, v in task["policy"].items() if k not in ("name", "label")})
    row.update({k.replace("workload.", "wl_"): v for k, v in task["point"].items()
                if k not in ("budget_frac", "l0")})
    row.update(metrics)
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _run_one(args):
    return run_task(*args)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kvsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, help="results root (default: results/)",
                    default=Path("results"))
    ap.add_argument("--task-id", type=int,
                    default=int(os.environ["SLURM_ARRAY_TASK_ID"])
                    if "SLURM_ARRAY_TASK_ID" in os.environ else None)
    ap.add_argument("--count", action="store_true", help="print number of tasks and exit")
    ap.add_argument("--merge", action="store_true", help="merge per-task CSVs")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    tasks = expand(cfg)
    parts = args.out / cfg["name"]

    if args.count:
        print(len(tasks))
        return
    if args.merge:
        rows = []
        for p in sorted(parts.glob("task_*.csv")):
            with p.open(newline="") as f:
                rows.extend(csv.DictReader(f))
        write_csv(rows, args.out / f"{cfg['name']}.csv")
        print(f"merged {len(rows)}/{len(tasks)} rows -> {args.out / (cfg['name'] + '.csv')}")
        return
    if args.task_id is not None:
        row = run_task(cfg, tasks[args.task_id])
        write_csv([row], parts / f"task_{args.task_id:05d}.csv")
        return

    jobs = [(cfg, t) for t in tasks]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            rows = pool.map(_run_one, jobs)
    else:
        rows = []
        for i, j in enumerate(jobs):
            rows.append(_run_one(j))
            r = rows[-1]
            print(f"[{i + 1}/{len(jobs)}] {r['policy']:<14} budget={r['budget_frac']:<5} "
                  f"seed={r['seed']} hit_rate={r['hit_rate']:.4f}", file=sys.stderr)
    out = args.out / f"{cfg['name']}.csv"
    write_csv(rows, out)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
