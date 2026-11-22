"""Print a policy x budget table of one metric from a results CSV.

Usage: uv run python scripts/table.py results/smoke.csv [--metric cost_saved_frac] [--by l0]
"""

import argparse
import csv
from collections import defaultdict
from statistics import mean

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--metric", default="cost_saved_frac")
ap.add_argument("--by", default="l0", help="extra column to split tables on")
ap.add_argument("--col", default="budget_frac")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))
groups = defaultdict(list)
for r in rows:
    groups[r.get(args.by, "")].append(r)

for g, sub in groups.items():
    cols = sorted({r[args.col] for r in sub}, key=float)
    pols = list(dict.fromkeys(r["policy"] for r in sub))
    cell = defaultdict(list)
    for r in sub:
        cell[(r["policy"], r[args.col])].append(float(r[args.metric]))
    print(f"\n{args.metric}  ({args.by}={g})")
    print("policy".ljust(16) + "".join(f"{c}"[:12].ljust(14) for c in cols)
          + "us/decision")
    for p in pols:
        dec = mean(float(r["decision_us_mean"]) for r in sub if r["policy"] == p)
        vals = "".join(f"{100 * mean(cell[(p, c)]):6.2f}".ljust(14) for c in cols)
        print(p.ljust(16) + vals + f"{dec:6.1f}")
