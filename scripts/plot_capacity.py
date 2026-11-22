"""Hit rate vs. cache budget, one line per policy (mean and 95% CI over seeds).

Usage: uv run --extra plot python scripts/plot_capacity.py results/e1_capacity.csv
Writes a PDF into the paper's images/ folder by default.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PAPER_IMAGES = Path(__file__).resolve().parents[2] / "6ae7f1760f7f8025d12aecc2" / "images"

ap = argparse.ArgumentParser()
ap.add_argument("csv", type=Path)
ap.add_argument("--out", type=Path)
ap.add_argument("--metric", default="cost_saved_frac")
ap.add_argument("--policies", nargs="+", help="subset to plot, in legend order")
ap.add_argument("--facet", help="column to split into side-by-side panels (e.g. wl_order)")
ap.add_argument("--logx", action="store_true")
args = ap.parse_args()

df = pd.read_csv(args.csv)
if args.policies:
    df = df[df["policy"].isin(args.policies)]
    df["policy"] = pd.Categorical(df["policy"], args.policies, ordered=True)
out = args.out or PAPER_IMAGES / f"{args.csv.stem}.pdf"

facets = sorted(df[args.facet].unique()) if args.facet else [None]
fig, axes = plt.subplots(1, len(facets), figsize=(3.3, 2.3), sharey=False, squeeze=False)
for ax, f in zip(axes[0], facets):
    sub = df if f is None else df[df[args.facet] == f]
    g = sub.groupby(["policy", "budget_frac"], observed=True)[args.metric]
    stats = g.agg(["mean", "std", "count"]).reset_index()
    stats["ci"] = 1.96 * stats["std"].fillna(0) / stats["count"] ** 0.5
    for pol, s in stats.groupby("policy", sort=True, observed=True):
        ax.errorbar(s["budget_frac"] * 100, s["mean"] * 100, yerr=s["ci"] * 100,
                    marker="o", ms=2.5, lw=1.0, capsize=1.5, label=pol)
    if f is not None:
        ax.set_title(str(f), fontsize=8)
    if args.logx:
        ax.set_xscale("log")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
fig.supxlabel("Cache budget (% of working set)", fontsize=8)
axes[0][0].set_ylabel({"cost_saved_frac": "Prefill compute saved (%)",
                       "hit_rate": "Prefix-token hit rate (%)"}.get(args.metric, args.metric),
                      fontsize=8)
axes[0][-1].legend(fontsize=5.5, frameon=False)
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}")
