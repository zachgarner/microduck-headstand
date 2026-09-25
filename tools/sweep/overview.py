"""One page for a family of one-axis sweeps: small multiples and failure edges.

    cd microduck_rl
    uv run --with matplotlib ../tools/sweep/overview.py ../results/sweeps/baseline_* --out ../results/sweeps/baseline_overview

Writes `overview.png` and `overview.md` to the output directory. The failure
edge on each side of the nominal value is the first grid value, moving outward,
whose success rate falls below `--threshold` (90% by default). The table also
reports each axis's worst success rate inside its training range.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from report import BAND, GRID, INK, INK_2, SERIES, SURFACE, by_value, fmt, load, style, wilson  # noqa: E402


def edges(xs, rate, nominal, threshold):
    below = [x for x, r in zip(xs, rate) if x < nominal and r < threshold]
    above = [x for x, r in zip(xs, rate) if x > nominal and r < threshold]
    return (max(below) if below else None, min(above) if above else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--title", default="Routine success by pinned parameter")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sweeps = [s for s in sorted(args.sweeps) if (s / "records.jsonl").exists()]
    cols = 4
    rows = math.ceil(len(sweeps) / cols)
    fig, grid = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.5 * rows), squeeze=False, facecolor=SURFACE)
    table = []
    for i, sweep in enumerate(sweeps):
        records, prov = load(sweep)
        axis = list(prov["config"]["axes"])[0]
        meta = prov["axes"][axis]
        groups = by_value(records, axis)
        xs = list(groups)
        k = [sum(r["success"] for r in g) for g in groups.values()]
        n = [len(g) for g in groups.values()]
        rate = [a / b for a, b in zip(k, n)]
        ci = [wilson(a, b) for a, b in zip(k, n)]
        t_lo, t_hi = meta["training_range"]
        ax = grid[i // cols][i % cols]
        style(ax)
        if t_hi > t_lo:
            ax.axvspan(t_lo, t_hi, color=BAND, zorder=0)
        ax.axhline(args.threshold, color=GRID, linewidth=0.8, zorder=1)
        ax.axvline(meta["nominal"], color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.fill_between(xs, [a for a, _ in ci], [b for _, b in ci], color=SERIES[0], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(xs, rate, color=SERIES[0], linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        label = prov["config"].get("stage", axis) if prov["config"]["mode"] == "handover" else axis
        ax.set_title(label, loc="left", fontsize=9, color=INK)
        ax.set_xlabel(meta["unit"], fontsize=7, color=INK_2)
        lo_edge, hi_edge = edges(xs, rate, meta["nominal"], args.threshold)
        inside = [r for x, r in zip(xs, rate) if t_lo <= x <= t_hi]
        fails = (Counter(r["first_unfinished_stage"] or "fell after finishing" for r in records if not r["success"])
                 if prov["config"]["mode"] == "routine" else Counter())
        table.append((label, meta, lo_edge, hi_edge, min(inside) if inside else None,
                      sum(k), sum(n), fails.most_common(3)))
    for j in range(len(sweeps), rows * cols):
        grid[j // cols][j % cols].axis("off")
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(args.out / "overview.png", dpi=150, facecolor=SURFACE)

    pct = lambda v: "none in range" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    lines = [f"# {args.title}", "",
             f"Each panel pins one parameter and runs the routine at every grid value. The shaded band is the training range, "
             f"the dashed line the nominal value, and the grey line the {100 * args.threshold:.0f}% threshold.", "",
             "![overview](overview.png)", "",
             f"The failure edge is the first value, moving outward from nominal, where success falls below "
             f"{100 * args.threshold:.0f}%.", "",
             "| Axis | Training range | Lower failure edge | Upper failure edge | Worst success inside the training range | All attempts | Most common first unfinished stages |",
             "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for axis, meta, lo_e, hi_e, worst_in, k, n, fails in table:
        t_lo, t_hi = meta["training_range"]
        rng = f"{fmt(t_lo)} to {fmt(t_hi)}" if t_hi > t_lo else "not randomized"
        lines.append(f"| {axis} ({meta['unit']}) | {rng} | {fmt(lo_e) if lo_e is not None else 'none in grid'} | "
                     f"{fmt(hi_e) if hi_e is not None else 'none in grid'} | {pct(worst_in)} | {k}/{n} | "
                     + ", ".join(f"{s} {c}" for s, c in fails) + " |")
    lines.append("")
    (args.out / "overview.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
