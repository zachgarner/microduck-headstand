"""Turn a sweep's records into the map: one chart and one table per axis.

    cd microduck_rl
    uv run --with matplotlib ../tools/sweep/report.py ../results/sweeps/<name>

Writes `map.png` and `summary.md` next to the records. For each swept axis the
chart shows the success rate at every grid value with its 95% Wilson interval,
and shades the axis's training range. A routine sweep adds, under each success
chart, the first unfinished stage of every failed attempt. A sweep over two
axes reports each axis with the other one pooled, and a table of the full grid.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette of the dataviz skill, light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#ecebe6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(sweep_dir: Path):
    records = [json.loads(line) for line in (sweep_dir / "records.jsonl").read_text().splitlines()]
    provenance = json.loads((sweep_dir / "provenance.json").read_text())
    return records, provenance


def by_value(records, axis):
    groups = defaultdict(list)
    for r in records:
        groups[r["pinned"][axis]].append(r)
    return dict(sorted(groups.items()))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fmt(v: float) -> str:
    return f"{v:g}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", type=Path)
    args = p.parse_args()
    records, prov = load(args.sweep_dir)
    cfg = prov["config"]
    axes = list(cfg["axes"])
    routine = cfg["mode"] == "routine"
    stage_order = ["fold", "legs_together", "roll", "settle", "fold2", "split",
                   "switch", "splitover", "settle2"] if routine else []
    rows = 2 if routine else 1
    fig, grid = (plt.subplots(rows, len(axes), figsize=(5.2 * len(axes), 3.4 * rows),
                              squeeze=False, facecolor=SURFACE) if axes else (None, None))
    lines = [f"# {cfg['name']}", "", cfg.get("description", ""), "",
             f"{len(records)} attempts, {sum(r['success'] for r in records)} successes. "
             f"Mode: {cfg['mode']}. Seed {cfg['seed']}, {cfg['attempts']} attempts per grid point.", ""]

    for c, axis in enumerate(axes):
        meta = prov["axes"][axis]
        groups = by_value(records, axis)
        xs = list(groups)
        k = [sum(r["success"] for r in g) for g in groups.values()]
        n = [len(g) for g in groups.values()]
        rate = [ki / ni for ki, ni in zip(k, n)]
        lo_hi = [wilson(ki, ni) for ki, ni in zip(k, n)]
        ax = grid[0][c]
        style(ax)
        t_lo, t_hi = meta["training_range"]
        if t_hi > t_lo:
            ax.axvspan(t_lo, t_hi, color=BAND, zorder=0)
            ax.text((t_lo + t_hi) / 2, 0.03, "training\nrange", ha="center", va="bottom",
                    fontsize=7, color=INK_2, transform=ax.get_xaxis_transform(), zorder=4)
        ax.axvline(meta["nominal"], color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.fill_between(xs, [a for a, _ in lo_hi], [b for _, b in lo_hi], color=SERIES[0], alpha=0.15,
                        linewidth=0, zorder=2)
        ax.plot(xs, rate, color=SERIES[0], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"Success rate by {axis}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel(f"{axis} ({meta['unit']})", fontsize=8, color=INK_2)

        header = "| " + axis + " | Attempts | Success | 95% interval |" + (" First unfinished stage |" if routine else "")
        lines += [f"## {axis}", "", meta.get("note", ""), "",
                  f"Unit: {meta['unit']}. Nominal {fmt(meta['nominal'])}. "
                  f"Training range {fmt(t_lo)} to {fmt(t_hi)}.", "",
                  header, "| --- | ---: | ---: | --- |" + (" --- |" if routine else "")]
        for x, ki, ni, (a, b), g in zip(xs, k, n, lo_hi, groups.values()):
            row = f"| {fmt(x)} | {ni} | {ki}/{ni} ({100 * ki / ni:.0f}%) | {100 * a:.0f}% to {100 * b:.0f}% |"
            if routine:
                fails = Counter(r["first_unfinished_stage"] for r in g if not r["success"])
                incomplete_standing = sum(1 for r in g if not r["success"] and r["first_unfinished_stage"] is None)
                parts = [f"{s} {fails[s]}" for s in stage_order if fails.get(s)]
                if incomplete_standing:
                    parts.append(f"fell after finishing {incomplete_standing}")
                row += " " + (", ".join(parts) or "none") + " |"
            lines.append(row)
        lines.append("")

        if routine:
            ax2 = grid[1][c]
            style(ax2)
            fails_by_stage = {s: [sum(1 for r in g if not r["success"] and r["first_unfinished_stage"] == s)
                                  for g in groups.values()] for s in stage_order}
            seen = [s for s in stage_order if any(fails_by_stage[s])]
            width = 0.7 * min(b - a for a, b in zip(xs, xs[1:])) if len(xs) > 1 else 0.5
            bottom = [0] * len(xs)
            for i, s in enumerate(seen):
                ax2.bar(xs, fails_by_stage[s], width=width, bottom=bottom, color=SERIES[i % len(SERIES)],
                        edgecolor=SURFACE, linewidth=1.5, label=s, zorder=2)
                bottom = [b + v for b, v in zip(bottom, fails_by_stage[s])]
            ax2.set_title("Failed attempts by first unfinished stage", loc="left", fontsize=10, color=INK)
            ax2.set_xlabel(f"{axis} ({meta['unit']})", fontsize=8, color=INK_2)
            ax2.set_ylabel("attempts", fontsize=8, color=INK_2)
            if seen:
                ax2.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="upper left", ncols=min(len(seen), 4))

    if routine:
        lines += ["## By seed", "", "| Seed | Success | Final standing | First unfinished stage of failures |",
                  "| --- | ---: | ---: | --- |"]
        for seed in sorted({r["seed"] for r in records}):
            g = [r for r in records if r["seed"] == seed]
            fails = Counter(r["first_unfinished_stage"] or "fell after finishing" for r in g if not r["success"])
            lines.append(f"| {seed} | {sum(r['success'] for r in g)}/{len(g)} | "
                         f"{sum(r['final_standing'] for r in g)} | "
                         + (", ".join(f"{s} {c}" for s, c in fails.items()) or "none") + " |")
        lines.append("")

    if len(axes) == 2:
        a, b = axes
        cells = defaultdict(list)
        for r in records:
            cells[(r["pinned"][a], r["pinned"][b])].append(r["success"])
        bs = sorted({key[1] for key in cells})
        lines += [f"## Full grid: success by {a} (rows) and {b} (columns)", "",
                  "| " + a + " | " + " | ".join(fmt(v) for v in bs) + " |",
                  "| --- |" + " ---: |" * len(bs)]
        for av in sorted({key[0] for key in cells}):
            lines.append(f"| {fmt(av)} | " + " | ".join(
                f"{sum(cells[(av, bv)])}/{len(cells[(av, bv)])}" for bv in bs) + " |")
        lines.append("")

    held = all(abs(r["params_start"][ax_] - r["params_end"][ax_]) < 1e-6 for r in records for ax_ in axes)
    chart = "Chart: [map.png](map.png). " if axes else ""
    lines += ["## Checks", "",
              f"Pinned values held for the whole rollout in every attempt: {'yes' if held else 'NO'}.",
              f"Resets inside a rollout: {max(r['resets_in_batch'] for r in records)}.", "",
              f"{chart}Records: `records.jsonl`. States: `states.npz`. "
              f"Provenance: `provenance.json`, {prov['elapsed_s']:.0f} s of simulation.", ""]
    if fig is not None:
        fig.tight_layout()
        fig.savefig(args.sweep_dir / "map.png", dpi=150, facecolor=SURFACE)
    (args.sweep_dir / "summary.md").write_text("\n".join(lines))
    print((args.sweep_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
