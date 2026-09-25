"""The sweep's grid and batch layout. Standard library only, so the Ray driver
can plan a sweep without the simulator installed."""
from __future__ import annotations

import itertools


def grid_points(axes: dict[str, list[float]]) -> list[dict[str, float]]:
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*(axes[n] for n in names))]


def layout(points, attempts: int, max_envs: int):
    """Assign attempts to (batch, env) slots, grid point after grid point."""
    slots = [(p, a) for p in range(len(points)) for a in range(attempts)]
    return [slots[i:i + max_envs] for i in range(0, len(slots), max_envs)]


def batches_for(cfg: dict):
    points = grid_points(cfg["axes"])
    return points, layout(points, cfg["attempts"], cfg.get("max_envs", 256))
