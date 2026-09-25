"""Run a parameter sweep from a config and write one record per attempt.

    cd microduck_rl
    uv run ../tools/sweep/run_sweep.py ../tools/sweep/configs/spike_backroll_friction.json

A config names the mode (one policy, the full routine, or one routine stage
started from a routine sweep's recorded handover states), the robot model
("plain", the default, or "backlash"), the axes and their
grid values, the attempts at each grid point, and the seed. The grid is the
full product of the axes' values. Each grid point gets `attempts` environments,
laid out point after point, and environments are split into batches of at most
`max_envs`. Batch b runs at seed `seed + b`. `--workers` runs batches in
parallel processes. Warp's CPU backend steps a batch on one core, so on a CPU
machine the throughput comes from running batches side by side.

Every batch writes its own files under `batches/`, and a merge combines them.
`--batch B` runs one batch and `--merge` only merges, which is how
`ray_sweep.py` spreads batches over a Ray cluster.

The output directory holds:

- `records.jsonl`: one line per attempt, with the pinned values, the readback
  of every axis at the start and the end, and the outcome.
- `states.npz`: the state at every handover and at the end, in the handover-set
  layout, indexed by the record's `attempt`.
- `provenance.json`: the config, checkpoint and tool hashes, code revisions and
  package versions.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes as sweep_axes  # noqa: E402
import rollout  # noqa: E402
from plan import batches_for  # noqa: E402

SWEEP_DIR = Path(__file__).resolve().parent
RESULTS = SWEEP_DIR.parent.parent / "results" / "sweeps"


def revision(directory: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()


def dirty(directory: Path) -> bool:
    """Whether tracked files differ from the revision. New result files do not count."""
    return bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=directory, text=True).strip())


def names_of_state_axes(cfg):
    return [a for a in cfg["axes"] if a in rollout.STATE_AXES]


def run_batch(cfg: dict, slots, points, seed: int, threads: int = 1):
    torch.set_num_threads(threads)
    n = len(slots)
    names = list(cfg["axes"])
    assignments = {name: np.array([points[p][name] for p, _ in slots], dtype=float) for name in names}
    mode = cfg["mode"]
    if mode == "policy":
        spec = cfg["policy"]
        task = spec["task"]
        env, wrapped = rollout.make_env(task, n, seed, robot=cfg.get("robot", "plain"))
        policy = rollout.load_policy(task, spec["checkpoint"], wrapped)
        rollout.force_spawn(env, spec.get("start", "hold"))
        obs = rollout.start(env, wrapped, assignments)
        before = sweep_axes.read_all(env)
        out = rollout.run_policy(env, wrapped, policy, obs, spec.get("goal") or rollout.default_goal(task),
                                 steps=spec.get("steps", 298), flag=spec.get("flag"))
    elif mode == "routine":
        spec = cfg["routine"]
        env, wrapped = rollout.make_env(rollout.ROUTINE_TASK["legs_together"], n, seed,
                                        episode_s=spec["seconds"] + 1.0, robot=cfg.get("robot", "plain"))
        stages = rollout.routine_stages(env, wrapped, spec)
        term = env.event_manager.get_term_cfg("set_headstand_spawn")
        term.params.update(standing_prob=1.0, partway_prob=0.0, hold_prob=0.0, pike_prob=0.0, handover_prob=0.0)
        obs = rollout.start(env, wrapped, assignments)
        rollout.assert_standing_starts(env, env.scene["robot"])
        before = sweep_axes.read_all(env)
        out = rollout.run_routine(env, wrapped, stages, obs, seconds=spec["seconds"])
        out["stage_names"] = [s[0] for s in stages]
    elif mode == "handover":
        spec = cfg["routine"]
        env, wrapped = rollout.make_env(rollout.ROUTINE_TASK["legs_together"], n, seed,
                                        episode_s=cfg["stage_seconds"] + 1.0)
        stages = rollout.routine_stages(env, wrapped, spec)
        names = [s_[0] for s_ in stages]
        k = names.index(cfg["stage"])
        source = np.load(Path(cfg["source"]) / "states.npz")
        valid = np.where(~np.isnan(source["handover_qpos"][:, k - 1, 0]))[0]
        order = np.random.default_rng(cfg["seed"]).permutation(valid)
        rows = np.array([order[rep % len(order)] for _, rep in slots])
        state_names = [a for a in names_of_state_axes(cfg)]
        q0, v0 = source["handover_qpos"][rows, k - 1], source["handover_qvel"][rows, k - 1]
        q, v = rollout.perturb_states(q0, v0, {a: assignments[a] for a in state_names}, seed)
        physics = {a: vals for a, vals in assignments.items() if a not in state_names}
        obs = rollout.start(env, wrapped, physics, start_states=(q, v))
        before = sweep_axes.read_all(env)
        _, policy, goal, hold = stages[k]
        out = rollout.run_stage(env, wrapped, policy, obs, goal, hold, cfg["stage_seconds"])
        out["source_attempt"] = rows
    else:
        raise ValueError(f"unknown mode {mode!r}")
    after = sweep_axes.read_all(env)
    env.close()
    return assignments, before, after, out, rollout.checkpoint_hashes()


def batch_records(cfg, b, slots, offset, result):
    """One record per attempt of batch b, and its states keyed by array name."""
    assignments, before, after, out, _ = result
    seed = cfg["seed"] + b
    records, states = [], {}
    for i, (point, rep) in enumerate(slots):
        rec = {
            "attempt": offset + i, "batch": b, "env": i, "seed": seed,
            "point": point, "repeat": rep,
            "pinned": {n: float(assignments[n][i]) for n in cfg["axes"]},
            "params_start": {n: float(v[i]) for n, v in before.items()},
            "params_end": {n: float(v[i]) for n, v in after.items()},
            "success": bool(out["success"][i]),
            "resets_in_batch": out["resets"],
        }
        if cfg["mode"] == "policy":
            rec["first_goal_s"] = float(out["first_goal_s"][i])
        elif cfg["mode"] == "handover":
            rec["stage"] = cfg["stage"]
            rec["done_s"] = float(out["done_s"][i])
            rec["source_attempt"] = int(out["source_attempt"][i])
        else:
            names = out["stage_names"]
            k = int(out["stage"][i])
            rec.update({
                "completed_stages": k,
                "first_unfinished_stage": names[k] if k < len(names) else None,
                "final_standing": bool(out["final_standing"][i]),
                "stage_times_s": [float(t) for t in out["stage_times_s"][i]],
            })
        records.append(rec)
    for key in ("final_qpos", "final_qvel", "handover_qpos", "handover_qvel"):
        if key in out:
            states[key] = out[key]
    if "stage_names" in out:
        states["stage_names"] = np.array(out["stage_names"])
    return records, states


def write_batch(out_dir: Path, cfg, b, slots, offset, result, elapsed_s: float):
    records, states = batch_records(cfg, b, slots, offset, result)
    d = out_dir / "batches"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"batch_{b:04d}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    np.savez_compressed(d / f"batch_{b:04d}.npz", **states)
    (d / f"batch_{b:04d}.json").write_text(json.dumps(
        {"sweep": cfg["name"], "batch": b, "attempts": len(records),
         "checkpoint_sha256": result[4], "elapsed_s": round(elapsed_s, 1)}) + "\n")
    return records


def run_one(cfg, b, slots, points, offset, threads, out_dir):
    t0 = time.time()
    result = run_batch(cfg, slots, points, cfg["seed"] + b, threads)
    return write_batch(out_dir, cfg, b, slots, offset, result, time.time() - t0)


def merge(cfg, out_dir: Path, n_batches: int, workers: int, wall_s: float | None = None):
    """Combine the batch files into records.jsonl, states.npz and provenance.json."""
    d = out_dir / "batches"
    records, states, hashes, sim_s = [], {}, {}, 0.0
    expected = [slot for batch in batches_for(cfg)[1] for slot in batch]
    stage_names = None
    for b in range(n_batches):
        records += [json.loads(line) for line in (d / f"batch_{b:04d}.jsonl").read_text().splitlines()]
        with np.load(d / f"batch_{b:04d}.npz") as z:
            for key in z.files:
                if key == "stage_names":
                    stage_names = z[key]
                else:
                    states.setdefault(key, []).append(z[key])
        meta = json.loads((d / f"batch_{b:04d}.json").read_text())
        if meta.get("sweep") != cfg["name"] or meta.get("batch") != b:
            raise ValueError(f"batch_{b:04d} belongs to sweep {meta.get('sweep')!r} batch {meta.get('batch')}, "
                             f"not {cfg['name']!r} batch {b}")
        hashes.update(meta["checkpoint_sha256"])
        sim_s += meta["elapsed_s"]
    got = [(r["point"], r["repeat"]) for r in records]   # point is the grid point's index
    if got != expected:
        raise ValueError(f"{cfg['name']}: the records do not match the config's grid and attempts")
    with open(out_dir / "records.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    arrays = {k: np.concatenate(v) for k, v in states.items()}
    if stage_names is not None:
        arrays["stage_names"] = stage_names
    np.savez_compressed(out_dir / "states.npz", **arrays)
    training_root = Path.cwd()
    provenance = {
        "config": cfg,
        "checkpoint_sha256": hashes,
        "tool_sha256": {f.name: rollout.sha256(f) for f in sorted(SWEEP_DIR.glob("*.py"))},
        "evaluator_sha256": {name: rollout.sha256(SWEEP_DIR.parent / name)
                             for name in ("routine_full.py", "eval_checkpoint.py")},
        "training_revision": revision(training_root), "training_dirty": dirty(training_root),
        "tools_revision": revision(SWEEP_DIR), "tools_dirty": dirty(SWEEP_DIR),
        "versions": {n: importlib.metadata.version(n) for n in ("mjlab", "mujoco", "torch", "warp-lang")},
        "axes": {**{n: {"unit": a.unit, "nominal": a.nominal, "training_range": a.training_range, "note": a.note}
                    for n, a in sweep_axes.AXES.items()},
                 **{n: {"unit": unit, "nominal": 0.0, "training_range": [0.0, 0.0],
                        "note": (fn.__doc__ or "").strip().replace("\n", " ")}
                    for n, (unit, fn) in rollout.STATE_AXES.items()}},
        "elapsed_s": round(sim_s, 1),
        "wall_s": None if wall_s is None else round(wall_s, 1),
        "workers": workers,
    }
    if cfg["mode"] in ("routine", "handover"):
        provenance["standing_onnx_sha256"] = rollout.sha256(cfg["routine"]["stand"])
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {len(records)} records to {out_dir}, {sum(r['success'] for r in records)} successes")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, help="output directory; defaults to results/sweeps/<name>")
    p.add_argument("--workers", type=int, default=1, help="batches run in parallel processes")
    p.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    p.add_argument("--batch", type=int, help="run only this batch and write its files")
    p.add_argument("--merge", action="store_true", help="only merge existing batch files")
    p.add_argument("--workers-recorded", type=int, help="worker count to record when merging a Ray run")
    p.add_argument("--wall-s", type=float, help="wall time to record when merging a Ray run")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    known = {**sweep_axes.AXES, **rollout.STATE_AXES}
    for name in cfg["axes"]:
        if name not in known:
            p.error(f"unknown axis {name!r}; known: {', '.join(known)}")
        if name in rollout.STATE_AXES and cfg["mode"] != "handover":
            p.error(f"{name} perturbs handover states and needs mode \"handover\"")
    out_dir = args.out or RESULTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    points, batches = batches_for(cfg)
    offsets = [sum(len(x) for x in batches[:b]) for b in range(len(batches))]
    if args.merge:
        merge(cfg, out_dir, len(batches), args.workers_recorded or args.workers, args.wall_s)
        return
    if args.batch is not None:
        b = args.batch
        records = run_one(cfg, b, batches[b], points, offsets[b], args.threads, out_dir)
        print(f"batch {b}: {sum(r['success'] for r in records)}/{len(records)} successes", flush=True)
        return
    print(f"{cfg['name']}: {len(points)} grid points x {cfg['attempts']} attempts = "
          f"{len(points) * cfg['attempts']} attempts in {len(batches)} batch(es)", flush=True)
    t0 = time.time()
    jobs = [(cfg, b, slots, points, offsets[b], args.threads, out_dir) for b, slots in enumerate(batches)]
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
        results = [pool.submit(run_one, *job) for job in jobs]
        for b, f in enumerate(results):
            recs = f.result()
            print(f"  batch {b + 1}/{len(batches)}: {sum(r['success'] for r in recs)}/{len(recs)} successes, "
                  f"{time.time() - t0:.0f} s", flush=True)
    else:
        for job in jobs:
            recs = run_one(*job)
            print(f"  batch {job[1] + 1}/{len(batches)}: {sum(r['success'] for r in recs)}/{len(recs)} successes, "
                  f"{time.time() - t0:.0f} s", flush=True)
    merge(cfg, out_dir, len(batches), args.workers, time.time() - t0)


if __name__ == "__main__":
    main()
