"""Spread a sweep's batches over a Ray cluster, one batch per CPU.

    cd microduck_rl                      # on the Anyscale workspace
    python ../tools/sweep/ray_sweep.py ../tools/sweep/configs/<config>.json [more configs ...]

Run this with the cluster's own Python, which has Ray. Each Ray task needs one
CPU and runs `run_sweep.py --batch B` in the training environment on its node:
the task syncs that environment with uv and downloads the evaluated
checkpoints the first time a node runs a batch, under a file lock so tasks
sharing a node wait for one another. The batch files come back to the driver,
and `run_sweep.py --merge` combines them in each sweep's output directory.
Several configs queue together, so one call can run a whole family of sweeps.

Warp's CPU backend is the verified evaluators' backend, so results stay
comparable with `results/verified/`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import ray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import batches_for  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# The node's training environment lives outside the per-job working directory,
# so every job on a node reuses it.
SETUP = """
set -e
export UV_PROJECT_ENVIRONMENT=$HOME/.microduck-venv UV_HTTP_TIMEOUT=600
cd microduck_rl
flock $HOME/.microduck-setup.lock sh -c 'uv sync -q --frozen && uv run -q ../tools/download_policies.py > /dev/null'
"""


@ray.remote(num_cpus=1)
def run_batch_task(config_text: str, b: int) -> dict:
    """Run batch b in the working directory Ray shipped, return its files."""
    out = Path("microduck_rl") / "_ray_batch_out"
    cfg_path = Path("microduck_rl") / f"_ray_config_{b}.json"
    cfg_path.write_text(config_text)
    cmd = SETUP + f"uv run -q ../tools/sweep/run_sweep.py {cfg_path.name} --batch {b} --out {out.name} --threads 1\n"
    t0 = time.time()
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"batch {b} failed on {ray.util.get_node_ip_address()}:\n{proc.stderr[-4000:]}")
    d = out / "batches"
    return {
        "b": b, "node": ray.util.get_node_ip_address(), "seconds": time.time() - t0,
        "files": {p.name: p.read_bytes() for p in d.glob(f"batch_{b:04d}.*")},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", type=Path, nargs="+")
    p.add_argument("--out-root", type=Path, default=REPO / "results" / "sweeps",
                   help="each sweep writes to <out-root>/<name>")
    args = p.parse_args()
    sweeps = []
    for path in args.configs:
        cfg = json.loads(path.read_text())
        out_dir = (args.out_root / cfg["name"]).resolve()
        (out_dir / "batches").mkdir(parents=True, exist_ok=True)
        sweeps.append({"path": path.resolve(), "cfg": cfg, "text": path.read_text(), "out": out_dir,
                       "n": len(batches_for(cfg)[1]), "left": len(batches_for(cfg)[1]), "t0": time.time()})
    ray.init(runtime_env={
        "working_dir": str(REPO),
        "excludes": [".git", "**/.venv", "results/policies", "results/routine", "results/strips",
                     "**/logs/rsl_rl", "**/wandb", "**/*.mp4", "**/_ray_*", "results/sweeps/*/batches"],
    })
    total = sum(s_["n"] for s_ in sweeps)
    print(f"{len(sweeps)} sweep(s), {total} batches, {ray.cluster_resources().get('CPU', 0):.0f} CPUs now; "
          "the autoscaler adds nodes as tasks queue", flush=True)
    t0 = time.time()
    pending = {}
    for i, s_ in enumerate(sweeps):
        for b in range(s_["n"]):
            pending[run_batch_task.remote(s_["text"], b)] = i
    nodes, done = set(), 0
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        i = pending.pop(ready[0])
        s_ = sweeps[i]
        result = ray.get(ready[0])
        for name, data in result["files"].items():
            (s_["out"] / "batches" / name).write_bytes(data)
        nodes.add(result["node"])
        done += 1
        s_["left"] -= 1
        print(f"  {s_['cfg']['name']} batch {result['b']} in {result['seconds']:.0f} s on {result['node']} "
              f"({done}/{total}, {time.time() - t0:.0f} s)", flush=True)
        if s_["left"] == 0:
            subprocess.run(["bash", "-c", "export UV_PROJECT_ENVIRONMENT=$HOME/.microduck-venv; "
                            f"uv run -q ../tools/sweep/run_sweep.py {s_['path']} --merge --out {s_['out']} "
                            f"--workers-recorded {s_['n']} --wall-s {time.time() - s_['t0']:.1f}"],
                           check=True, cwd=REPO / "microduck_rl")
    print(f"{total} batches on {len(nodes)} node(s) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
