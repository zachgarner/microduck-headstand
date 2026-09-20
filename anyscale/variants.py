"""Submit several headstand training variants to Anyscale at once.

Each variant is the same task with different env-var knobs (read by
microduck_headstand_env_cfg.py and mdp.py) and its own GPU instance. Zach,
Sep 20 2026: "Anyscale does not care what size or how many machines i'm
using at all", so variants run in parallel instead of in sequence.

    uv run scripts/anyscale/variants.py --iterations 1000            # submit every variant below
    uv run scripts/anyscale/variants.py --only A,C --instance g6.2xlarge
    uv run scripts/anyscale/variants.py --dry-run

The wandb key comes from ~/.netrc (wandb login). Checkpoints land in the
artifact bucket under microduck/<run-name>/logs/ and in each wandb run.
"""

import argparse
import netrc
import subprocess
import tempfile
from pathlib import Path

VARIANTS = {
    # name: (knobs, one-line reason)
    "A": ({"HEADSTAND_SLAM_N": "12", "HEADSTAND_PARK_TAX": "-0.5"},
          "run 2's config plus the spawn grace window: the baseline"),
    "B": ({"HEADSTAND_SLAM_N": "18", "HEADSTAND_PARK_TAX": "-0.5"},
          "looser slam gate: 12 N may forbid every real landing"),
    "C": ({"HEADSTAND_SLAM_N": "18", "HEADSTAND_PARK_TAX": "-1.5", "HEADSTAND_PROGRESS_W": "1.0"},
          "heavy park tax, half the pay for reaching horizontal"),
    "D": ({"HEADSTAND_SLAM_N": "12", "HEADSTAND_PARK_TAX": "-1.5", "HEADSTAND_STANDING_P0": "0.15"},
          "heavy tax, 12 N gate, normal spawn mix (40% standing starts fell backward in run 4)"),
}

JOB_TEMPLATE = """name: microduck-headstand-{name}
entrypoint: bash scripts/anyscale/train.sh Mjlab-Headstand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations {iterations}
image_uri: anyscale/image/microduck-train:1
compute_config:
  cloud: aws-public-us-west-2
  head_node:
    instance_type: {instance}
  worker_nodes: []
excludes: [.git, .venv, logs, wandb, policies, __pycache__, "*.onnx", renders]
env_vars:
  RUN_NAME: headstand-{name}
max_retries: 0
timeout_s: 43200
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None, help="comma-separated variant names")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--instance", default="g6e.xlarge", help="g6e.xlarge = L40S, g6.2xlarge = L4")
    p.add_argument("--tag", default="", help="suffix for the run names, e.g. run3")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    key = netrc.netrc().authenticators("api.wandb.ai")[2]
    names = args.only.split(",") if args.only else list(VARIANTS)
    for name in names:
        knobs, reason = VARIANTS[name]
        run = f"{name}{('-' + args.tag) if args.tag else ''}".lower()
        yaml = JOB_TEMPLATE.format(name=run, iterations=args.iterations, instance=args.instance)
        env = [f"WANDB_API_KEY={key}"] + [f"{k}={v}" for k, v in knobs.items()]
        print(f"[{name}] {reason}\n     knobs {knobs} on {args.instance}, {args.iterations} iterations")
        if args.dry_run:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=f"-{run}.yaml", delete=False, dir=Path("scripts/anyscale")) as f:
            f.write(yaml)
            path = f.name
        cmd = ["anyscale", "job", "submit", "-f", path, "--working-dir", "."]
        for e in env:
            cmd += ["--env", e]
        out = subprocess.run(cmd, capture_output=True, text=True)
        Path(path).unlink()
        line = [l for l in (out.stdout + out.stderr).splitlines() if "submitted" in l or "rror" in l]
        print("     " + (line[-1] if line else out.stderr[-300:]))


if __name__ == "__main__":
    main()
