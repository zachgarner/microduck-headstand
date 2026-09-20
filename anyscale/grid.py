"""Submit the headstand knob grid as ONE multi-GPU Anyscale job, one variant per GPU.

The grid is slam gate {12, 18} N × park tax {-0.5, -1.5} × stage-0 standing
share {0.15, 0.4}: eight variants, eight GPUs, one node. AWS has no single-GPU
A100/H100 node, so the big GPU comes eight at a time and the grid fills it.
Zach, Sep 20 2026: "Start with an A100 even those are hard" — p4d is the
default; availability is the only limit.

    uv run scripts/anyscale/grid.py --iterations 1000                          # 8x A100 (p4d.24xlarge)
    uv run scripts/anyscale/grid.py --instance p5.48xlarge --iterations 1000   # 8x H100, if AWS has one
    uv run scripts/anyscale/grid.py --dry-run
"""

import argparse
import itertools
import json
import netrc
import subprocess
import tempfile
from pathlib import Path

JOB_TEMPLATE = """name: microduck-headstand-grid
entrypoint: bash scripts/anyscale/train_multi.sh Mjlab-Headstand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations {iterations}
image_uri: anyscale/image/microduck-train:1
compute_config:
  cloud: aws-public-us-west-2
  head_node:
    instance_type: {instance}
  worker_nodes: []
excludes: [.git, .venv, logs, wandb, policies, __pycache__, "*.onnx", renders]
env_vars:
  RUN_NAME: headstand-grid-{tag}
max_retries: 0
timeout_s: 43200
"""


def grid():
    out = []
    for slam, tax, p0 in itertools.product(("12", "18"), ("-0.5", "-1.5"), ("0.15", "0.4")):
        name = f"g-s{slam}-t{tax.lstrip('-').replace('.', '')}-p{p0.replace('0.', '')}"
        out.append({"name": name, "env": {"HEADSTAND_SLAM_N": slam, "HEADSTAND_PARK_TAX": tax, "HEADSTAND_STANDING_P0": p0}})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default="p4d.24xlarge")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--tag", default="1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    variants = grid()
    for v in variants:
        print(v["name"], v["env"])
    if args.dry_run:
        return
    key = netrc.netrc().authenticators("api.wandb.ai")[2]
    yaml = JOB_TEMPLATE.format(iterations=args.iterations, instance=args.instance, tag=args.tag)
    with tempfile.NamedTemporaryFile("w", suffix="-grid.yaml", delete=False, dir=Path("scripts/anyscale")) as f:
        f.write(yaml)
        path = f.name
    cmd = ["anyscale", "job", "submit", "-f", path, "--working-dir", ".",
           "--env", f"WANDB_API_KEY={key}", "--env", f"VARIANTS_JSON={json.dumps(variants)}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    Path(path).unlink()
    line = [l for l in (out.stdout + out.stderr).splitlines() if "submitted" in l or "rror" in l]
    print(line[-1] if line else out.stderr[-400:])


if __name__ == "__main__":
    main()
