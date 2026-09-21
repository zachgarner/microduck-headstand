"""Submit one training run to Anyscale.

    uv run anyscale/submit.py --task Mjlab-HeadstandKickup-Flat-MicroDuck --name kickup-4 --iterations 1500 \
        --warm-start 076n5wpa:model_999.pt --env HEADSTAND_OMEGA_MAX=2.0 --env HEADSTAND_BANK_PROB=0.5

One L40S node (g6e.xlarge) per run, about 2.1 s per iteration at 4,096 envs.
The wandb key comes from ~/.netrc (wandb login) and is passed with --env on
the command line, because this anyscale CLI (0.26) does not substitute
${VAR} placeholders inside the YAML. Checkpoints land in the wandb run and in
the artifact bucket under microduck/<name>/logs/. runs.md lists the runs of record.
"""
import argparse, netrc, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--name", required=True, help="run name, e.g. kickup-split-chain-1")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--warm-start", default=None, help="wandb run:checkpoint to start from, e.g. 076n5wpa:model_999.pt")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE knob read by the env cfg (repeatable)")
    p.add_argument("--instance", default="g6e.xlarge")
    p.add_argument("--working-dir", default=".", help="this repo's root; the job cds into microduck_rl/")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    entry = f"cd microduck_rl && bash ../anyscale/train.sh {a.task} --env.scene.num-envs {a.envs} --agent.max_iterations {a.iterations}"
    env = {"RUN_NAME": f"headstand-{a.name}"}
    if a.warm_start:
        run, ck = a.warm_start.split(":")
        entry += f" --agent.resume True --wandb-run-path zachgarner-ai/mjlab_microduck/{run} --wandb-checkpoint-name {ck}"
        env["MICRODUCK_WARM_START"] = "1"   # restart the curricula at 0; keep the weights
    for kv in a.env:
        k, v = kv.split("=", 1); env[k] = v
    yaml = (HERE / "job-template.yaml").read_text()
    yaml = yaml.replace("__NAME__", f"microduck-headstand-{a.name}").replace("__ENTRYPOINT__", entry).replace("__INSTANCE__", a.instance)
    yaml = yaml.replace("__ENV_VARS__", "\n".join(f"  {k}: \"{v}\"" for k, v in env.items()))
    print(yaml)
    if a.dry_run:
        return
    key = netrc.netrc().authenticators("api.wandb.ai")[2]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml); path = f.name
    cmd = ["anyscale", "job", "submit", "-f", path, "--working-dir", a.working_dir, "--env", f"WANDB_API_KEY={key}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    Path(path).unlink()
    print("\n".join(l for l in (out.stdout + out.stderr).splitlines() if "submitted" in l or "rror" in l).replace(key, "***"))


if __name__ == "__main__":
    main()
