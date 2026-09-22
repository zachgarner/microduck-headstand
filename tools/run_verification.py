"""Run the documented checkpoint battery from the microduck_rl directory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routine-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--individual", action="store_true", help="also evaluate each policy at seed 0")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    tools = Path(__file__).resolve().parent
    output = tools.parent / "results" / "verified"
    output.mkdir(parents=True, exist_ok=True)
    common = ["--fold", "vorty4kb:model_1000.pt", "--legs-together", "gx4v9lcz:model_1499.pt",
              "--roll", "ax1vgv8z:model_1999.pt", "--split", "y2fllvgj:model_1499.pt",
              "--switch", "whv1lcu7:model_500.pt", "--switch-hold-s", "1", "--single-switch",
              "--splitover", "2v46kg6g:model_1499.pt", "--stand", "policies/pollen/alpha_stand.onnx",
              "--hold-s", "2", "--settle-s", "1", "--pike-settle-s", "0", "--episodes", "32", "--seconds", "18"]
    jobs = [(f"seed-{seed}", "routine_full.py", common + ["--seed", str(seed)], {})
            for seed in args.routine_seeds]
    if args.individual:
        for name, task, run, checkpoint, bucket in [
            ("fold", "Fold", "vorty4kb", "1000", "standing"),
            ("legs-together", "KickupLegsTogether", "gx4v9lcz", "1499", "handover"),
            ("split", "Kickup", "y2fllvgj", "1499", "handover"),
            ("backroll", "BackrollLegsTogether", "ax1vgv8z", "1999", "hold"),
            ("splitover", "SplitOver", "2v46kg6g", "1499", "hold"),
        ]:
            jobs.append((name, "check32.py", ["--run", run, "--checkpoint", f"model_{checkpoint}.pt",
                         "--bucket", bucket, "--episodes", "32", "--seed", "0"],
                         {"HEADSTAND_TASK": f"Mjlab-Headstand{task}-Flat-MicroDuck"}))
        jobs.append(("switch", "check_switch.py", ["--run", "whv1lcu7", "--checkpoint", "model_500.pt",
                     "--episodes", "32", "--seed", "0"], {}))

    def run(job):
        name, script, options, extra_env = job
        command = [sys.executable, "-u", str(tools / script), *options, "--json", str(output / f"{name}.json")]
        print(f"Starting {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **extra_env})
        print(f"Finished {name}: exit {result.returncode}", flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    if any(codes):
        raise SystemExit("One or more evaluations failed; inspect results/verified/*.log")


if __name__ == "__main__":
    main()
