"""Watch a set of wandb runs, eval each checkpoint as it lands, drop the videos on the Desktop.

    uv run scripts/headstand/watch_variants.py --after "2026-09-20T13:00" --names A,B,C,D --tag run4 --checkpoints 250,500,1000

Runs created after --after are matched to --names in creation order (the
variants are submitted 15 s apart in that order). Each (run, checkpoint) is
evaluated once with eval_checkpoint.py; the standing-start video is copied
to ~/Desktop/microduck/variant<name>_<tag>_iter<N>_standing_start.mp4 and the
result lines are appended to renders/<tag>_results.txt.
"""
import argparse, os, re, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import wandb

p = argparse.ArgumentParser()
p.add_argument("--after", required=True, help="ISO time (local) the variants were submitted")
p.add_argument("--names", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--checkpoints", default="250,500,1000")
p.add_argument("--episodes", type=int, default=16)
p.add_argument("--hours", type=float, default=3.0)
args = p.parse_args()

names = args.names.split(",")
ckpts = [f"model_{c}.pt" for c in args.checkpoints.split(",")]
after = datetime.fromisoformat(args.after).astimezone(timezone.utc)
api = wandb.Api()
done = set()
results = Path(f"renders/{args.tag}_results.txt")
desktop = Path.home() / "Desktop" / "microduck"
desktop.mkdir(parents=True, exist_ok=True)
deadline = time.time() + args.hours * 3600

def newer_runs():
    runs = [r for r in api.runs("zachgarner-ai/mjlab_microduck", order="+created_at")
            if datetime.fromisoformat(r.created_at.replace("Z", "+00:00")).astimezone(timezone.utc) > after]
    return runs[: len(names)]

while time.time() < deadline:
    runs = newer_runs()
    for name, r in zip(names, runs):
        files = {f.name for f in r.files()}
        for ck in ckpts:
            key = (r.id, ck)
            if key in done or ck not in files:
                continue
            n = re.search(r"model_(\d+)", ck).group(1)
            video = f"renders/{args.tag}_{name}_{n}.mp4"
            out = subprocess.run(
                ["uv", "run", "scripts/headstand/eval_checkpoint.py", "--wandb-run-path",
                 f"zachgarner-ai/mjlab_microduck/{r.id}", "--checkpoint", ck, "--episodes", str(args.episodes), "--video", video],
                capture_output=True, text=True, env={**os.environ, "WANDB_MODE": "offline"})
            lines = [l for l in out.stdout.splitlines() if re.match(r"^(standing|partway|hold) ", l)]
            block = f"== variant {name} ({r.id}) iteration {n}\n" + "\n".join(lines) + "\n"
            print(block, flush=True)
            with results.open("a") as f:
                f.write(block)
            if Path(video).exists():
                shutil.copy(video, desktop / f"variant{name}_{args.tag}_iter{n}_standing_start.mp4")
            done.add(key)
    if len(done) >= len(names) * len(ckpts):
        break
    time.sleep(90)
print("WATCH DONE", len(done), "evals")
