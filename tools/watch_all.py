"""Overnight watcher: evaluate every new checkpoint of a set of runs with check32 as it lands.

    uv run scripts/headstand/watch_all.py --spec specs.json --hours 6

specs.json: [{"run": "<wandb id>" | {"after": "<iso utc>", "name_contains": "..."}, "task": "...", "bucket": "standing|tripod", "tag": "fold2"}]
A run given by {after, name_contains} is resolved to the first wandb run created after that time whose name contains the text.
Results append to renders/overnight_results.txt; videos land in ~/Desktop/microduck/<tag>_iter<N>_<bucket>_start.mp4.
"""
import argparse, json, os, re, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import wandb

p = argparse.ArgumentParser()
p.add_argument("--spec", required=True); p.add_argument("--hours", type=float, default=6.0)
args = p.parse_args()
specs = json.load(open(args.spec))
api = wandb.Api()
done = set(); results = Path("renders/overnight_results.txt"); desktop = Path.home() / "Desktop" / "microduck"
deadline = time.time() + args.hours * 3600

def resolve(spec):
    if isinstance(spec["run"], str):
        return spec["run"]
    after = datetime.fromisoformat(spec["run"]["after"]).astimezone(timezone.utc)
    for r in api.runs("zachgarner-ai/mjlab_microduck", order="+created_at"):
        created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00")).astimezone(timezone.utc)
        match = r.name.endswith(spec["run"]["name_endswith"]) if "name_endswith" in spec["run"] else spec["run"]["name_contains"] in r.name
        if created > after and match:
            spec["run"] = r.id; return r.id
    return None

while time.time() < deadline:
    for spec in specs:
        rid = resolve(spec)
        if rid is None: continue
        try:
            r = api.run(f"zachgarner-ai/mjlab_microduck/runs/{rid}")
            files = sorted({f.name for f in r.files() if re.match(r"model_\d+\.pt", f.name)}, key=lambda x: int(re.search(r"\d+", x).group()))
        except Exception as e:
            print("wandb error", rid, e, flush=True); continue
        for ck in files:
            n = int(re.search(r"\d+", ck).group())
            if (rid, ck) in done or n == 0: continue
            video = f"renders/{spec['tag']}_{n}.mp4"
            out = subprocess.run(["uv", "run", "scripts/headstand/check32.py", "--run", rid, "--checkpoint", ck, "--bucket", spec["bucket"], "--video", video],
                                 capture_output=True, text=True, env={**os.environ, "WANDB_MODE": "offline", "HEADSTAND_TASK": spec["task"]})
            lines = [l for l in out.stdout.splitlines() if l.startswith("  ") or l.startswith(rid)]
            block = f"== {spec['tag']} ({rid}) {ck} [{time.strftime('%H:%M')}]\n" + "\n".join(lines) + "\n"
            print(block, flush=True)
            with results.open("a") as f: f.write(block)
            if Path(video).exists():
                shutil.copy(video, desktop / f"{spec['tag']}_iter{n}_{'standing' if spec['bucket']=='standing' else 'pike'}_start.mp4")
            done.add((rid, ck))
    time.sleep(180)
print("WATCH_ALL DONE", flush=True)
