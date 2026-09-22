"""The 32-episode, no-resets check of a headstand checkpoint from one start.

    HEADSTAND_TASK=Mjlab-HeadstandKickup-Flat-MicroDuck uv run ../tools/check32.py --run 076n5wpa --checkpoint model_999.pt --bucket tripod --video out.mp4

Terminations are disabled so nothing can reset inside the rollout (the
mechanism that faked three "successes" on Sep 20 2026). Reports end states,
time to reach the headstand angle, and peak head force. Records env 0.
"""
import argparse, hashlib, json, math, os, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "tools"))
from dataclasses import asdict
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp as m
from eval_checkpoint import TASK, contacts, floor_bodies, force_spawn, classify, in_headstand, in_pike, is_standing, in_shaped_headstand

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--checkpoint", required=True)
p.add_argument("--bucket", default="tripod"); p.add_argument("--episodes", type=int, default=32)
p.add_argument("--video", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--json", type=Path)
args = p.parse_args()
torch.set_num_threads(4)
exp = load_rl_cfg(TASK).experiment_name
ck = Path("logs/rsl_rl") / exp / "wandb_checkpoints" / args.run / args.checkpoint
if not ck.exists():
    import wandb
    wandb.Api().run(f"zachgarner-ai/mjlab_microduck/runs/{args.run}").file(args.checkpoint).download(str(ck.parent), replace=True)
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = args.episodes; cfg.curriculum.clear()
cfg.seed = args.seed
cfg.auto_reset = False
for n in list(cfg.terminations):
    if n != "time_out":
        del cfg.terminations[n]
cfg.viewer.distance = 0.6; cfg.viewer.elevation = -8; cfg.viewer.azimuth = 135; cfg.viewer.height = 480; cfg.viewer.width = 640
env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array" if args.video else None)
w = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
runner = (load_runner_cls(TASK) or OnPolicyRunner)(w, asdict(load_rl_cfg(TASK)), device="cpu"); runner.load(str(ck), map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
force_spawn(env, args.bucket); obs, _ = w.reset()
asset = env.scene["robot"]; N = args.episodes
# The roll tasks' head sensor has no force field, so head force is not
# measured there; say so rather than reporting a zero.
force_available = (m._HEADSTAND_HEAD_SENSOR in env.scene.sensors
                   and env.scene.sensors[m._HEADSTAND_HEAD_SENSOR].data.force is not None)
style = "legs_together" if "LegsTogether" in TASK else "split"
def goal():
    if "Fold" in TASK:
        return in_pike(env, asset)
    if "Backroll" in TASK or "SplitOver" in TASK:
        return is_standing(env, asset)
    return in_shaped_headstand(env, asset, style)
t_up = np.full(N, -1.0); peak = torch.zeros(N); frames = []
with torch.no_grad():
    for i in range(298):
        obs, *_ = w.step(policy(obs))
        inv = m._inverted_cos(asset).numpy()
        first = (t_up < 0) & goal(); t_up[first] = (i + 1) * env.step_dt
        if force_available:
            peak = torch.maximum(peak, m._head_floor_force(env).cpu())
        if args.video:
            frames.append(env.render())
inv = m._inverted_cos(asset).numpy(); nose = m._nose_up(asset).numpy(); touching = floor_bodies(env)
labels = [classify(float(inv[i]), touching[i], float(nose[i])) for i in range(N)]
# The gates, not the angle: "headstand" needs the head alone carrying it, "pike"
# needs head and BOTH feet down. Anything inverted but propped keeps its label.
pike, head_alone, standing = in_pike(env, asset), in_headstand(env, asset), is_standing(env, asset)
labels = ["pike" if pike[i] else ("headstand" if head_alone[i] else ("standing" if standing[i] else labels[i])) for i in range(N)]
labels = ["other" if l == "standing" and not standing[i] else l for i, l in enumerate(labels)]
print(f"{args.run} {args.checkpoint} from {args.bucket}, {N} episodes, resets inside rollout: {int((env.episode_length_buf < 298).sum())}")
print("  end states:", {k: labels.count(k) for k in sorted(set(labels))})
reached = t_up[t_up >= 0]
print(f"  reached the task goal: {len(reached)}/{N}; time median {np.median(reached) if len(reached) else float('nan'):.2f} s, max {reached.max() if len(reached) else float('nan'):.2f} s")
print(f"  peak head force: median {float(peak.median()):.1f} N, max {float(peak.max()):.1f} N" if force_available
      else "  peak head force: not measured (this task's head sensor has no force field)")
print(f"  standing upright on both feet at the end: {int(standing.sum())}/{N}   (the exits' success)")
if args.video:
    imageio.mimwrite(args.video, frames, fps=50, quality=8); print("  video:", args.video)

success = goal()
print(f"  strict task success: {int(success.sum())}/{N}")
if args.json:
    reached = t_up[t_up >= 0]
    report = {"task": TASK, "run": args.run, "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
        "seed": args.seed, "bucket": args.bucket, "episodes": N, "steps": 298,
        "seconds": 298 * env.step_dt, "successes": int(success.sum()),
        "median_time_to_goal_s": float(np.median(reached)) if len(reached) else None,
        "sampled_head_force_median_n": float(peak.median()) if force_available else None,
        "sampled_head_force_max_n": float(peak.max()) if force_available else None,
        "reset_count": int((env.episode_length_buf < 298).sum()),
        "tool_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in ("check32.py", "eval_checkpoint.py")},
        "per_episode": [{"success": bool(success[j]), "first_goal_s": float(t_up[j])} for j in range(N)]}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
env.close()
