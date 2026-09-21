"""The 32-episode, no-resets check of a headstand checkpoint from one start.

    HEADSTAND_TASK=Mjlab-HeadstandKickup-Flat-MicroDuck uv run scripts/headstand/check32.py --run 076n5wpa --checkpoint model_999.pt --bucket tripod --video out.mp4

Terminations are disabled so nothing can reset inside the rollout (the
mechanism that faked three "successes" on Sep 20 2026). Reports end states,
time to reach the headstand angle, and peak head force. Records env 0.
"""
import argparse, math, os, sys
sys.path.insert(0, "scripts/headstand")
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
from eval_checkpoint import TASK, force_spawn, floor_bodies, classify

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--checkpoint", required=True)
p.add_argument("--bucket", default="tripod"); p.add_argument("--episodes", type=int, default=32)
p.add_argument("--video", default=None)
args = p.parse_args()
exp = load_rl_cfg(TASK).experiment_name
ck = Path("logs/rsl_rl") / exp / "wandb_checkpoints" / args.run / args.checkpoint
if not ck.exists():
    import wandb
    wandb.Api().run(f"zachgarner-ai/mjlab_microduck/runs/{args.run}").file(args.checkpoint).download(str(ck.parent), replace=True)
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = args.episodes; cfg.curriculum.clear()
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
t_up = np.full(N, -1.0); peak = torch.zeros(N); frames = []
with torch.no_grad():
    for i in range(298):
        obs, *_ = w.step(policy(obs))
        inv = m._inverted_cos(asset).numpy()
        first = (t_up < 0) & (inv > math.cos(math.radians(35))); t_up[first] = i * env.step_dt
        f = m._head_floor_force(env)
        if f is not None:
            peak = torch.maximum(peak, f.cpu())
        if args.video:
            frames.append(env.render())
inv = m._inverted_cos(asset).numpy(); nose = m._nose_up(asset).numpy(); touching = floor_bodies(env)
labels = [classify(float(inv[i]), touching[i], float(nose[i])) for i in range(N)]
# The fold's yardstick: resting in the pike = head and both feet down, nose down, trunk 60-95 degrees.
pitch_deg = np.degrees(m._trunk_pitch(asset).numpy())
in_pike = [touching[i] == {"head", "foot"} and nose[i] < -0.3 and 60 <= pitch_deg[i] <= 95 for i in range(N)]
labels = ["pike" if in_pike[i] else labels[i] for i in range(N)]
print(f"{args.run} {args.checkpoint} from {args.bucket}, {N} episodes, resets inside rollout: {int((env.episode_length_buf < 298).sum())}")
print("  end states:", {k: labels.count(k) for k in sorted(set(labels))})
reached = t_up[t_up >= 0]
print(f"  reached the headstand angle: {len(reached)}/{N}; time median {np.median(reached) if len(reached) else float('nan'):.2f} s, max {reached.max() if len(reached) else float('nan'):.2f} s")
print(f"  peak head force: median {float(peak.median()):.1f} N, max {float(peak.max()):.1f} N")
upright = (inv < -math.cos(math.radians(30))) & np.array([t == {"foot"} for t in touching])
print(f"  standing upright on the feet at the end: {int(upright.sum())}/{N}   (the back-roll's success)")
if args.video:
    imageio.mimwrite(args.video, frames, fps=50, quality=8); print("  video:", args.video)
