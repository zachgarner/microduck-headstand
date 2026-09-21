"""The 32-episode flag-flip check of a split-switch checkpoint.

    uv run scripts/headstand/check_switch.py --run xikztubv --checkpoint model_250.pt --video out.mp4

Spawns in the split hold (flag 0, left leg forward), holds 1.5 s, flips the
flag to 1 for every env and freezes resampling, then watches 3.5 s. Reports
how many end inverted on the head in the MIRRORED split (closer to the
mirrored target than to the original), and how many are still inverted at
all. Terminations are disabled; resets inside the rollout are counted.
"""
import argparse, math, sys
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
from mjlab_microduck.tasks.microduck_headstand_env_cfg import HEADSTAND_OVERRIDES
from eval_checkpoint import force_spawn, floor_bodies

TASK = "Mjlab-HeadstandSplitSwitch-Flat-MicroDuck"
p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--checkpoint", required=True)
p.add_argument("--episodes", type=int, default=32); p.add_argument("--video", default=None)
p.add_argument("--flip-at", type=float, default=1.5); p.add_argument("--seconds", type=float, default=5.0)
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
force_spawn(env, "hold"); obs, _ = w.reset()
asset = env.scene["robot"]; N = args.episodes
cmd = env.command_manager.get_term("twist")
cmd.time_left[:] = 1e6            # no resampling during the check
cmd.vel_command_b[:, 0] = 0.0
target = m._servo_default_joint_pos(env, asset).clone()
for i, v in HEADSTAND_OVERRIDES.items():
    target[:, i] = v
mirrored = target.clone()
for i, v in m._mirror_overrides(HEADSTAND_OVERRIDES).items():
    mirrored[:, i] = v
def dist(t):
    return ((m._servo_joint_pos(env, asset) - t) ** 2).mean(dim=-1).sqrt()
steps = int(args.seconds / env.step_dt); flip = int(args.flip_at / env.step_dt)
frames = []; t_mirror = np.full(N, -1.0); resets = 0
with torch.no_grad():
    for i in range(steps):
        if i == flip:
            print(f"before the flip (t={i*env.step_dt:.2f} s): inverted {int((m._inverted_cos(asset) > math.cos(math.radians(35))).sum())}/{N}, "
                  f"nearer the original split {int((dist(target) < dist(mirrored)).sum())}/{N}")
            cmd.vel_command_b[:, 0] = 1.0
        obs, *_ = w.step(policy(obs))
        resets += int((env.episode_length_buf == 0).sum())
        if i > flip:
            inv = m._inverted_cos(asset) > math.cos(math.radians(35))
            arrived = (inv & (dist(mirrored) < dist(target))).numpy() & (t_mirror < 0)
            t_mirror[arrived] = (i - flip) * env.step_dt
        if args.video:
            frames.append(env.render())
inv = (m._inverted_cos(asset) > math.cos(math.radians(35))).numpy()
touching = floor_bodies(env)
on_head = np.array([t == {"head"} for t in touching])
near_mirror = (dist(mirrored) < dist(target)).numpy()
alpha = getattr(cmd, "alpha", None)
print(f"{args.run} {args.checkpoint}: {N} episodes from the split hold, flag flipped at {args.flip_at} s, resets inside rollout: {resets}")
print(f"  alpha at the end: {float(alpha.mean()) if alpha is not None else 'n/a'}")
print(f"  inverted at the end: {int(inv.sum())}/{N}; on the head alone: {int(on_head.sum())}/{N}")
print(f"  inverted AND in the mirrored split: {int((inv & near_mirror).sum())}/{N}")
print(f"  inverted, still the original split: {int((inv & ~near_mirror).sum())}/{N}")
ok = t_mirror[t_mirror >= 0]
if len(ok):
    print(f"  time from flip to the mirrored split: median {np.median(ok):.2f} s, max {ok.max():.2f} s")
lhp = m._servo_joint_pos(env, asset)[:, 2].numpy(); rhp = m._servo_joint_pos(env, asset)[:, 11].numpy()
print(f"  hip pitch at the end (left, right) median: {np.median(lhp):.2f}, {np.median(rhp):.2f}   (original target 1.20, 0.80; mirrored -0.80, -1.20)")
if args.video:
    imageio.mimwrite(args.video, frames, fps=int(round(1 / env.step_dt)))
    print(f"  video: {args.video}")
