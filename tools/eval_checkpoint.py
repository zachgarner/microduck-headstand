"""Headless eval of a headstand checkpoint: per-spawn-type success rates and a video.

AGENTS.md: "Measure before theorizing" and "watch the video AND check which
geom/axis touches". This runs full episodes on CPU from each spawn bucket
(standing / partway / hold), scores the END STATE of every episode by the
trunk orientation and the bodies on the floor, and writes an mp4 of the
standing-spawn rollouts (the entry is what we want to see).

    uv run scripts/headstand/eval_checkpoint.py --wandb-run-path zachgarner-ai/mjlab_microduck/rfcisbwg --checkpoint model_250.pt
    uv run scripts/headstand/eval_checkpoint.py --checkpoint-file logs/.../model_250.pt --episodes 32

Success = trunk within 35° of inverted AND the head is the only body on the
floor, at the last step. "Tripod" = head plus a foot. "Flop" = anything else
down. Numbers are what the rollouts show, nothing is extrapolated.
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_microduck.tasks import mdp as microduck_mdp

TASK = "Mjlab-Headstand-Flat-MicroDuck"
HEAD = {"jaw_soft", "yaw_roll_motion", "neck_pitch"}
FEET = {"ankle_left", "ankle_right"}


def floor_bodies(env) -> list[set]:
    """Bodies touching the terrain in each env, from the three headstand sensors."""
    head = microduck_mdp._sensor_any_contact(env, microduck_mdp._HEADSTAND_HEAD_SENSOR)
    feet = microduck_mdp._sensor_any_contact(env, microduck_mdp._HEADSTAND_FEET_SENSOR)
    other = microduck_mdp._sensor_any_contact(env, microduck_mdp._HEADSTAND_OTHER_SENSOR)
    out = []
    for i in range(env.num_envs):
        s = set()
        if head is not None and bool(head[i]):
            s.add("head")
        if feet is not None and bool(feet[i]):
            s.add("foot")
        if other is not None and bool(other[i]):
            s.add("other")
        out.append(s)
    return out


def classify(inverted_cos: float, touching: set, nose_up: float = 0.0) -> str:
    inverted = inverted_cos > math.cos(math.radians(35.0))
    if inverted and touching == {"head"}:
        return "headstand"
    if nose_up > 0.3 and "head" in touching:
        return "fell_backward"   # top of the head down, feet out front, nose up
    if touching == {"head", "foot"}:
        return "tripod"
    if inverted and not touching:
        return "airborne"
    if "other" in touching:
        return "flop"
    if touching == {"foot"}:
        return "standing"
    return "other"


def force_spawn(env, bucket: str):
    """Point the spawn event at one bucket via the manager (cfg writes are no-ops)."""
    term = env.event_manager.get_term_cfg("set_headstand_spawn")
    term.params["standing_prob"] = 1.0 if bucket == "standing" else 0.0
    term.params["partway_prob"] = 1.0 if bucket == "partway" else 0.0
    term.params["hold_prob"] = 1.0 if bucket == "hold" else 0.0


def run_bucket(env, wrapped, policy, bucket: str, record: bool, video_length: int):
    force_spawn(env, bucket)
    obs, _ = wrapped.reset()
    frames = []
    # Stop one step short of the time-out: on the final step mjlab auto-resets
    # the env and the state read afterwards would be the NEXT spawn.
    steps = int(env.max_episode_length) - 1
    peak = torch.zeros(env.num_envs)
    with torch.no_grad():
        for t in range(steps):
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            f = microduck_mdp._head_floor_force(env)
            if f is not None:
                peak = torch.maximum(peak, f.cpu())
            if record and t < video_length:
                frames.append(env.render())
    env._eval_peak_force = peak
    asset = env.scene["robot"]
    inv = microduck_mdp._inverted_cos(asset).cpu().numpy()
    nose = microduck_mdp._nose_up(asset).cpu().numpy()
    touching = floor_bodies(env)
    labels = [classify(float(inv[i]), touching[i], float(nose[i])) for i in range(env.num_envs)]
    slammed = int(env._headstand_slammed.sum()) if hasattr(env, "_headstand_slammed") else None
    peak = getattr(env, "_eval_peak_force", None)
    return labels, frames, slammed, peak


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wandb-run-path", default=None)
    p.add_argument("--checkpoint", default=None, help="e.g. model_250.pt (with --wandb-run-path)")
    p.add_argument("--checkpoint-file", default=None)
    p.add_argument("--episodes", type=int, default=16, help="envs per bucket")
    p.add_argument("--video", default=None, help="output mp4 (default: renders/headstand_<ckpt>.mp4)")
    p.add_argument("--video-length", type=int, default=300)
    p.add_argument("--buckets", default="standing,partway,hold")
    args = p.parse_args()

    if args.checkpoint_file:
        ckpt = Path(args.checkpoint_file)
    else:
        assert args.wandb_run_path and args.checkpoint, "give --wandb-run-path and --checkpoint, or --checkpoint-file"
        log_root = Path("logs") / "rsl_rl" / "microduck_headstand"
        run_id = args.wandb_run_path.split("/")[-1]
        ckpt = log_root / "wandb_checkpoints" / run_id / args.checkpoint
        if not ckpt.exists():
            import wandb
            wandb.Api().run(args.wandb_run_path).file(args.checkpoint).download(str(ckpt.parent), replace=True)
    print(f"checkpoint: {ckpt}")

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.episodes
    # The spawn-mix curriculum rewrites the spawn probabilities at every reset
    # (stage 0 at step 0), which silently undid force_spawn on the first eval.
    # No curriculum belongs in an eval: freeze everything at the cfg's values.
    env_cfg.curriculum.clear()
    # Frame the duck: the play default sits 3 m away and the robot is 25 cm.
    env_cfg.viewer.distance = 0.6
    env_cfg.viewer.elevation = -10.0
    env_cfg.viewer.azimuth = 135.0
    env_cfg.viewer.height = 480
    env_cfg.viewer.width = 640
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
    runner.load(str(ckpt), map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    video = args.video or f"renders/headstand_{ckpt.stem}.mp4"
    Path(video).parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for bucket in args.buckets.split(","):
        record = bucket == "standing"
        labels, frames, slammed, peak = run_bucket(env, wrapped, policy, bucket, record, args.video_length)
        counts = {k: labels.count(k) for k in sorted(set(labels))}
        summary[bucket] = counts
        extra = ""
        if peak is not None:
            extra = f"  head force median={float(peak.median()):.1f}N max={float(peak.max()):.1f}N"
        if slammed is not None:
            extra += f"  slam-marked={slammed}"
        print(f"{bucket:9s} n={len(labels):3d}  " + "  ".join(f"{k}={v}" for k, v in counts.items()) + extra)
        if record and frames:
            fps = int(round(1.0 / env.step_dt))
            imageio.mimwrite(video, frames, fps=fps, quality=8)
            print(f"video: {video} ({len(frames)} frames at {fps} fps)")
    env.close()
    return summary


if __name__ == "__main__":
    main()
