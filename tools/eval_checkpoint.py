"""Headless eval of a headstand checkpoint: per-spawn-type success rates and a video.

AGENTS.md: "Measure before theorizing" and "watch the video AND check which
geom/axis touches". This runs full episodes on CPU from each spawn bucket
(standing / partway / hold), scores the END STATE of every episode by the
trunk orientation and the bodies on the floor, and writes an mp4 of the
standing-spawn rollouts (the entry is what we want to see).

    uv run ../tools/eval_checkpoint.py --wandb-run-path zachgarner-ai/mjlab_microduck/rfcisbwg --checkpoint model_250.pt
    uv run ../tools/eval_checkpoint.py --checkpoint-file logs/.../model_250.pt --episodes 32

Success = trunk within 35° of inverted AND the head is the only body on the
floor, at the last step. "Tripod" = head plus a foot. "Flop" = anything else
down. Numbers are what the rollouts show, nothing is extrapolated.
"""

import argparse
import os
import math
import numpy as np
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_microduck.tasks import mdp as microduck_mdp

TASK = os.environ.get("HEADSTAND_TASK", "Mjlab-HeadstandKickup-Flat-MicroDuck")
HEAD = {"jaw_soft", "yaw_roll_motion", "neck_pitch"}
FEET = {"ankle_left", "ankle_right"}


def contacts(env) -> dict:
    """What each env touches the floor with, from the headstand sensors.

    feet is a COUNT (0, 1 or 2), not a flag: the pike and standing need both
    feet, and a one-foot pose passed every check that collapsed them.
    """
    head = microduck_mdp._sensor_any_contact(env, microduck_mdp._HEADSTAND_HEAD_SENSOR)
    other = microduck_mdp._sensor_any_contact(env, microduck_mdp._HEADSTAND_OTHER_SENSOR)
    found = env.scene.sensors[microduck_mdp._HEADSTAND_FEET_SENSOR].data.found
    per_foot = (found.view(found.shape[0], -1) > 0)
    if per_foot.shape[-1] != 2:
        raise ValueError(f"the feet sensor must have one slot per foot, got {per_foot.shape[-1]}")
    return {"head": head.cpu().numpy(), "feet": per_foot.sum(dim=-1).cpu().numpy(),
            "other": other.cpu().numpy()}


def in_headstand(env, asset, deg: float = 35.0) -> "np.ndarray":
    """Inverted within `deg` AND supported by the head alone: no foot, nothing
    else. Angle alone passes a duck propped on a foot."""
    inv = microduck_mdp._inverted_cos(asset).cpu().numpy()
    c = contacts(env)
    return (inv > math.cos(math.radians(deg))) & c["head"] & (c["feet"] == 0) & ~c["other"]


def leg_shape_ok(joints, style):
    """Full leg-shape thresholds: knees <= 0.3 rad; together <= 0.2,
    split >= 1.0 rad. Hip separation uses the mirrored joint convention.
    These thresholds are fixed before evaluation and match full training gates.
    """
    q = np.asarray(joints)
    knees = np.maximum(np.abs(q[:, 3]), np.abs(q[:, 12])) <= 0.3
    separation = np.abs(q[:, 2] + q[:, 11])
    if style == "legs_together":
        return knees & (separation <= 0.2)
    if style == "split":
        return knees & (separation >= 1.0)
    raise ValueError(f"Unknown leg style: {style}")


def in_shaped_headstand(env, asset, style):
    return in_headstand(env, asset) & leg_shape_ok(
        microduck_mdp._servo_joint_pos(env, asset).cpu().numpy(), style
    )


def routine_success(stage, stage_count, standing):
    """Every required hold completed AND strictly standing at the last frame."""
    return (np.asarray(stage) == stage_count) & np.asarray(standing, dtype=bool)


def in_pike(env, asset) -> "np.ndarray":
    """Resting in the pike: head and BOTH feet down, nothing else, nose down,
    trunk 60-95 degrees from standing. Matches the fold's training gate."""
    c = contacts(env)
    nose = microduck_mdp._nose_up(asset).cpu().numpy()
    pitch = np.degrees(microduck_mdp._trunk_pitch(asset).cpu().numpy())
    return c["head"] & (c["feet"] == 2) & ~c["other"] & (nose < -0.3) & (pitch >= 60) & (pitch <= 95)


def is_standing(env, asset, deg: float = 30.0) -> "np.ndarray":
    """Upright on both feet, nothing else touching."""
    inv = microduck_mdp._inverted_cos(asset).cpu().numpy()
    c = contacts(env)
    return (c["feet"] == 2) & ~c["head"] & ~c["other"] & (inv < -math.cos(math.radians(deg)))


def floor_bodies(env) -> list[set]:
    """The old label set, kept for the end-state classifier. "foot" means at
    least one foot; use contacts() when the count matters."""
    c = contacts(env)
    out = []
    for i in range(env.num_envs):
        s = set()
        if c["head"][i]:
            s.add("head")
        if c["feet"][i] > 0:
            s.add("foot")
        if c["other"][i]:
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
    """Point the spawn event at one bucket via the manager (cfg writes are no-ops).

    Every bucket is set, so no weight the task registers can leak into a
    forced start. "pike" is the measured resting pike, "handover" a recorded
    one. "tripod" and "bank" are the old names, kept so older commands run.

    "tripod" is the partway bucket pinned to 90-110° with legs near HOME: head
    down, feet down, the state the kick-up policy has to leave.
    """
    if "set_headstand_spawn" not in env.event_manager.active_terms.get("reset", []):
        return   # the back-roll task spawns through the roulade's event, already pinned to the hold
    term = env.event_manager.get_term_cfg("set_headstand_spawn")
    term.params["standing_prob"] = 1.0 if bucket == "standing" else 0.0
    term.params["partway_prob"] = 1.0 if bucket == "partway" else 0.0
    term.params["hold_prob"] = 1.0 if bucket == "hold" else 0.0
    term.params["pike_prob"] = 1.0 if bucket in ("tripod", "pike") else 0.0   # the measured RESTING pike
    term.params["handover_prob"] = 1.0 if bucket in ("bank", "handover") else 0.0   # pikes as the fold policy leaves them
    if bucket.startswith("pitch"):   # e.g. "pitch150": dropped head-down at that angle
        deg = float(bucket[5:])
        term.params["partway_prob"] = 1.0
        term.params["partway_pitch_min"] = math.radians(deg)
        term.params["partway_pitch_max"] = math.radians(deg + 1.0)
        term.params["partway_lerp_range"] = (0.0, 0.3)
    else:
        term.params["partway_pitch_min"] = env.cfg.events["set_headstand_spawn"].params["partway_pitch_min"]
        term.params["partway_pitch_max"] = env.cfg.events["set_headstand_spawn"].params["partway_pitch_max"]
        term.params["partway_lerp_range"] = env.cfg.events["set_headstand_spawn"].params["partway_lerp_range"]


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
    p.add_argument("--record-bucket", default="standing", help="which bucket the video shows")
    args = p.parse_args()

    if args.checkpoint_file:
        ckpt = Path(args.checkpoint_file)
    else:
        assert args.wandb_run_path and args.checkpoint, "give --wandb-run-path and --checkpoint, or --checkpoint-file"
        log_root = Path("logs") / "rsl_rl" / ("microduck_headstand_kickup" if "Kickup" in TASK else "microduck_headstand")
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
        record = bucket == args.record_bucket
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
