"""Run a chain of headstand policies in one rollout: fold -> kick-up -> hold -> back roll.

Each stage is a trained checkpoint (task id + wandb run + checkpoint file).
Stages hand over on a STATE condition, not a clock: the fold hands over once
the duck rests in the pike, the kick-up (which also holds) hands over to the
roll after `hold_s` seconds of headstand. All policies share the 61-number
observation, which is how the real robot hot-swaps them too.

    HEADSTAND_TASK=Mjlab-HeadstandKickupStraight-Flat-MicroDuck uv run scripts/headstand/routine.py \\
        --fold vorty4kb:model_250.pt --kickup geexqesa:model_250.pt [--roll <run>:<ckpt>] --hold-s 2 \\
        --video ~/Desktop/microduck/routine_straight.mp4

Physics run in the kick-up's environment (allcollisions model, terminations
off). Reports per-episode outcomes and where each stage handed over.
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
from eval_checkpoint import floor_bodies, classify

TASKS = dict(
    fold="Mjlab-HeadstandFold-Flat-MicroDuck",
    kickup=os.environ.get("HEADSTAND_TASK", "Mjlab-HeadstandKickupStraight-Flat-MicroDuck"),
    roll="Mjlab-HeadstandBackrollStraight-Flat-MicroDuck",
)


def load_policy(task, spec, wrapped):
    run, ck = spec.split(":")
    exp = load_rl_cfg(task).experiment_name
    path = Path("logs/rsl_rl") / exp / "wandb_checkpoints" / run / ck
    if not path.exists():
        import wandb
        wandb.Api().run(f"zachgarner-ai/mjlab_microduck/runs/{run}").file(ck).download(str(path.parent), replace=True)
    runner = (load_runner_cls(task) or OnPolicyRunner)(wrapped, asdict(load_rl_cfg(task)), device="cpu")
    runner.load(str(path), map_location="cpu")
    return runner.get_inference_policy(device="cpu")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", required=True); p.add_argument("--kickup", required=True); p.add_argument("--roll", default=None)
    p.add_argument("--hold-s", type=float, default=2.0); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--seconds", type=float, default=8.0); p.add_argument("--video", default=None)
    args = p.parse_args()
    N = args.episodes
    cfg = load_env_cfg(TASKS["kickup"], play=True); cfg.scene.num_envs = N; cfg.curriculum.clear()
    cfg.episode_length_s = args.seconds + 1.0
    for n in list(cfg.terminations):
        if n != "time_out":
            del cfg.terminations[n]
    cfg.viewer.distance = 0.65; cfg.viewer.elevation = -8; cfg.viewer.azimuth = 135; cfg.viewer.height = 480; cfg.viewer.width = 640
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array" if args.video else None)
    w = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASKS["kickup"]).clip_actions)
    policies = {"fold": load_policy(TASKS["fold"], args.fold, w), "kickup": load_policy(TASKS["kickup"], args.kickup, w)}
    if args.roll:
        policies["roll"] = load_policy(TASKS["roll"], args.roll, w)
    term = env.event_manager.get_term_cfg("set_headstand_spawn")
    term.params.update(standing_prob=1.0, partway_prob=0.0, hold_prob=0.0, tripod_prob=0.0)
    obs, _ = w.reset()
    asset = env.scene["robot"]
    stage = np.zeros(N, dtype=int)                 # 0 fold, 1 kickup/hold, 2 roll
    t_handoff = np.full((N, 2), -1.0); t_headstand = np.full(N, -1.0); headstand_time = np.zeros(N)
    steps = int(args.seconds / env.step_dt); frames = []
    order = ["fold", "kickup"] + (["roll"] if args.roll else [])
    with torch.no_grad():
        for i in range(steps):
            t = i * env.step_dt
            acts = torch.zeros(N, 14)
            for k, name in enumerate(order):
                idx = torch.tensor(np.where(stage == k)[0])
                if len(idx):
                    acts[idx] = policies[name](obs)[idx]
            obs, *_ = w.step(acts)
            inv = m._inverted_cos(asset).numpy(); nose = m._nose_up(asset).numpy()
            pitch = np.degrees(m._trunk_pitch(asset).numpy()); touching = floor_bodies(env)
            in_pike = np.array([touching[j] == {"head", "foot"} and nose[j] < -0.3 and 60 <= pitch[j] <= 95 for j in range(N)])
            is_headstand = inv > math.cos(math.radians(35))
            # fold -> kick-up once resting in the pike for 0.3 s
            for j in range(N):
                if stage[j] == 0 and in_pike[j]:
                    t_handoff[j, 0] = t if t_handoff[j, 0] < 0 else t_handoff[j, 0]
                    if t - t_handoff[j, 0] >= 0.3:
                        stage[j] = 1
                elif stage[j] == 0:
                    t_handoff[j, 0] = -1.0
                if stage[j] == 1 and is_headstand[j]:
                    headstand_time[j] += env.step_dt
                    if t_headstand[j] < 0:
                        t_headstand[j] = t
                    if args.roll and headstand_time[j] >= args.hold_s:
                        stage[j] = 2; t_handoff[j, 1] = t
            if args.video:
                frames.append(env.render())
    inv = m._inverted_cos(asset).numpy(); nose = m._nose_up(asset).numpy(); touching = floor_bodies(env)
    labels = [classify(float(inv[j]), touching[j], float(nose[j])) for j in range(N)]
    print(f"routine, {N} standing starts, resets inside rollout: {int((env.episode_length_buf < steps).sum())}")
    print(f"  reached the pike (fold handed over): {int((stage >= 1).sum())}/{N}, at {np.median(t_handoff[stage>=1,0]) if (stage>=1).any() else float('nan'):.2f} s median")
    print(f"  reached the headstand: {int((t_headstand >= 0).sum())}/{N}, at {np.median(t_headstand[t_headstand>=0]) if (t_headstand>=0).any() else float('nan'):.2f} s median; held {np.median(headstand_time):.2f} s median")
    if args.roll:
        print(f"  handed to the roll: {int((stage == 2).sum())}/{N}")
    print(f"  end states at {args.seconds:.0f} s:", {k: labels.count(k) for k in sorted(set(labels))})
    if args.video:
        imageio.mimwrite(args.video, frames, fps=50, quality=8); print("  video:", args.video)


if __name__ == "__main__":
    main()
