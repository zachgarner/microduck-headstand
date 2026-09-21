"""Zach's full routine as one rollout, seven policies handing over on state.

    fold -> straight kick-up -> hold -> back roll -> stand -> fold -> split kick-up -> hold -> split exit -> stand

    HEADSTAND_TASK=Mjlab-HeadstandKickupStraight-Flat-MicroDuck uv run scripts/headstand/routine_full.py \\
        --fold vorty4kb:model_1000.pt --straight geexqesa:model_1000.pt --roll 44wneb8l:model_1000.pt \\
        --split 076n5wpa:model_999.pt --splitexit fxhauoka:model_750.pt --stand policies/pollen/alpha_stand.onnx \\
        --hold-s 2 --episodes 16 --video ~/Desktop/microduck/routine_full.mp4

Physics: the allcollisions model, terminations off. Handovers are state
conditions held for a short time (in the pike, in the headstand, standing),
which is how the runtime would switch policies. Reports how many episodes
reach each stage and the end state.
"""
import argparse, math, os, sys
sys.path.insert(0, "scripts/headstand")
from dataclasses import asdict
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp as m
from eval_checkpoint import floor_bodies, classify

TASK = {
    "fold": "Mjlab-HeadstandFold-Flat-MicroDuck",
    "straight": "Mjlab-HeadstandKickupStraight-Flat-MicroDuck",
    "roll": "Mjlab-HeadstandBackrollStraight-Flat-MicroDuck",
    "split": "Mjlab-HeadstandKickup-Flat-MicroDuck",
    "splitexit": "Mjlab-HeadstandSplitExit-Flat-MicroDuck",
    "switch": "Mjlab-HeadstandSplitSwitch-Flat-MicroDuck",
    "splitroll": "Mjlab-HeadstandBackrollSplit-Flat-MicroDuck",
}
TWIST_VX = 48   # obs layout: 48 proprio, then [twist(3), head_pose(4), body_pose(6)]


def flagged(pol, flag: float):
    """The split-switch policy reads its flag from the twist vx slot (AGENTS.md
    obs contract). The routine env's twist command is ~0, so set the slot."""
    def run(obs):
        o = dict(obs); o["actor"] = obs["actor"].clone(); o["actor"][:, TWIST_VX] = flag
        return pol(o)
    return run


def torch_policy(task, spec, wrapped):
    run, ck = spec.split(":")
    path = Path("logs/rsl_rl") / load_rl_cfg(task).experiment_name / "wandb_checkpoints" / run / ck
    if not path.exists():
        import wandb
        wandb.Api().run(f"zachgarner-ai/mjlab_microduck/runs/{run}").file(ck).download(str(path.parent), replace=True)
    runner = (load_runner_cls(task) or OnPolicyRunner)(wrapped, asdict(load_rl_cfg(task)), device="cpu")
    runner.load(str(path), map_location="cpu")
    pol = runner.get_inference_policy(device="cpu")
    return lambda obs: pol(obs)


def onnx_policy(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
    def run(obs):
        o = obs["actor"].numpy().astype(np.float32)
        return torch.tensor(np.concatenate([sess.run(None, {inp: o[j:j + 1]})[0] for j in range(o.shape[0])], axis=0))
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("fold", "straight", "roll", "split", "splitexit"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--stand", required=True, help="ONNX of Pollen's standing policy")
    p.add_argument("--switch", default=None, help="run:ckpt of the split-switch policy; adds switch there and back after the split hold")
    p.add_argument("--switch-hold-s", type=float, default=1.5, help="hold in each split of the switch")
    p.add_argument("--splitroll", default=None, help="run:ckpt of the split back roll; replaces the split exit and the stand-up")
    p.add_argument("--hold-s", type=float, default=2.0); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--seconds", type=float, default=16.0); p.add_argument("--video", default=None)
    p.add_argument("--settle-s", type=float, default=1.0, help="standing policy holds this long after the roll")
    p.add_argument("--pike-settle-s", type=float, default=0.5, help="extra time resting in the pike before the kick-up")
    args = p.parse_args()
    N = args.episodes
    cfg = load_env_cfg(TASK["straight"], play=True); cfg.scene.num_envs = N; cfg.curriculum.clear()
    cfg.episode_length_s = args.seconds + 1.0
    for n in list(cfg.terminations):
        if n != "time_out":
            del cfg.terminations[n]
    cfg.viewer.distance = 0.7; cfg.viewer.elevation = -8; cfg.viewer.azimuth = 135; cfg.viewer.height = 480; cfg.viewer.width = 640
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array" if args.video else None)
    w = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK["straight"]).clip_actions)
    fold_pol = torch_policy(TASK["fold"], args.fold, w)
    stand_pol = onnx_policy(args.stand)
    # Settle stages (Sep 21): after the roll the duck is standing with momentum
    # and the fold policy has only seen a still stand; Pollen's standing policy
    # holds it for `--settle-s` first. The pike gets `--pike-settle-s` of the
    # fold policy holding before the kick-up takes over.
    stages = [
        ("fold",      fold_pol,                                           "pike",      0.3 + args.pike_settle_s),
        ("straight",  torch_policy(TASK["straight"], args.straight, w),   "headstand", args.hold_s),
        ("roll",      torch_policy(TASK["roll"], args.roll, w),           "standing",  0.3),
        ("settle",    stand_pol,                                          "standing",  args.settle_s),
        ("fold2",     fold_pol,                                           "pike",      0.3 + args.pike_settle_s),
        ("split",     torch_policy(TASK["split"], args.split, w),         "headstand", args.hold_s),
        ("splitexit", torch_policy(TASK["splitexit"], args.splitexit, w), "pike",      0.3),
        ("stand",     stand_pol,                                          "standing",  1.0),
    ]
    if args.switch:
        # Zach's flair (Sep 21): "switch its split while in the air". Flag 1
        # mirrors the split, flag 0 brings it back so the split exit sees the
        # split it was trained on. Both count as "in the headstand" held.
        sw = torch_policy(TASK["switch"], args.switch, w)
        k = [n for n, *_ in stages].index("splitexit")
        stages[k:k] = [("switch",     flagged(sw, 1.0), "mirrored", args.switch_hold_s),
                       ("switchback", flagged(sw, 0.0), "original", args.switch_hold_s)]
    if args.splitroll:
        # Zach, Sep 21: the split exit should "continue the split" over into a
        # back roll, not come back down the way it went up. The split back
        # roll ends standing on its own, so Pollen's stand-up goes too.
        k = [n for n, *_ in stages].index("splitexit")
        stages[k:k + 2] = [("splitroll", torch_policy(TASK["splitroll"], args.splitroll, w), "standing", 0.3),
                           ("settle2",   stand_pol,                                         "standing", 1.0)]
    term = env.event_manager.get_term_cfg("set_headstand_spawn")
    term.params.update(standing_prob=1.0, partway_prob=0.0, hold_prob=0.0, tripod_prob=0.0)
    obs, _ = w.reset()
    asset = env.scene["robot"]
    # The switch stages count only when the joints are nearer the mirrored
    # (or the original) split target than the other, not merely inverted.
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import HEADSTAND_OVERRIDES
    split_t = m._servo_default_joint_pos(env, asset).clone(); mirror_t = split_t.clone()
    for i_, v_ in HEADSTAND_OVERRIDES.items():
        split_t[:, i_] = v_
    for i_, v_ in m._mirror_overrides(HEADSTAND_OVERRIDES).items():
        mirror_t[:, i_] = v_
    stage = np.zeros(N, dtype=int); held = np.zeros(N); t_stage = np.full((N, len(stages) + 1), -1.0); t_stage[:, 0] = 0.0
    steps = int(args.seconds / env.step_dt); frames = []
    with torch.no_grad():
        for i in range(steps):
            t = i * env.step_dt
            acts = torch.zeros(N, 14)
            for k, (name, pol, _, _) in enumerate(stages):
                # finished episodes keep running the last policy (standing) to the end
                idx = np.where(np.minimum(stage, len(stages) - 1) == k)[0]
                if len(idx):
                    acts[idx] = pol(obs)[idx]
            obs, *_ = w.step(acts)
            inv = m._inverted_cos(asset).numpy(); nose = m._nose_up(asset).numpy()
            pitch = np.degrees(m._trunk_pitch(asset).numpy()); touching = floor_bodies(env)
            cond = {
                "pike": np.array([touching[j] == {"head", "foot"} and nose[j] < -0.3 and 60 <= pitch[j] <= 95 for j in range(N)]),
                "headstand": inv > math.cos(math.radians(35)),
                "mirrored": (inv > math.cos(math.radians(35))) & (((m._servo_joint_pos(env, asset) - mirror_t) ** 2).sum(-1) < ((m._servo_joint_pos(env, asset) - split_t) ** 2).sum(-1)).numpy(),
                "original": (inv > math.cos(math.radians(35))) & (((m._servo_joint_pos(env, asset) - split_t) ** 2).sum(-1) < ((m._servo_joint_pos(env, asset) - mirror_t) ** 2).sum(-1)).numpy(),
                "standing": np.array([touching[j] == {"foot"} and inv[j] < -math.cos(math.radians(30)) for j in range(N)]),
            }
            for j in range(N):
                k = stage[j]
                if k >= len(stages):
                    continue
                if cond[stages[k][2]][j]:
                    held[j] += env.step_dt
                    if held[j] >= stages[k][3]:
                        stage[j] += 1; held[j] = 0.0; t_stage[j, stage[j]] = t
                else:
                    held[j] = 0.0
            if args.video:
                frames.append(env.render())
    inv = m._inverted_cos(asset).numpy(); nose = m._nose_up(asset).numpy(); touching = floor_bodies(env)
    labels = [classify(float(inv[j]), touching[j], float(nose[j])) for j in range(N)]
    print(f"full routine, {N} standing starts, {args.seconds:.0f} s, resets inside rollout: {int((env.episode_length_buf < steps).sum())}")
    for k, (name, _, goal, _) in enumerate(stages):
        reached = t_stage[:, k + 1] >= 0
        med = np.median(t_stage[reached, k + 1]) if reached.any() else float("nan")
        print(f"  {k+1}. {name:9s} -> {goal:9s}: {int(reached.sum()):2d}/{N} done, at {med:5.2f} s median")
    print("  end states:", {k: labels.count(k) for k in sorted(set(labels))})
    if args.video:
        imageio.mimwrite(args.video, frames, fps=50, quality=8); print("  video:", args.video)


if __name__ == "__main__":
    main()
