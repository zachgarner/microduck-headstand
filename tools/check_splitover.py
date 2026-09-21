"""Does the exit keep the split while going over? 32 episodes from the split hold.

    uv run scripts/headstand/check_splitover.py --task Mjlab-HeadstandBackrollSplit-Flat-MicroDuck --run bhcxmnvs --checkpoint model_1499.pt

While the trunk goes over (roll accumulator 170-330 deg) records the hip
split |left + right hip pitch| (2.0 rad in the hold, 0 legs together) and
the worst knee bend, per episode: the minimum split and the maximum knee
bend seen in the window, and which foot touches the floor first.
"""
import argparse, math, sys
sys.path.insert(0, "scripts/headstand")
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp as m
from eval_checkpoint import floor_bodies

p = argparse.ArgumentParser()
p.add_argument("--task", default="Mjlab-HeadstandBackrollSplit-Flat-MicroDuck")
p.add_argument("--mirrored", action="store_true", help="spawn in the mirrored split (right leg forward)")
p.add_argument("--run", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--episodes", type=int, default=32)
args = p.parse_args()
ck = Path("logs/rsl_rl") / load_rl_cfg(args.task).experiment_name / "wandb_checkpoints" / args.run / args.checkpoint
if not ck.exists():
    import wandb
    wandb.Api().run(f"zachgarner-ai/mjlab_microduck/runs/{args.run}").file(args.checkpoint).download(str(ck.parent), replace=True)
cfg = load_env_cfg(args.task, play=True); cfg.scene.num_envs = args.episodes; cfg.curriculum.clear()
for n in list(cfg.terminations):
    if n != "time_out":
        del cfg.terminations[n]
env = ManagerBasedRlEnv(cfg=cfg, device="cpu"); w = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)
runner = (load_runner_cls(args.task) or OnPolicyRunner)(w, asdict(load_rl_cfg(args.task)), device="cpu"); runner.load(str(ck), map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
if args.mirrored:
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import HEADSTAND_OVERRIDES
    env.event_manager.get_term_cfg("set_roulade_state").params["tuck_overrides"] = m._mirror_overrides(HEADSTAND_OVERRIDES)
obs, _ = w.reset(); asset = env.scene["robot"]; N = args.episodes
min_split = np.full(N, 9.0); max_knee = np.zeros(N); seen = np.zeros(N, dtype=bool); first_foot = [None] * N
with torch.no_grad():
    for i in range(150):
        obs, *_ = w.step(pol(obs))
        m._update_roulade_accum(env, asset); acc = m._roulade_state(env)[0].numpy()
        q = m._servo_joint_pos(env, asset).numpy()
        split = np.abs(q[:, 2] + q[:, 11]); knee = np.maximum(np.abs(q[:, 3]), np.abs(q[:, 12]))
        win = (acc > math.radians(170)) & (acc < math.radians(330))
        min_split = np.where(win, np.minimum(min_split, split), min_split); max_knee = np.where(win, np.maximum(max_knee, knee), max_knee); seen |= win
        touching = floor_bodies(env)
        for j in range(N):
            if first_foot[j] is None and win[j] and "foot" in touching[j]:
                first_foot[j] = i * env.step_dt
inv = m._inverted_cos(asset).numpy(); touching = floor_bodies(env)
standing = sum(1 for j in range(N) if touching[j] == {"foot"} and inv[j] < -math.cos(math.radians(30)))
print(f"{args.run} {args.checkpoint}, {N} episodes from the split hold; standing at the end {standing}/{N}; went over (window seen) {int(seen.sum())}/{N}")
print(f"  hip split while going over (hold = 2.0 rad, together = 0): min per episode, median {np.median(min_split[seen]):.2f} rad, min {min_split[seen].min():.2f}, max {min_split[seen].max():.2f}")
print(f"  worst knee bend while going over (straight = 0): median {np.median(max_knee[seen]):.2f} rad, max {max_knee[seen].max():.2f}")
print(f"  kept the split (min split > 1.0 rad AND knees < 0.6 rad): {int(((min_split > 1.0) & (max_knee < 0.6) & seen).sum())}/{N}")
print(f"  a foot touched the floor during the go-over: {sum(1 for t in first_foot if t is not None)}/{N}")
