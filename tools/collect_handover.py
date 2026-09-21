"""Record the states a policy hands over, to train the next policy from them.

    HEADSTAND_TASK=Mjlab-HeadstandFold-Flat-MicroDuck uv run scripts/headstand/collect_handover.py \\
        --run vorty4kb --checkpoint model_1000.pt --bucket standing --condition pike --n 512 \\
        --out src/mjlab_microduck/tasks/handover_pike_from_fold.npz

Runs the policy from `bucket` spawns, waits until `condition` (pike / headstand /
standing) has held for `--hold-s`, and saves the full qpos (base pose + 14
joints) and qvel at that moment. The kick-up's spawn can then draw from this
bank (`pike_bank`), so it trains from the pikes the fold actually produces
instead of the one measured resting pike.
"""
import argparse, math, os, sys
sys.path.insert(0, "scripts/headstand")
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.tasks import mdp as m
from eval_checkpoint import TASK, force_spawn, floor_bodies

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--checkpoint", required=True)
p.add_argument("--bucket", default="standing"); p.add_argument("--condition", default="pike")
p.add_argument("--hold-s", type=float, default=0.3); p.add_argument("--n", type=int, default=512)
p.add_argument("--out", required=True)
args = p.parse_args()
N = 64
exp = load_rl_cfg(TASK).experiment_name
ck = Path("logs/rsl_rl") / exp / "wandb_checkpoints" / args.run / args.checkpoint
if not ck.exists():
    import wandb
    wandb.Api().run(f"zachgarner-ai/mjlab_microduck/runs/{args.run}").file(args.checkpoint).download(str(ck.parent), replace=True)
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = N; cfg.curriculum.clear()
for n in list(cfg.terminations):
    if n != "time_out":
        del cfg.terminations[n]
env = ManagerBasedRlEnv(cfg=cfg, device="cpu"); w = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
runner = (load_runner_cls(TASK) or OnPolicyRunner)(w, asdict(load_rl_cfg(TASK)), device="cpu"); runner.load(str(ck), map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
asset = env.scene["robot"]
qpos_bank, qvel_bank = [], []
while len(qpos_bank) < args.n:
    force_spawn(env, args.bucket); obs, _ = w.reset()
    held = np.zeros(N); taken = np.zeros(N, dtype=bool)
    with torch.no_grad():
        for i in range(150):
            obs, *_ = w.step(policy(obs))
            inv = m._inverted_cos(asset).numpy(); nose = m._nose_up(asset).numpy()
            pitch = np.degrees(m._trunk_pitch(asset).numpy()); touching = floor_bodies(env)
            if args.condition == "pike":
                ok = np.array([touching[j] == {"head", "foot"} and nose[j] < -0.3 and 60 <= pitch[j] <= 95 for j in range(N)])
            elif args.condition == "headstand":
                ok = inv > math.cos(math.radians(35))
            else:
                ok = np.array([touching[j] == {"foot"} and inv[j] < -math.cos(math.radians(30)) for j in range(N)])
            held = np.where(ok, held + env.step_dt, 0.0)
            ready = (held >= args.hold_s) & ~taken
            if ready.any():
                q = env.sim.data.qpos.cpu().numpy(); v = env.sim.data.qvel.cpu().numpy()
                for j in np.where(ready)[0]:
                    qpos_bank.append(q[j].copy()); qvel_bank.append(v[j].copy()); taken[j] = True
    print(f"collected {len(qpos_bank)}", flush=True)
qpos_bank = np.array(qpos_bank[: args.n]); qvel_bank = np.array(qvel_bank[: args.n])
# base x/y are not part of the state that matters; zero them so spawns land at each env's origin
qpos_bank[:, 0:2] = 0.0
np.savez(args.out, qpos=qpos_bank, qvel=qvel_bank)
print("saved", args.out, qpos_bank.shape, "trunk z median %.3f pitch median %.1f" % (np.median(qpos_bank[:, 2]), np.median(np.degrees(np.arctan2(-2*(qpos_bank[:,1]*qpos_bank[:,3]-qpos_bank[:,0+3]*qpos_bank[:,2]), 1)))))
