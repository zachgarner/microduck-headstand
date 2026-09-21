"""Can the duck hop on its head? Scripted, no learning, BAM actuators.

Zach, Sep 21 2026: "i'm guessing there's no ability to sort of 'jump' with
its head right?" From the split hold, snap the neck (push the head into the
floor), pump the legs (hips flex then extend, the arm push a gymnast would
use), or both, and measure whether the head ever leaves the floor (head
contact force 0 for 2+ control steps) and how far the trunk rises.

    uv run scripts/headstand/head_hop.py
"""
import itertools, math, sys
sys.path.insert(0, "scripts/headstand")
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks.microduck_headstand_env_cfg import HEADSTAND_OVERRIDES
from eval_checkpoint import force_spawn, floor_bodies

TASK = "Mjlab-HeadstandKickup-Flat-MicroDuck"
variants = list(itertools.product((0.0, 0.4, 0.8), (0.0, 0.6, 1.2), (0.04, 0.10)))   # neck snap rad, leg pump rad, pump time s
N = len(variants)
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = N; cfg.curriculum.clear()
for n in list(cfg.terminations):
    if n != "time_out":
        del cfg.terminations[n]
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
force_spawn(env, "hold")
term = env.event_manager.get_term_cfg("set_headstand_spawn"); term.params.update(hold_pitch_noise=0.0, joint_noise_std=0.0)   # one identical start
env.reset()
asset = env.scene["robot"]
default = m._servo_default_joint_pos(env, asset).clone()
hold = default.clone()
for i, v in HEADSTAND_OVERRIDES.items():
    hold[:, i] = v
dt = env.step_dt; t_go = 0.6
z0 = None; zmax = torch.zeros(N); air = torch.zeros(N, dtype=torch.int); air_best = torch.zeros(N, dtype=torch.int)
with torch.no_grad():
    for i in range(int(2.5 / dt)):
        t = i * dt
        tgt = hold.clone()
        for j, (neck, pump, tp) in enumerate(variants):
            if t_go <= t < t_go + tp:            # load: hips flex (legs toward the belly), neck folds
                tgt[j, 2] -= pump; tgt[j, 11] -= pump; tgt[j, 5] -= neck * 0.5; tgt[j, 6] -= neck * 0.5
            elif t_go + tp <= t < t_go + tp + 0.08:   # snap: hips extend past the hold, neck pushes into the floor
                tgt[j, 2] += pump; tgt[j, 11] += pump; tgt[j, 5] += neck; tgt[j, 6] += neck
        env.step(tgt - default)
        z = asset.data.root_link_pos_w[:, 2].cpu()
        if i == int(t_go / dt) - 1:
            z0 = z.clone()
        if z0 is not None:
            zmax = torch.maximum(zmax, z - z0)
            f = m._head_floor_force(env).cpu(); up = torch.from_numpy(m._inverted_cos(asset).numpy() > math.cos(math.radians(25))) & (z > z0 - 0.005)
            air = torch.where((f < 0.05) & up, air + 1, torch.zeros_like(air)); air_best = torch.maximum(air_best, air)
inv = m._inverted_cos(asset).numpy(); touching = floor_bodies(env)
print("BAM actuators, split hold, scripted. neck snap (rad) / leg pump (rad) / load time (s):")
for j, (neck, pump, tp) in enumerate(variants):
    end = "headstand" if inv[j] > math.cos(math.radians(35)) and touching[j] == {"head"} else ("fell: " + ",".join(sorted(touching[j])))
    print(f"  neck {neck:.1f} pump {pump:.1f} load {tp:.2f}s: trunk rose {zmax[j]*100:4.1f} cm, head off the floor {int(air_best[j])} steps ({int(air_best[j])*dt*1000:.0f} ms), end {end}")
