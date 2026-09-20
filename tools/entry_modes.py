"""Split-pike hop (a) versus two-foot hop (b), scripted, under the REAL actuator model.

Zach, Sep 20 2026: "(a) a split pike hop is viable or (b) we need to have both
feet on the ground for the hop. I'd rather go with (a)". Each mode is a
hand-written joint-target schedule from the pike, played in the training
environment (BAM voltage-limited servos, allcollisions model, no
terminations, no policy). For every timing/strength combination we record the
trunk's furthest angle toward 180° and what the body rests on at the end.
This is a lower bound on each mode: a script has no balance in it.

    HEADSTAND_TASK=Mjlab-HeadstandKickup-Flat-MicroDuck uv run scripts/headstand/entry_modes.py
"""
import itertools, json, os, sys
from pathlib import Path
sys.path.insert(0, "scripts/headstand")
import numpy as np
import torch
import imageio.v2 as imageio
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp as m
from eval_checkpoint import floor_bodies

TASK = "Mjlab-HeadstandKickup-Flat-MicroDuck"
PIKE = np.array(json.load(open(Path(__file__).parent / "pike_start.json"))["qpos"], dtype=np.float32)
HOLD = np.zeros(14, dtype=np.float32); HOLD[2] = 1.2; HOLD[11] = 0.8; HOLD[5] = 1.0; HOLD[6] = 1.25
START = PIKE[7:].copy()


def schedule(mode, t, p):
    """Joint targets at time t for mode 'a' (split first, one-foot push) or 'b' (two-foot push, split at the top)."""
    T, neck_push, foot_push, lead = p["T"], p["neck_push"], p["foot_push"], p["lead"]
    c = START.copy()
    s = np.clip(t / T, 0.0, 1.0)
    # neck levers the trunk over the head in both modes
    c[5] = START[5] + s * (HOLD[5] + neck_push - START[5]); c[6] = START[6] + s * (HOLD[6] - START[6])
    if mode == "a":
        # free (left) leg swings into the split first, with a head start of `lead` seconds
        sl = np.clip((t + lead) / T, 0.0, 1.0)
        c[2] = START[2] + sl * (HOLD[2] - START[2]); c[3] = 0.0; c[4] = 0.0
        # the standing (right) foot pushes off, then its leg follows into the split
        c[13] = START[13] + s * (-foot_push - START[13]); c[11] = START[11] + s * (HOLD[11] - START[11])
    else:
        # (b) crouched two-foot hop (Zach: "Knees staying crouched is what a human
        # would do first"): first `lead` s the knees bend and the feet come in,
        # then both legs extend against the floor (hips + knees + ankles) and the
        # legs stay TUCKED over the head. Target is the tucked headstand that
        # balanced on its own in the day-one sweep (knees +-1.5), no split.
        TUCK = START.copy(); TUCK[3] = 1.2; TUCK[12] = -1.2; TUCK[2] = -1.2; TUCK[11] = 1.2
        END = np.zeros(14, dtype=np.float32); END[2] = 0.5; END[11] = -0.5; END[3] = -1.5; END[12] = 1.5; END[5] = 1.0; END[6] = 1.5
        sc = np.clip(t / max(lead, 0.05), 0.0, 1.0)             # crouch phase
        sp = np.clip((t - lead) / T, 0.0, 1.0)                    # push phase
        c = START + sc * (TUCK - START)
        c[5] = START[5] + sp * (HOLD[5] + neck_push - START[5]); c[6] = START[6] + sp * (END[6] - START[6])
        c[4] = TUCK[4] + sp * (foot_push - TUCK[4]); c[13] = TUCK[13] + sp * (-foot_push - TUCK[13])
        c[2] = TUCK[2] + sp * (END[2] - TUCK[2]); c[11] = TUCK[11] + sp * (END[11] - TUCK[11])
        c[3] = TUCK[3] + sp * (END[3] - TUCK[3]); c[12] = TUCK[12] + sp * (END[12] - TUCK[12])
        if t > lead + T:
            c[:] = END
        return c
    if t > T + 0.4:
        c[:] = HOLD
    return c


def main():
    grid = [dict(T=T, neck_push=n, foot_push=f, lead=l) for T, n, f, l in itertools.product((0.3, 0.5, 0.8), (0.0, 0.3, 0.6), (0.5, 1.0, 1.4), (0.2, 0.4))]
    n = len(grid)
    cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = n; cfg.curriculum.clear()
    for name in list(cfg.terminations):
        if name != "time_out":
            del cfg.terminations[name]
    cfg.viewer.distance = 0.6; cfg.viewer.elevation = -8; cfg.viewer.azimuth = 90; cfg.viewer.height = 480; cfg.viewer.width = 640
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
    default = env.scene["robot"].data.default_joint_pos[0].numpy()
    results = {}
    for mode in (os.environ.get("ENTRY_MODES", "ab")):
        env.reset()
        env.sim.data.qpos[:, :] = torch.tensor(PIKE).unsqueeze(0).repeat(n, 1); env.sim.data.qvel[:, :] = 0.0; env.sim.forward()
        peak = np.full(n, -180.0); frames = []
        with torch.no_grad():
            for i in range(150):   # 3 s
                t = i * env.step_dt
                targets = np.stack([schedule(mode, t, p) for p in grid])
                env.step(torch.tensor(targets - default, dtype=torch.float32))
                q = env.scene["robot"].data.root_link_quat_w.numpy()
                # trunk pitch from standing, degrees: atan2(-R20, R22)
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                r20 = 2 * (x * z - w * y); r22 = 1 - 2 * (x * x + y * y)
                pitch = np.degrees(np.arctan2(-r20, r22)); pitch = np.where(pitch < -90, pitch + 360, pitch)
                peak = np.maximum(peak, pitch)
                if i % 2 == 0:
                    frames.append(env.render())
        touching = floor_bodies(env)
        rows = sorted(zip(peak, [sorted(t) for t in touching], grid), key=lambda r: -r[0])
        results[mode] = rows
        on_head = sum(1 for pk, t, p in rows if t == ["head"] and pk > 150)
        print(f"mode {mode}: {n} scripts; peak trunk angle best {rows[0][0]:.0f} deg, median {np.median(peak):.0f}; ended on the head alone past 150: {on_head}")
        for pk, t, p in rows[:4]:
            print(f"   peak {pk:6.1f}  ends on {t}  {p}")
        video = f"/Users/zach/Desktop/microduck/entry_mode_{mode}_scripted_real_actuators.mp4"
        imageio.mimwrite(video, frames, fps=25, quality=8)
        print("   video (env 0 = first grid entry):", video)
    env.close()


if __name__ == "__main__":
    main()
