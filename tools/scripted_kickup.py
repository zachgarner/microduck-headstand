"""Scripted kick-up from the pike, no learning: can the servos do it at all?

From the settled pike (head top on the floor, both feet down, trunk 76°),
play a hand-written motion over T seconds and measure the trunk's furthest
angle toward the headstand (180° from standing) and what the body rests on
at the end. Sweeps the three things a person would coordinate: how hard the
neck pushes, how fast the free leg lifts into the split, and how hard the
standing foot pushes off. AGENTS.md: verify physics before training.

    uv run scripts/headstand/scripted_kickup.py --video ~/Desktop/microduck/scripted_kickup_best.mp4
"""
import argparse, itertools, json, sys
from pathlib import Path
sys.path.insert(0, "scripts/headstand")
import imageio.v2 as imageio
import numpy as np
import mujoco
from settle_sweep import SCENE, floor_contacts

PIKE = json.load(open(Path(__file__).parent / "pike_start.json"))
HOLD = np.zeros(14); HOLD[2] = 1.2; HOLD[11] = 0.8; HOLD[5] = 1.0; HOLD[6] = 1.25

def pitch_deg(d):
    R = d.xmat[1].reshape(3, 3)
    return float(np.degrees(np.arctan2(-R[2, 0], R[2, 2])))

def play(m, d, neck_push, leg_time, foot_push, T=1.5, hold_after=2.0, frames=None):
    d.qpos[:] = PIKE["qpos"]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    start = np.array(PIKE["qpos"][7:], dtype=float)
    ctrl = start.copy(); d.ctrl[:] = ctrl
    for _ in range(int(0.3 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    best = pitch_deg(d)
    n = int((T + hold_after) / m.opt.timestep)
    for i in range(n):
        t = i * m.opt.timestep
        s = min(1.0, t / T)
        ctrl = start.copy()
        # Neck: push the head into the floor to lever the trunk over (neck_pitch
        # toward +, head_pitch toward the hold value). neck_push scales how far past.
        ctrl[5] = start[5] + s * (HOLD[5] + neck_push - start[5])
        ctrl[6] = start[6] + s * (HOLD[6] - start[6])
        # Free (left) leg: swing up into the split over leg_time seconds.
        sl = min(1.0, t / max(leg_time, 1e-3))
        ctrl[2] = start[2] + sl * (HOLD[2] - start[2]); ctrl[3] = 0.0; ctrl[4] = 0.0
        # Standing (right) leg: push off with the ankle and straighten, then follow into the split.
        ctrl[13] = start[13] + s * (-foot_push - start[13])
        ctrl[11] = start[11] + s * (HOLD[11] - start[11])
        if t > T:
            ctrl[:] = HOLD
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        best = max(best, pitch_deg(d))
        if frames is not None and i % 4 == 0:
            frames.append(None)  # placeholder; rendering done by caller via callback
    return best, pitch_deg(d), sorted(floor_contacts(m, d))

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=None)
    args = p.parse_args()
    m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m)
    rows = []
    for neck_push, leg_time, foot_push, T in itertools.product((0.0, 0.3, 0.6), (0.4, 0.8, 1.5), (0.5, 1.0, 1.5), (0.8, 1.5)):
        best, end, touching = play(m, d, neck_push, leg_time, foot_push, T=T)
        rows.append((best, end, touching, dict(neck_push=neck_push, leg_time=leg_time, foot_push=foot_push, T=T)))
    rows.sort(key=lambda r: -r[0])
    print(f"{len(rows)} scripted attempts. Furthest trunk angle reached (180 = headstand), and where it ended:")
    for best, end, touching, c in rows[:8]:
        print(f"  peak {best:6.1f}  end {end:6.1f}  on {touching}  {c}")
    held = [r for r in rows if r[2] == ["jaw_soft"] and r[1] > 145]
    print(f"attempts ending on the head alone past 145 degrees: {len(held)}")
    if args.video:
        best, end, touching, c = rows[0]
        ren = mujoco.Renderer(m, 480, 640); frames = []
        d.qpos[:] = PIKE["qpos"]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        # re-play the best with rendering
        start = np.array(PIKE["qpos"][7:], dtype=float); d.ctrl[:] = start
        for _ in range(int(0.3 / m.opt.timestep)): mujoco.mj_step(m, d)
        T = c["T"]; n = int((T + 2.0) / m.opt.timestep)
        for i in range(n):
            t = i * m.opt.timestep; s = min(1.0, t / T); sl = min(1.0, t / c["leg_time"])
            ctrl = start.copy()
            ctrl[5] = start[5] + s * (HOLD[5] + c["neck_push"] - start[5]); ctrl[6] = start[6] + s * (HOLD[6] - start[6])
            ctrl[2] = start[2] + sl * (HOLD[2] - start[2]); ctrl[3] = 0.0; ctrl[4] = 0.0
            ctrl[13] = start[13] + s * (-c["foot_push"] - start[13]); ctrl[11] = start[11] + s * (HOLD[11] - start[11])
            if t > T: ctrl[:] = HOLD
            d.ctrl[:] = ctrl; mujoco.mj_step(m, d)
            if i % 10 == 0:
                cam = mujoco.MjvCamera(); cam.lookat[:] = (d.xpos[1][0], d.xpos[1][1], 0.09); cam.distance = 0.6; cam.azimuth = 90; cam.elevation = -8
                ren.update_scene(d, camera=cam); frames.append(ren.render())
        imageio.mimwrite(args.video, frames, fps=50, quality=8)
        print("video:", args.video, "best params", c)

if __name__ == "__main__":
    main()
