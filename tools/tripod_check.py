"""Headstand entry check: how much weight is on the last foot before the hop?

Zach's entry (b), Sep 18 2026: head to the floor, legs in, extend one leg
into a deep split, then a tiny calibrated hop lifts the foot that is still
on the ground. The size of that hop is the share of body weight the foot is
carrying in the tripod pose (head plus one foot). Zero means the foot just
lifts. This sweeps tripod poses, keeps the ones that settle on the head plus
one foot, and reports the foot's share of the weight, smallest first.

    uv run scripts/headstand/tripod_check.py --workers 8 --top 20
"""

import argparse
import itertools
from multiprocessing import Pool

import mujoco
import numpy as np

from settle_sweep import SCENE, pose_from, quat_from_pitch, lowest_point_z, floor_contacts

FOOT_BODIES = {"ankle_left", "ankle_right"}


def settle_tripod(m, d, joints, base_pitch, seconds=3.0, drop_height=0.01):
    mujoco.mj_resetData(m, d)
    d.qpos[7:] = joints
    d.qpos[3:7] = quat_from_pitch(base_pitch)
    d.qpos[0:3] = 0.0
    mujoco.mj_forward(m, d)
    d.qpos[2] = drop_height - lowest_point_z(m, d)
    d.ctrl[:] = joints
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    # Normal force on the floor from each body, from the contact solver.
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    force = np.zeros(6)
    load = {}
    for i in range(d.ncon):
        c = d.contact[i]
        other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
        if other is None:
            continue
        mujoco.mj_contactForce(m, d, i, force)
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other])
        load[body] = load.get(body, 0.0) + abs(force[0])  # force[0] is the contact normal
    total = sum(load.values()) or 1.0
    foot_share = sum(v for b, v in load.items() if b in FOOT_BODIES) / total
    return dict(
        trunk_up_z=float(d.xmat[1].reshape(3, 3)[2, 2]),
        touching=floor_contacts(m, d),
        foot_share=float(foot_share),
        load={b: round(v, 3) for b, v in load.items()},
    )


def is_tripod(r):
    t = r["touching"]
    feet = t & FOOT_BODIES
    return r["trunk_up_z"] < -0.6 and len(feet) == 1 and (t - feet) <= {"jaw_soft", "yaw_roll_motion", "neck_pitch"}


_model = None


def _worker_init():
    global _model
    m = mujoco.MjModel.from_xml_path(SCENE)
    _model = (m, mujoco.MjData(m))


def _run_one(c):
    m, d = _model
    joints = pose_from(c["neck_pitch"], c["head_pitch"], c["lean"], c["split"], 0.0, c["knee"])
    r = settle_tripod(m, d, joints, c["base_pitch"])
    r["pose"] = c
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    # Wider lean and split than the headstand sweep, so one leg can reach the floor.
    grid = dict(
        neck_pitch=[0.25, 0.5, 0.75, 1.0],
        head_pitch=[0.5, 1.0, 1.5],
        lean=[-0.8, -0.4, 0.0, 0.4, 0.8],
        split=[0.8, 1.0, 1.2, 1.4, 1.57],
        knee=[0.0, 0.5, 1.0],
        base_pitch=[np.pi - 0.8, np.pi - 0.4, np.pi, np.pi + 0.4, np.pi + 0.8],
    )
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    print(f"{len(combos)} poses, {args.workers} workers")
    with Pool(args.workers, initializer=_worker_init) as pool:
        results = pool.map(_run_one, combos, chunksize=16)

    tripods = [r for r in results if is_tripod(r)]
    print(f"settled as head plus one foot: {len(tripods)} / {len(results)}")
    tripods.sort(key=lambda r: r["foot_share"])
    for r in tripods[: args.top]:
        pose = {k: round(float(v), 2) for k, v in r["pose"].items()}
        print(f"foot_share={r['foot_share']:.2f} up_z={r['trunk_up_z']:+.3f} load={r['load']} {pose}")


if __name__ == "__main__":
    main()
