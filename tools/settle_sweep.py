"""Headstand physics check: which static poses balance the duck on its head?

AGENTS.md step 2 says to verify a target pose is a stable equilibrium in sim
before training anything. This drops the robot inverted onto the floor in
each pose of a grid, holds the pose with the XML position servos for a few
seconds, and reports the poses that end up still inverted and resting on the
head only. Survivors are sorted with the feet highest first, since a
headstand wants the legs up.

    uv run scripts/headstand/settle_sweep.py --grid tucked   # the first sweep (Sep 18 2026)
    uv run scripts/headstand/settle_sweep.py --grid split    # split-leg poses, Zach's ask
    uv run scripts/headstand/settle_sweep.py --grid split --workers 8 --top 30
"""

import argparse
import itertools
from multiprocessing import Pool

import mujoco
import numpy as np

SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"

# Joint order in qpos after the 7 freejoint values (AGENTS.md joint layout):
# 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5-8 neck/head
# (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg.
LEFT_HIP_ROLL, LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE = 1, 2, 3, 4
NECK_PITCH, HEAD_PITCH = 5, 6
RIGHT_HIP_ROLL, RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE = 10, 11, 12, 13

# Bodies allowed to touch the floor in a headstand. Anything else touching
# means the pose is a flop or a tripod, not a headstand.
HEAD_BODIES = {"jaw_soft", "yaw_roll_motion", "neck_pitch"}

# Left and right legs mirror each other with opposite signs (STAND keyframe:
# left hip_pitch -0.458, right +0.458), so "lean" moves both legs the same
# way and "split" moves them apart, one forward and one back.
GRIDS = {
    "tucked": dict(
        neck_pitch=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        head_pitch=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        lean=[-0.5, 0.0, 0.5],
        split=[0.0],
        straddle=[0.0],
        knee=[-1.5, -0.8, 0.0, 0.8, 1.5],
        base_pitch=[np.pi - 0.5, np.pi, np.pi + 0.5],
    ),
    "split": dict(
        neck_pitch=[-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
        head_pitch=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        lean=[-0.4, 0.0, 0.4],
        split=[0.0, 0.4, 0.8, 1.2],
        straddle=[0.0, 0.35],   # hip_roll range is only +/-0.38 rad
        knee=[0.0],
        base_pitch=[np.pi - 0.5, np.pi, np.pi + 0.5],
    ),
}


def quat_from_pitch(pitch_rad):
    # MuJoCo quaternion is (w, x, y, z); a pure rotation about the y axis.
    return np.array([np.cos(pitch_rad / 2), 0.0, np.sin(pitch_rad / 2), 0.0])


def lowest_point_z(m, d):
    # Lowest corner of any collision geom's bounding box, in world frame.
    lowest = np.inf
    for g in range(1, m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        center, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        rot = d.geom_xmat[g].reshape(3, 3)
        for signs in itertools.product((-1, 1), repeat=3):
            corner = d.geom_xpos[g] + rot @ (center + half * np.array(signs))
            lowest = min(lowest, corner[2])
    return lowest


def floor_contacts(m, d):
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    bodies = set()
    for i in range(d.ncon):
        c = d.contact[i]
        other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
        if other is not None:
            bodies.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]))
    return bodies


def settle(m, d, joints, base_pitch, seconds=3.0, drop_height=0.01, rng=None, joint_noise=0.0):
    """Drop the robot in `joints` at orientation `base_pitch` and hold the pose.

    Returns a dict: trunk_up_z (the trunk's own z axis dotted with world up,
    -1 is a perfect headstand), com_z (whole-body CoM height), feet_z (mean
    foot height), and touching (bodies on the floor at the end).
    """
    mujoco.mj_resetData(m, d)
    q = np.array(joints, dtype=float)
    if rng is not None and joint_noise > 0:
        q = q + rng.uniform(-joint_noise, joint_noise, size=q.shape)
    d.qpos[7:] = q
    d.qpos[3:7] = quat_from_pitch(base_pitch)
    d.qpos[0:3] = 0.0
    mujoco.mj_forward(m, d)
    # Lift so the lowest collision corner sits drop_height above the floor.
    d.qpos[2] = drop_height - lowest_point_z(m, d)
    d.ctrl[:] = joints  # servos hold the commanded pose, noise or not
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")
    right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ankle_right")
    return dict(
        trunk_up_z=float(d.xmat[1].reshape(3, 3)[2, 2]),
        com_z=float(d.subtree_com[1][2]),
        feet_z=float((d.xpos[left][2] + d.xpos[right][2]) / 2),
        touching=floor_contacts(m, d),
    )


def pose_from(neck_pitch, head_pitch, lean, split, straddle, knee):
    j = np.zeros(14)
    j[LEFT_HIP_PITCH], j[RIGHT_HIP_PITCH] = lean + split, -lean + split
    j[LEFT_HIP_ROLL], j[RIGHT_HIP_ROLL] = straddle, -straddle
    j[LEFT_KNEE], j[RIGHT_KNEE] = knee, -knee
    j[NECK_PITCH], j[HEAD_PITCH] = neck_pitch, head_pitch
    return j


def is_headstand(r):
    return r["trunk_up_z"] < -0.8 and bool(r["touching"]) and r["touching"] <= HEAD_BODIES


_model = None


def _worker_init():
    global _model
    _model = (mujoco.MjModel.from_xml_path(SCENE), None)
    _model = (_model[0], mujoco.MjData(_model[0]))


def _run_one(args):
    c, seconds = args
    m, d = _model
    joints = pose_from(c["neck_pitch"], c["head_pitch"], c["lean"], c["split"], c["straddle"], c["knee"])
    r = settle(m, d, joints, c["base_pitch"], seconds=seconds)
    r["pose"] = c
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=GRIDS, default="split")
    p.add_argument("--top", type=int, default=15, help="how many results to print, best first")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    grid = GRIDS[args.grid]
    keys = list(grid)
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]
    print(f"grid {args.grid}: {len(combos)} poses, {args.seconds:.1f} s each, {args.workers} workers")
    with Pool(args.workers, initializer=_worker_init) as pool:
        results = pool.map(_run_one, [(c, args.seconds) for c in combos], chunksize=16)

    inverted = [r for r in results if r["trunk_up_z"] < -0.8]
    headstands = [r for r in results if is_headstand(r)]
    print(f"still inverted after settle: {len(inverted)} / {len(results)}")
    print(f"inverted AND resting on head bodies only: {len(headstands)}")
    # Survivors first, feet highest first; then the rest by how inverted they are.
    results.sort(key=lambda r: (not is_headstand(r), -r["feet_z"], r["trunk_up_z"]))
    for r in results[: args.top]:
        pose = {k: round(float(v), 2) for k, v in r["pose"].items()}
        print(f"up_z={r['trunk_up_z']:+.3f} com_z={r['com_z']:.3f} feet_z={r['feet_z']:.3f} "
              f"headstand={is_headstand(r)} touching={sorted(r['touching'])} {pose}")


if __name__ == "__main__":
    main()
