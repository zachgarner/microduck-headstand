"""Headstand physics check: does any static pose balance the duck on its head?

AGENTS.md step 2 says to verify a target pose is a stable equilibrium in sim
before training anything. This sweeps neck, head and leg joint angles, drops
the robot inverted onto the floor, holds each pose with the XML position
servos for a few seconds, and reports which poses end up still inverted and
resting on the head only.

    uv run scripts/headstand/settle_sweep.py            # full sweep, prints survivors
    uv run scripts/headstand/settle_sweep.py --top 20   # print the 20 best by tilt
"""

import argparse
import itertools

import mujoco
import numpy as np

SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"

# Joint order in qpos after the 7 freejoint values (AGENTS.md joint layout):
# 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5-8 neck/head
# (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg.
LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE = 2, 3, 4
NECK_PITCH, HEAD_PITCH = 5, 6
RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE = 11, 12, 13

# Bodies allowed to touch the floor in a headstand. Anything else touching
# means the pose is a flop or a tripod, not a headstand.
HEAD_BODIES = {"jaw_soft", "yaw_roll_motion", "neck_pitch"}


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

    Returns trunk_up_z (the trunk's own z axis dotted with world up; -1 is a
    perfect headstand), the whole-body CoM height, and the set of bodies on
    the floor at the end.
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
    steps = int(seconds / m.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    trunk_up_z = d.xmat[1].reshape(3, 3)[2, 2]
    return trunk_up_z, d.subtree_com[1][2], floor_contacts(m, d)


def symmetric_pose(neck_pitch, head_pitch, hip_pitch, knee, ankle):
    # Left and right legs mirror each other with opposite signs, as in the
    # STAND keyframe (left hip_pitch -0.458, right +0.458).
    j = np.zeros(14)
    j[LEFT_HIP_PITCH], j[LEFT_KNEE], j[LEFT_ANKLE] = hip_pitch, knee, ankle
    j[RIGHT_HIP_PITCH], j[RIGHT_KNEE], j[RIGHT_ANKLE] = -hip_pitch, -knee, -ankle
    j[NECK_PITCH], j[HEAD_PITCH] = neck_pitch, head_pitch
    return j


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=15, help="how many results to print, best first")
    p.add_argument("--seconds", type=float, default=3.0)
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)

    grid = dict(
        neck_pitch=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        head_pitch=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        hip_pitch=[-0.5, 0.0, 0.5],
        knee=[-1.5, -0.8, 0.0, 0.8, 1.5],
        ankle=[0.0],
        base_pitch=[np.pi - 0.5, np.pi, np.pi + 0.5],
    )
    keys = list(grid)
    results = []
    combos = list(itertools.product(*grid.values()))
    print(f"{len(combos)} poses, {args.seconds:.1f} s each")
    for combo in combos:
        c = dict(zip(keys, combo))
        joints = symmetric_pose(c["neck_pitch"], c["head_pitch"], c["hip_pitch"], c["knee"], c["ankle"])
        up_z, com_z, touching = settle(m, d, joints, c["base_pitch"], seconds=args.seconds)
        head_only = bool(touching) and touching <= HEAD_BODIES
        results.append((up_z, com_z, head_only, touching, c))

    inverted = [r for r in results if r[0] < -0.8]
    headstands = [r for r in inverted if r[2]]
    print(f"still inverted after settle: {len(inverted)} / {len(results)}")
    print(f"inverted AND resting on head bodies only: {len(headstands)}")
    results.sort(key=lambda r: (not r[2], r[0]))
    for up_z, com_z, head_only, touching, c in results[: args.top]:
        pose = {k: round(float(v), 2) for k, v in c.items()}
        print(f"up_z={up_z:+.3f} com_z={com_z:.3f} head_only={head_only} touching={sorted(touching)} {pose}")


if __name__ == "__main__":
    main()
