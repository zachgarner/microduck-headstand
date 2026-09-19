"""Show a headstand pose in the MuJoCo viewer: drop the duck into it, hold, repeat.

    .venv/bin/mjpython scripts/headstand/show_pose.py            # the split pose
    .venv/bin/mjpython scripts/headstand/show_pose.py --tucked   # the tucked pose

Every 6 seconds the duck is dropped again from a noisy start, so you see both
the landings and the misses. Close the window to stop.
"""

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, "scripts/headstand")
from settle_sweep import SCENE, pose_from, quat_from_pitch, lowest_point_z  # noqa: E402

SPLIT = pose_from(0.75, 1.0, 0.4, 1.2, 0.0, 0.0)
TUCKED = pose_from(1.0, 1.5, 0.5, 0.0, 0.0, -1.5)


def drop(m, d, joints, rng, joint_noise, pitch_noise):
    mujoco.mj_resetData(m, d)
    d.qpos[7:] = joints + rng.uniform(-joint_noise, joint_noise, size=14)
    d.qpos[3:7] = quat_from_pitch(np.pi + rng.uniform(-pitch_noise, pitch_noise))
    mujoco.mj_forward(m, d)
    d.qpos[2] = 0.01 - lowest_point_z(m, d)
    d.ctrl[:] = joints


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tucked", action="store_true")
    p.add_argument("--noise", type=float, default=0.05, help="joint noise in rad on each drop")
    p.add_argument("--hold", type=float, default=6.0, help="seconds between drops")
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    joints = TUCKED if args.tucked else SPLIT
    rng = np.random.default_rng(0)
    drop(m, d, joints, rng, 0.0, 0.0)  # first drop is the clean pose
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = (0.0, 0.0, 0.08)
        v.cam.distance = 0.5
        v.cam.azimuth = 135
        v.cam.elevation = -8
        last_drop = time.time()
        while v.is_running():
            t0 = time.time()
            mujoco.mj_step(m, d)
            v.sync()
            if time.time() - last_drop > args.hold:
                drop(m, d, joints, rng, args.noise, 0.15)
                last_drop = time.time()
            time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()
