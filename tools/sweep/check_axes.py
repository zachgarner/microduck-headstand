"""Check that every sweep axis pins its value, keeps it, and reaches the physics.

    cd microduck_rl
    uv run ../tools/sweep/check_axes.py --task Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck

Each axis runs three times at the same seed with 8 environments: every
environment pinned low, pinned low again, and pinned high. Each run takes 50
control steps with actions computed from the observation, so the sensor axes
and the command delay change what the robot does.

- The readback must equal the pinned value before and after the steps.
- The control is the largest joint-position difference (rad) between the two
  low runs. It must be zero, or the comparison below measures noise.
- The effect is the largest servo-position difference between the low and
  the high run. An effect of zero means the pin never reached the simulation.

Prints one line per axis and exits non-zero on any failure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes as sweep_axes  # noqa: E402
import rollout  # noqa: E402

LOW_HIGH = {
    "friction_scale": (0.5, 2.0),
    "battery_voltage": (6.0, 8.0),
    "voltage_drop_gain": (0.0, 0.5),
    "command_delay": (0, 12),
    "trunk_com_x": (-0.02, 0.02),
    "trunk_com_y": (-0.02, 0.02),
    "trunk_com_z": (-0.02, 0.02),
    "head_com_x": (-0.015, 0.015),
    "head_com_z": (-0.015, 0.015),
    "trunk_mass_scale": (0.8, 1.2),
    "armature_scale": (0.5, 1.5),
    "encoder_bias": (0.0, 0.05),
    "imu_misalignment": (0.0, 12.0),
    "contact_friction_scale": (0.5, 1.5),
    "backlash_deg": (0.0, 4.0),
}
N = 8


def stored_ok(name, value, read):
    if name == "encoder_bias":   # readback is the largest joint bias, drawn within ±value
        return bool(np.all(read <= value + 1e-6) and (value == 0 or np.all(read > 0)))
    return bool(np.allclose(read, value, atol=1e-5, rtol=1e-4))


def run(task, name, value, robot="plain"):
    """Joint positions after 50 steps, and whether the value stayed pinned."""
    env, wrapped = rollout.make_env(task, N, seed=0, robot=robot)
    obs = rollout.start(env, wrapped, {name: np.full(N, value, dtype=float)})
    ok = stored_ok(name, value, sweep_axes.AXES[name].read(env))
    with torch.no_grad():
        for _ in range(50):
            obs, *_ = wrapped.step(0.3 * torch.tanh(obs["actor"][:, :14]))
    ok &= stored_ok(name, value, sweep_axes.AXES[name].read(env))
    q = rollout.m._servo_joint_pos(env, env.scene["robot"]).cpu().numpy().copy()
    env.close()
    return q, ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck")
    p.add_argument("--axes", nargs="*", default=list(LOW_HIGH))
    args = p.parse_args()
    torch.set_num_threads(8)
    failures = 0
    for name in args.axes:
        lo, hi = LOW_HIGH[name]
        robot = "backlash" if name == "backlash_deg" else "plain"
        try:
            q_lo, ok1 = run(args.task, name, lo, robot)
        except KeyError as e:
            print(f"  {name:24s} SKIP  the task has no event term {e}", flush=True)
            continue
        q_lo2, ok2 = run(args.task, name, lo, robot)
        q_hi, ok3 = run(args.task, name, hi, robot)
        control = float(np.abs(q_lo - q_lo2).max())
        effect = float(np.abs(q_lo - q_hi).max())
        ok = ok1 and ok2 and ok3 and control < 1e-6 and effect > 1e-4
        failures += not ok
        print(f"  {name:24s} {'ok  ' if ok else 'FAIL'}  stored {ok1 and ok2 and ok3}, "
              f"control {control:.2e} rad, effect {effect:.4f} rad ({lo} vs {hi})", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
