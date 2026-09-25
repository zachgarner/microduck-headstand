"""Sweep axes: named physics parameters pinned to one value per environment.

Training draws each of these at random for every environment. A sweep pins one
of them instead, so environment i runs at the i-th grid value and every other
parameter keeps its training draw. The pin runs after the reset, when the
task's own randomization has already fired, and goes through the same
functions and tensors the task uses. It restores the nominal value before
applying the pinned one, so nothing accumulates.

Each axis also reads its current value back per environment. The harness
records that readback for every axis in every attempt, pinned or not, so an
unpinned draw is evidence too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from mjlab.utils.buffers.delay_buffer import DelayBuffer

from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
from mjlab_microduck.tasks import mdp as m


@dataclass(frozen=True)
class Axis:
    name: str
    unit: str
    nominal: float
    training_range: tuple[float, float]
    pin: Callable        # pin(env, env_ids, value)
    read: Callable       # read(env) -> np.ndarray of shape (num_envs,)
    note: str = ""


def _bam(env):
    """The robot's BAM actuators. The headstand models have one group of 14."""
    return [a for a in env.scene["robot"].actuators if isinstance(a, FrictionDRBamActuator)]


def _ids(env, env_ids):
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.int)


def _fire(env, term_name, env_ids, **overrides):
    """Run one of the task's event terms on some environments with overridden
    parameters, then recompute the model constants the term invalidates."""
    term = env.event_manager.get_term_cfg(term_name)
    term.func(env, _ids(env, env_ids), **{**term.params, **overrides})
    level = getattr(term.func, "recompute", None)
    if level is not None and level.value > 0:
        env.sim.recompute_constants(level)


def _event_ids(env, term_name):
    return env.event_manager.get_term_cfg(term_name).params["asset_cfg"]


# ── Actuator ─────────────────────────────────────────────────────────────────

def _pin_friction(env, env_ids, value):
    _fire(env, "randomize_joint_friction", env_ids, scale_range=(value, value))


def _read_friction(env):
    return _bam(env)[0].friction_scale[:, 0].cpu().numpy()


def _pin_voltage(env, env_ids, value):
    for a in _bam(env):
        a.vin_tensor[_ids(env, env_ids).long()] = value


def _read_voltage(env):
    return _bam(env)[0].vin_tensor[:, 0].cpu().numpy()


def _pin_drop_gain(env, env_ids, value):
    for a in _bam(env):
        a.vin_drop_gain[_ids(env, env_ids).long()] = value


def _read_drop_gain(env):
    return _bam(env)[0].vin_drop_gain[:, 0].cpu().numpy()


class PinnedDelayBuffer(DelayBuffer):
    """A delay buffer whose lag is fixed per environment. A lag of -1 keeps the
    task's random draw for that environment, resampled every physics step."""

    def __init__(self, source: DelayBuffer, max_lag: int):
        super().__init__(min_lag=0, max_lag=max(max_lag, source.max_lag),
                         batch_size=source.batch_size, device=source.device)
        self.random_min, self.random_max = source.min_lag, source.max_lag
        self.pinned = torch.full((source.batch_size,), -1, dtype=torch.long, device=source.device)

    def _sample_lags(self, mask):
        random = torch.randint(self.random_min, self.random_max + 1, (self.batch_size,),
                               dtype=torch.long, device=self.device)
        lags = torch.where(self.pinned >= 0, self.pinned, random)
        return torch.where(mask, lags, self._current_lags)


def _pin_delay(env, env_ids, value, max_lag=16):
    for a in _bam(env):
        if not isinstance(a._delay_buffer, PinnedDelayBuffer):
            a._delay_buffer = PinnedDelayBuffer(a._delay_buffer, max_lag)
        a._delay_buffer.pinned[_ids(env, env_ids).long()] = int(round(value))


def _read_delay(env):
    buf = _bam(env)[0]._delay_buffer
    if isinstance(buf, PinnedDelayBuffer):
        return torch.where(buf.pinned >= 0, buf.pinned, -1).cpu().numpy().astype(float)
    return np.full(env.num_envs, -1.0)   # random every physics step


# ── Body ─────────────────────────────────────────────────────────────────────

def _pin_com(term_name, axis):
    def pin(env, env_ids, value):
        _fire(env, term_name, env_ids, ranges={axis: (value, value)}, axes=[axis])
    return pin


def _read_com(term_name, axis):
    def read(env):
        asset_cfg = _event_ids(env, term_name)
        body = env.scene["robot"].indexing.body_ids[asset_cfg.body_ids][0]
        now = env.sim.model.body_ipos[:, body, axis]
        default = env.sim.get_default_field("body_ipos")[body, axis]
        return (now - default).cpu().numpy()
    return read


def _pin_mass(env, env_ids, value):
    # pseudo_inertia rebuilds the trunk's mass, inertia and CoM from the model
    # defaults, which erases the reset's CoM draw. Redraw the CoM afterwards so
    # the trunk's CoM still follows the task's distribution.
    alpha = math.log(value) / 2.0
    _fire(env, "randomize_mass_inertia", env_ids, alpha_range=(alpha, alpha))
    if "randomize_com" in env.event_manager.active_terms.get("reset", []):
        _fire(env, "randomize_com", env_ids)


def _read_mass(env):
    asset_cfg = _event_ids(env, "randomize_mass_inertia")
    body = env.scene["robot"].indexing.body_ids[asset_cfg.body_ids][0]
    return (env.sim.model.body_mass[:, body] / env.sim.get_default_field("body_mass")[body]).cpu().numpy()


def _pin_armature(env, env_ids, value):
    _fire(env, "randomize_armature", env_ids, ranges=(value, value))


def _read_armature(env):
    dof = env.scene["robot"].indexing.joint_v_adr[0]
    now = env.sim.model.dof_armature[:, dof]
    return (now / env.sim.get_default_field("dof_armature")[dof]).cpu().numpy()


# ── Sensors ──────────────────────────────────────────────────────────────────

def _pin_encoder_bias(env, env_ids, value):
    """Each joint's bias drawn from ±value: the axis is the calibration error's
    size, not one fixed offset shared by every joint."""
    _fire(env, "encoder_bias", env_ids, bias_range=(-value, value))


def _read_encoder_bias(env):
    return env.scene["robot"].data.encoder_bias.abs().max(dim=-1).values.cpu().numpy()


def _pin_imu(env, env_ids, value):
    """A fixed IMU mounting error of `value` degrees about a random axis."""
    q = m._imu_misalignment_quat(env, math.radians(IMU_MAX_DEG))
    ids = _ids(env, env_ids).long()
    axis = torch.randn(len(ids), 3, device=env.device)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    angle = torch.full((len(ids),), math.radians(value), device=env.device)
    q[ids] = m.quat_from_angle_axis(angle, axis)


def _read_imu(env):
    q = m._imu_misalignment_quat(env, math.radians(IMU_MAX_DEG))
    return np.degrees(2.0 * torch.acos(q[:, 0].abs().clamp(max=1.0)).cpu().numpy())


# ── Contact ──────────────────────────────────────────────────────────────────

def _pin_contact_friction(env, env_ids, value):
    """Scale the sliding friction of every geom. MuJoCo combines a contact's
    two friction values with max, so scaling the floor alone would do nothing
    below the robot's own friction."""
    if "geom_friction" not in env.sim.expanded_fields:
        env.sim.expand_model_fields(("geom_friction",))
    ids = _ids(env, env_ids).long()
    default = env.sim.get_default_field("geom_friction")[:, 0]
    env.sim.model.geom_friction[ids, :, 0] = default * value


def _read_contact_friction(env):
    if "geom_friction" not in env.sim.expanded_fields:
        return np.ones(env.num_envs)
    default = env.sim.get_default_field("geom_friction")[:, 0]
    now = env.sim.model.geom_friction[:, :, 0]
    return (now / default.clamp(min=1e-9)).max(dim=-1).values.cpu().numpy()


# ── Gear play ────────────────────────────────────────────────────────────────

def _backlash_joints(env):
    asset = env.scene["robot"]
    if not any(n.endswith("_backlash") for n in asset.joint_names):
        raise KeyError("backlash joints: run this axis with \"robot\": \"backlash\"")
    local, _ = asset.find_joints((r"passive_.*_backlash",))
    return asset.indexing.joint_ids[torch.as_tensor(local)].long()


def _pin_backlash(env, env_ids, value):
    """Total play in degrees, peak to peak, the unit add_backlash.py uses. Each
    backlash hinge's range becomes plus or minus half of it. Zero play keeps a
    range of 1e-5 rad so the limit constraint stays well defined."""
    joints = _backlash_joints(env)
    if "jnt_range" not in env.sim.expanded_fields:
        env.sim.expand_model_fields(("jnt_range",))
    half = max(math.radians(value) / 2.0, 1e-5)
    ids = _ids(env, env_ids).long()
    rng = env.sim.model.jnt_range
    rng[ids.unsqueeze(1), joints.unsqueeze(0), 0] = -half
    rng[ids.unsqueeze(1), joints.unsqueeze(0), 1] = half


def _read_backlash(env):
    joints = _backlash_joints(env)
    rng = env.sim.model.jnt_range
    if rng.dim() == 2:   # not expanded: the XML's range, shared by every environment
        width = (rng[joints, 1] - rng[joints, 0]).max()
        return np.full(env.num_envs, math.degrees(float(width)))
    width = (rng[:, joints, 1] - rng[:, joints, 0]).max(dim=-1).values
    deg = np.degrees(width.cpu().numpy())
    return np.where(deg < 0.01, 0.0, deg)


IMU_MAX_DEG = 6.0   # IMU_ORIENTATION_RANDOMIZATION_ANGLE in the headstand cfg

# Training ranges are the headstand task's final values. The CoM ranges are
# the curriculum's end (15 mm trunk, 10 mm head), not the 3 mm start that an
# evaluator with its curriculum cleared would draw from.
AXES = {a.name: a for a in [
    Axis("friction_scale", "x nominal", 1.0, (0.9, 1.1), _pin_friction, _read_friction,
         "Scales the BAM actuator's friction budget: Coulomb, Stribeck and load-dependent."),
    Axis("battery_voltage", "V", 7.4, (6.5, 8.2), _pin_voltage, _read_voltage,
         "Above 8.2 V the MuJoCo force range, set from the top of the training range, clips the torque."),
    Axis("voltage_drop_gain", "V/Nm", 0.1, (0.0, 0.2), _pin_drop_gain, _read_drop_gain,
         "Voltage sag per newton-metre of total servo torque."),
    Axis("command_delay", "physics steps (5 ms)", 4.5, (3, 6), _pin_delay, _read_delay,
         "Training resamples the lag every physics step. A pinned lag is constant."),
    Axis("trunk_com_x", "m", 0.0, (-0.015, 0.015), _pin_com("randomize_com", 0), _read_com("randomize_com", 0)),
    Axis("trunk_com_y", "m", 0.0, (-0.015, 0.015), _pin_com("randomize_com", 1), _read_com("randomize_com", 1)),
    Axis("trunk_com_z", "m", 0.0, (-0.015, 0.015), _pin_com("randomize_com", 2), _read_com("randomize_com", 2)),
    Axis("head_com_x", "m", 0.0, (-0.010, 0.010), _pin_com("randomize_head_com", 0), _read_com("randomize_head_com", 0)),
    Axis("head_com_z", "m", 0.0, (-0.010, 0.010), _pin_com("randomize_head_com", 2), _read_com("randomize_head_com", 2)),
    Axis("trunk_mass_scale", "x nominal", 1.0, (0.95, 1.05), _pin_mass, _read_mass,
         "Scales the trunk's mass and inertia together."),
    Axis("armature_scale", "x nominal", 1.0, (0.9, 1.1), _pin_armature, _read_armature),
    Axis("encoder_bias", "rad", 0.0, (0.0, 0.015), _pin_encoder_bias, _read_encoder_bias,
         "Each joint's bias is drawn from plus or minus this value. Readback is the largest joint bias."),
    Axis("imu_misalignment", "deg", 0.0, (0.0, 6.0), _pin_imu, _read_imu,
         "A fixed mounting error of this angle about a random axis."),
    Axis("contact_friction_scale", "x nominal", 1.0, (1.0, 1.0), _pin_contact_friction, _read_contact_friction,
         "Training never randomizes contact friction."),
    Axis("backlash_deg", "deg total", 0.0, (0.0, 0.0), _pin_backlash, _read_backlash,
         "Needs the backlash robot model. The headstand policies trained without backlash; "
         "the -Backlash- tasks use 2 degrees."),
]}


def pin_grid(env, assignments: dict[str, np.ndarray]):
    """Pin each named axis to its per-environment values.

    NaN leaves an environment on its training draw. The trunk mass goes first
    because it rebuilds the trunk's CoM, which a CoM pin must then override.
    """
    order = sorted(assignments, key=lambda n: 0 if n == "trunk_mass_scale" else 1)
    for name in order:
        values = np.asarray(assignments[name], dtype=float)
        for value in np.unique(values[~np.isnan(values)]):
            AXES[name].pin(env, np.where(values == value)[0], float(value))


def read_all(env) -> dict[str, np.ndarray]:
    out = {}
    for name, axis in AXES.items():
        try:
            out[name] = axis.read(env)
        except KeyError:   # the task has no such event term
            out[name] = np.full(env.num_envs, np.nan)
    return out
