"""No-reset rollouts for sweeps: one policy from one start, or the full routine.

Both modes build the environment the verified evaluators build (`check32.py`
and `routine_full.py`): play config, curriculum cleared, terminations off,
automatic resets off. After the reset, `axes.pin_grid` pins the sweep's axes.
Every attempt then returns one record and the states it passed through.

States use the handover-set layout (qpos [21], qvel [20] on the plain model,
base position relative to the environment's origin), so a recorded state can seed a
training task's handover bucket or a later replay.
"""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
from eval_checkpoint import force_spawn, in_headstand, in_pike, in_shaped_headstand, is_standing, routine_success  # noqa: E402
import routine_full  # noqa: E402
from routine_full import TASK as ROUTINE_TASK, flagged, onnx_policy, torch_policy  # noqa: E402
from mjlab_microduck.tasks import mdp as m  # noqa: E402

import axes as sweep_axes  # noqa: E402

STEP_DT = 0.02


def make_env(task: str, num_envs: int, seed: int, episode_s: float | None = None, video: bool = False,
             robot: str = "plain"):
    """`robot="backlash"` swaps in the all-collisions model with a passive
    backlash hinge after every servo, and the encoder observations that read
    through it, the same conversion the `-Backlash-` tasks use. The policies
    keep their 61-value observation and 14 actions."""
    cfg = load_env_cfg(task, play=True)
    if robot == "backlash":
        from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
        from mjlab_microduck.tasks.backlash import make_backlash_variant
        cfg = make_backlash_variant(cfg, MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG)
    elif robot != "plain":
        raise ValueError(f"unknown robot {robot!r}; use plain or backlash")
    cfg.scene.num_envs = num_envs
    cfg.curriculum.clear()
    cfg.seed = seed
    cfg.auto_reset = False
    if episode_s is not None:
        cfg.episode_length_s = episode_s
    for name in list(cfg.terminations):
        if name != "time_out":
            del cfg.terminations[name]
    cfg.viewer.distance = 0.7; cfg.viewer.elevation = -8; cfg.viewer.azimuth = 135
    cfg.viewer.height = 480; cfg.viewer.width = 640
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array" if video else None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)
    return env, wrapped


def robot_state(env) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel of every environment, base position relative to its origin."""
    qpos = env.sim.data.qpos.cpu().numpy().copy()
    qpos[:, 0:3] -= env.scene.env_origins.cpu().numpy()
    return qpos, env.sim.data.qvel.cpu().numpy().copy()


def write_state(env, env_ids, qpos: np.ndarray, qvel: np.ndarray):
    """Place environments in recorded states at their own origins. Writes qpos
    and qvel directly, the way the kick-up task's handover spawn does."""
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    q = torch.as_tensor(qpos, dtype=torch.float32, device=env.device).clone()
    q[:, 0:3] += env.scene.env_origins[ids]
    env.sim.data.qpos[ids] = q
    env.sim.data.qvel[ids] = torch.as_tensor(qvel, dtype=torch.float32, device=env.device)
    env.sim.forward()


def start(env, wrapped, assignments: dict[str, np.ndarray], start_states=None):
    """Reset, optionally place recorded start states, pin the axes, and return
    the first observation computed after the pins."""
    wrapped.reset()
    if start_states is not None:
        write_state(env, np.arange(env.num_envs), *start_states)
    sweep_axes.pin_grid(env, assignments)
    return wrapped.get_observations()


def load_policy(task, spec, wrapped):
    """A trained checkpoint `run:model_N.pt` of `task`, or an ONNX path."""
    if spec.endswith(".onnx"):
        return onnx_policy(spec)
    return torch_policy(task, spec, wrapped)


def _goal(env, asset, goal: str) -> np.ndarray:
    if goal == "pike":
        return in_pike(env, asset)
    if goal == "standing":
        return is_standing(env, asset)
    if goal in ("legs_together", "split"):
        return in_shaped_headstand(env, asset, goal)
    if goal == "headstand":
        return in_headstand(env, asset)
    raise ValueError(f"unknown goal {goal!r}")


def default_goal(task: str) -> str:
    if "Fold" in task:
        return "pike"
    if "Backroll" in task or "SplitOver" in task:
        return "standing"
    return "legs_together" if "LegsTogether" in task else "split"


# ── Single policy ────────────────────────────────────────────────────────────

def run_policy(env, wrapped, policy, obs, goal: str, steps: int = 298, flag: float | None = None):
    """Run one policy for `steps` control steps. Success is the goal state at
    the last step, the same criterion `check32.py` reports as strict success."""
    asset = env.scene["robot"]
    if flag is not None:
        policy = flagged(policy, flag)
    N = env.num_envs
    first = np.full(N, -1.0)
    with torch.no_grad():
        for i in range(steps):
            obs, *_ = wrapped.step(policy(obs))
            reached = _goal(env, asset, goal)
            first[(first < 0) & reached] = (i + 1) * STEP_DT
    success = _goal(env, asset, goal)
    qpos, qvel = robot_state(env)
    return {
        "success": success, "first_goal_s": first,
        "resets": int((env.episode_length_buf < steps).sum()),
        "final_qpos": qpos, "final_qvel": qvel,
    }


# ── Full routine ─────────────────────────────────────────────────────────────

def routine_stages(env, wrapped, spec: dict):
    """The stage list of `routine_full.py` with the v16 options: one switch,
    the split-over exit, and the standing policy settling at the end."""
    fold = load_policy(ROUTINE_TASK["fold"], spec["fold"], wrapped)
    stand = load_policy(None, spec["stand"], wrapped)
    return [
        ("fold",          fold, "pike", 0.3 + spec["pike_settle_s"]),
        ("legs_together", load_policy(ROUTINE_TASK["legs_together"], spec["legs_together"], wrapped), "legs_together", spec["hold_s"]),
        ("roll",          load_policy(ROUTINE_TASK["roll"], spec["roll"], wrapped), "standing", 0.3),
        ("settle",        stand, "standing", spec["settle_s"]),
        ("fold2",         fold, "pike", 0.3 + spec["pike_settle_s"]),
        ("split",         load_policy(ROUTINE_TASK["split"], spec["split"], wrapped), "split", spec["hold_s"]),
        ("switch",        flagged(load_policy(ROUTINE_TASK["switch"], spec["switch"], wrapped), 1.0), "mirrored", spec["switch_hold_s"]),
        ("splitover",     load_policy(ROUTINE_TASK["splitover"], spec["splitover"], wrapped), "standing", 0.3),
        ("settle2",       stand, "standing", 1.0),
    ]


def assert_standing_starts(env, asset):
    """The checks `routine_full.py` makes before anything is reported."""
    from eval_checkpoint import contacts
    c0 = contacts(env)
    inv0 = m._inverted_cos(asset).numpy()
    z0 = (asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).numpy()
    upright0 = (inv0 < -math.cos(math.radians(30))) & ~c0["head"] & (z0 > 0.10) & (z0 < 0.13)
    assert upright0.all(), f"{int((~upright0).sum())} of {env.num_envs} episodes did not start upright at standing height"
    assert not in_pike(env, asset).any(), "an episode started in the pike"
    assert not in_headstand(env, asset).any(), "an episode started in a headstand"


def run_routine(env, wrapped, stages, obs, seconds: float = 18.0):
    """Run the routine with the stage machine of `routine_full.py`. A stage is
    done when its goal has held for its hold time, and the next policy takes
    over from that step. The state at each takeover is recorded, so every
    handover the routine made can be replayed."""
    from mjlab_microduck.tasks.microduck_headstand_env_cfg import HEADSTAND_OVERRIDES
    asset = env.scene["robot"]
    N, S = env.num_envs, len(stages)
    split_t = m._servo_default_joint_pos(env, asset).clone(); mirror_t = split_t.clone()
    for i_, v_ in HEADSTAND_OVERRIDES.items():
        split_t[:, i_] = v_
    for i_, v_ in m._mirror_overrides(HEADSTAND_OVERRIDES).items():
        mirror_t[:, i_] = v_
    stage = np.zeros(N, dtype=int); held = np.zeros(N)
    t_stage = np.full((N, S), -1.0)
    nq, nv = env.sim.data.qpos.shape[1], env.sim.data.qvel.shape[1]
    handover_qpos = np.full((N, S, nq), np.nan, dtype=np.float32)
    handover_qvel = np.full((N, S, nv), np.nan, dtype=np.float32)
    steps = int(seconds / env.step_dt)
    with torch.no_grad():
        for i in range(steps):
            t = (i + 1) * env.step_dt
            acts = torch.zeros(N, 14)
            for k, (_, pol, _, _) in enumerate(stages):
                idx = np.where(np.minimum(stage, S - 1) == k)[0]
                if len(idx):
                    acts[idx] = pol(obs)[idx]
            obs, *_ = wrapped.step(acts)
            q = m._servo_joint_pos(env, asset)
            nearer_mirror = (((q - mirror_t) ** 2).sum(-1) < ((q - split_t) ** 2).sum(-1)).numpy()
            split = in_shaped_headstand(env, asset, "split")
            cond = {
                "pike": in_pike(env, asset),
                "legs_together": in_shaped_headstand(env, asset, "legs_together"),
                "split": split, "mirrored": split & nearer_mirror, "original": split & ~nearer_mirror,
                "standing": is_standing(env, asset),
            }
            done_now = []
            for j in range(N):
                k = stage[j]
                if k >= S:
                    continue
                if cond[stages[k][2]][j]:
                    held[j] += env.step_dt
                    if held[j] >= stages[k][3]:
                        t_stage[j, k] = t; stage[j] += 1; held[j] = 0.0; done_now.append((j, k))
                else:
                    held[j] = 0.0
            if done_now:
                qpos, qvel = robot_state(env)
                for j, k in done_now:
                    handover_qpos[j, k] = qpos[j]; handover_qvel[j, k] = qvel[j]
    standing = is_standing(env, asset)
    qpos, qvel = robot_state(env)
    return {
        "stage": stage, "stage_times_s": t_stage, "final_standing": standing,
        "success": routine_success(stage, S, standing),
        "resets": int((env.episode_length_buf < steps).sum()),
        "handover_qpos": handover_qpos, "handover_qvel": handover_qvel,
        "final_qpos": qpos, "final_qvel": qvel,
    }


# ── One routine stage from recorded handover states ─────────────────────────

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _perturb_pitch(qpos, qvel, value, rng):
    """Rotate the trunk by `value` degrees about its own pitch axis."""
    half = math.radians(value) / 2.0
    rot = np.array([math.cos(half), 0.0, math.sin(half), 0.0])
    qpos[:, 3:7] = _quat_mul(qpos[:, 3:7], np.broadcast_to(rot, qpos[:, 3:7].shape))


def _perturb_pitch_rate(qpos, qvel, value, rng):
    """Add `value` rad/s to the trunk's pitch rate. MuJoCo's free-joint angular
    velocity is in the body frame, so index 4 is the pitch axis."""
    qvel[:, 4] += value


def _perturb_joints(qpos, qvel, value, rng):
    """Add Gaussian noise with standard deviation `value` rad to every servo
    position. Plain-model layout: the 14 servos follow the 7 base values."""
    qpos[:, 7:21] += rng.normal(0.0, value, size=qpos[:, 7:21].shape) if value > 0 else 0.0


STATE_AXES = {
    "handover_pitch_offset": ("deg", _perturb_pitch),
    "handover_pitch_rate_offset": ("rad/s", _perturb_pitch_rate),
    "handover_joint_noise": ("rad", _perturb_joints),
}


def perturb_states(qpos, qvel, assignments: dict[str, np.ndarray], seed: int):
    """Apply each state axis to its environments. Returns perturbed copies."""
    qpos, qvel = qpos.copy(), qvel.copy()
    rng = np.random.default_rng(seed)
    for name, values in assignments.items():
        fn = STATE_AXES[name][1]
        for value in np.unique(values):
            idx = np.where(values == value)[0]
            q, v = qpos[idx], qvel[idx]
            fn(q, v, float(value), rng)
            qpos[idx], qvel[idx] = q, v
    return qpos, qvel


def run_stage(env, wrapped, policy, obs, goal: str, hold_s: float, seconds: float):
    """Run one routine stage's policy until its goal has held for `hold_s`, the
    rule the routine uses to hand over. Success is completing within `seconds`.
    The observation's last-action slot starts at zero here, where the routine
    carries the previous policy's last action."""
    asset = env.scene["robot"]
    N = env.num_envs
    held = np.zeros(N); done_t = np.full(N, -1.0)
    steps = int(seconds / env.step_dt)
    with torch.no_grad():
        for i in range(steps):
            obs, *_ = wrapped.step(policy(obs))
            ok = _goal_or_switch(env, asset, goal)
            held = np.where(ok, held + env.step_dt, 0.0)
            done_t[(done_t < 0) & (held >= hold_s)] = (i + 1) * env.step_dt
    qpos, qvel = robot_state(env)
    return {"success": done_t >= 0, "done_s": done_t,
            "resets": int((env.episode_length_buf < steps).sum()),
            "final_qpos": qpos, "final_qvel": qvel}


def _goal_or_switch(env, asset, goal):
    if goal in ("mirrored", "original"):
        from mjlab_microduck.tasks.microduck_headstand_env_cfg import HEADSTAND_OVERRIDES
        split_t = m._servo_default_joint_pos(env, asset).clone(); mirror_t = split_t.clone()
        for i_, v_ in HEADSTAND_OVERRIDES.items():
            split_t[:, i_] = v_
        for i_, v_ in m._mirror_overrides(HEADSTAND_OVERRIDES).items():
            mirror_t[:, i_] = v_
        q = m._servo_joint_pos(env, asset)
        nearer = (((q - mirror_t) ** 2).sum(-1) < ((q - split_t) ** 2).sum(-1)).numpy()
        split = in_shaped_headstand(env, asset, "split")
        return split & (nearer if goal == "mirrored" else ~nearer)
    return _goal(env, asset, goal)


def checkpoint_hashes() -> dict:
    """Hashes of every checkpoint `torch_policy` has loaded in this process."""
    return dict(routine_full.CHECKPOINT_FILES)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
