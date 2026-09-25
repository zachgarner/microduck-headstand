"""Regression tests for the sweep harness's pure functions. No simulation.

    cd microduck_rl
    uv run --with pytest --with matplotlib pytest ../tools/test_sweep.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "sweep"))
import report  # noqa: E402
import rollout  # noqa: E402
import run_sweep  # noqa: E402


def test_grid_is_the_full_product_in_axis_order():
    points = run_sweep.grid_points({"a": [1, 2], "b": [10, 20, 30]})
    assert len(points) == 6
    assert points[0] == {"a": 1, "b": 10} and points[-1] == {"a": 2, "b": 30}


def test_no_axes_is_one_point():
    assert run_sweep.grid_points({}) == [{}]


def test_layout_keeps_grid_points_contiguous_and_caps_batches():
    batches = run_sweep.layout([{"a": 1}, {"a": 2}, {"a": 3}], attempts=4, max_envs=5)
    assert [len(b) for b in batches] == [5, 5, 2]
    flat = [slot for b in batches for slot in b]
    assert flat == [(p, a) for p in range(3) for a in range(4)]


def test_unpinned_routine_layout_matches_the_verified_seeds():
    # 96 attempts in batches of 32 run at seeds 0, 1 and 2, the evaluation's seeds.
    batches = run_sweep.layout([{}], attempts=96, max_envs=32)
    assert len(batches) == 3 and all(len(b) == 32 for b in batches)


def test_wilson_interval_bounds():
    lo, hi = report.wilson(32, 32)
    assert hi == pytest.approx(1.0) and 0.85 < lo < 0.92
    lo, hi = report.wilson(0, 16)
    assert lo == pytest.approx(0.0) and 0.15 < hi < 0.25


def test_pitch_offset_rotates_about_the_body_pitch_axis():
    qpos = np.zeros((1, 21)); qpos[0, 3] = 1.0
    qvel = np.zeros((1, 20))
    q, _ = rollout.perturb_states(qpos, qvel, {"handover_pitch_offset": np.array([90.0])}, seed=0)
    assert q[0, 3:7] == pytest.approx([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
    assert qpos[0, 3] == 1.0   # the source states are not modified


def test_zero_offsets_leave_states_unchanged():
    rng = np.random.default_rng(1)
    qpos, qvel = rng.normal(size=(4, 21)), rng.normal(size=(4, 20))
    zeros = np.zeros(4)
    q, v = rollout.perturb_states(qpos, qvel, {"handover_pitch_offset": zeros,
                                               "handover_pitch_rate_offset": zeros,
                                               "handover_joint_noise": zeros}, seed=0)
    assert np.array_equal(q, qpos) and np.array_equal(v, qvel)


def test_joint_noise_touches_only_the_servos():
    qpos, qvel = np.zeros((64, 21)), np.zeros((64, 20))
    q, v = rollout.perturb_states(qpos, qvel, {"handover_joint_noise": np.full(64, 0.1)}, seed=0)
    assert np.all(q[:, :7] == 0) and np.all(v == 0)
    assert 0.08 < q[:, 7:].std() < 0.12
