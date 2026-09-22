"""Counterexamples for the routine's success criteria and training overrides."""

import numpy as np
import torch

import eval_checkpoint as evaluate


def test_leg_shapes_reject_tucks_and_wrong_split():
    q = np.zeros((4, 14))
    q[1, 2], q[1, 11] = 1.2, 0.8
    q[2] = q[1]
    q[2, 3] = 0.8
    q[3, 2], q[3, 11] = -0.8, -1.2
    assert evaluate.leg_shape_ok(q, "legs_together").tolist() == [True, False, False, False]
    assert evaluate.leg_shape_ok(q, "split").tolist() == [False, True, False, True]


def test_finishing_then_falling_is_not_success():
    assert evaluate.routine_success([9, 9, 8], 9, [True, False, True]).tolist() == [True, False, False]


def test_headstand_rejects_foot_support_and_other_body_contact(monkeypatch):
    monkeypatch.setattr(evaluate, "contacts", lambda _: {
        "head": np.array([True, True, True, False]),
        "feet": np.array([0, 1, 0, 0]),
        "other": np.array([False, False, True, False]),
    })
    monkeypatch.setattr(evaluate.microduck_mdp, "_inverted_cos", lambda _: torch.ones(4))
    assert evaluate.in_headstand(None, None).tolist() == [True, False, False, False]


def test_pike_requires_both_feet(monkeypatch):
    monkeypatch.setattr(evaluate, "contacts", lambda _: {
        "head": np.array([True, True]), "feet": np.array([2, 1]),
        "other": np.array([False, False]),
    })
    monkeypatch.setattr(evaluate.microduck_mdp, "_nose_up", lambda _: torch.full((2,), -0.9))
    monkeypatch.setattr(evaluate.microduck_mdp, "_trunk_pitch", lambda _: torch.full((2,), 1.3))
    assert evaluate.in_pike(None, None).tolist() == [True, False]


def test_standing_requires_two_feet_and_upright(monkeypatch):
    monkeypatch.setattr(evaluate, "contacts", lambda _: {
        "head": np.zeros(3, dtype=bool), "feet": np.array([2, 1, 2]),
        "other": np.zeros(3, dtype=bool),
    })
    monkeypatch.setattr(evaluate.microduck_mdp, "_inverted_cos", lambda _: torch.tensor([-1., -1., 0.]))
    assert evaluate.is_standing(None, None).tolist() == [True, False, False]


def test_variant_applies_historical_switch_settings():
    from train_variant import make_config
    cfg = make_config("Mjlab-HeadstandSplitSwitch-Flat-MicroDuck",
                      {"polish_at": 0, "handover_prob": 0.3, "switch_ramp_s": 0.3})
    assert cfg.env.events["set_headstand_spawn"].params["handover_prob"] == 0.3
    stages = cfg.env.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages == [{"step": 0, "weight": -0.46}]
    assert cfg.env.commands["twist"].ramp_s == 0.3
