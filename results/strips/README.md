# Headstand

Static poses that balance the Microduck on its head in sim, found by
`scripts/headstand/settle_sweep.py` on Sep 18 2026. Each render is the pose
after a 3 second settle under the XML position servos on the `allcollisions`
model.

| Render | Pose | Result |
|---|---|---|
| `headstand_candidate_2026-09-18.png` | tucked: neck 1.0, head 1.5, hips 0.5, knees -1.5 | on the head alone, trunk 25° off vertical, 5 of 20 noisy starts land |
| `headstand_split_side_2026-09-18.png`, `headstand_split_three_quarter_2026-09-18.png` | split: neck 0.75, head 1.0, lean 0.4, split 1.2, knees straight | on the head alone, trunk 16° off vertical, feet at 15 cm, 6 of 20 noisy starts land |

Joint values are radians in the AGENTS.md joint order. "Split" is the
front/back hip pitch split, one leg forward and one back. The symmetric
legs-up pose and the side straddle did not survive the sweep.

## The training target (Sep 18 2026, later the same day)

`headstand_target_side_2026-09-18.png`: left hip 1.2, right hip 0.8, neck 1.0, head 1.25,
everything else 0. This is `HEADSTAND_OVERRIDES` in `microduck_headstand_env_cfg.py`. Trunk
z 0.117 at rest, feet at 0.16 m, 15 of 30 noisy drops land. It replaced the 1.6 rad pose
above because that one sat on the left hip's hard limit.
