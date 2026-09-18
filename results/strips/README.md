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
