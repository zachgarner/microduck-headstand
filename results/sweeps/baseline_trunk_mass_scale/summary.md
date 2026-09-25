# baseline_trunk_mass_scale

Issue #13: the v16 routine with trunk_mass_scale pinned per environment, 32 attempts per value, every other parameter on its training draw.

224 attempts, 197 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## trunk_mass_scale

Scales the trunk's mass and inertia together.

Unit: x nominal. Nominal 1. Training range 0.95 to 1.05.

| trunk_mass_scale | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0.8 | 32 | 30/32 (94%) | 80% to 98% | legs_together 1, switch 1 |
| 0.9 | 32 | 30/32 (94%) | 80% to 98% | roll 2 |
| 0.95 | 32 | 29/32 (91%) | 76% to 97% | roll 2, switch 1 |
| 1 | 32 | 28/32 (88%) | 72% to 95% | legs_together 1, roll 2, switch 1 |
| 1.05 | 32 | 27/32 (84%) | 68% to 93% | legs_together 2, roll 3 |
| 1.1 | 32 | 29/32 (91%) | 76% to 97% | roll 1, switch 2 |
| 1.2 | 32 | 24/32 (75%) | 58% to 87% | legs_together 3, roll 4, splitover 1 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 30/32 | 30 | switch 1, legs_together 1 |
| 101 | 30/32 | 30 | roll 2 |
| 102 | 29/32 | 29 | roll 2, switch 1 |
| 103 | 28/32 | 28 | roll 2, legs_together 1, switch 1 |
| 104 | 27/32 | 27 | roll 3, legs_together 2 |
| 105 | 29/32 | 29 | switch 2, roll 1 |
| 106 | 24/32 | 24 | legs_together 3, roll 4, splitover 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 4022 s of simulation.
