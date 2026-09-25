# baseline_contact_friction_scale

Issue #13: the v16 routine with contact_friction_scale pinned per environment, 32 attempts per value, every other parameter on its training draw.

192 attempts, 145 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## contact_friction_scale

Training never randomizes contact friction.

Unit: x nominal. Nominal 1. Training range 1 to 1.

| contact_friction_scale | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0.3 | 32 | 11/32 (34%) | 20% to 52% | legs_together 3, roll 14, switch 1, splitover 3 |
| 0.5 | 32 | 26/32 (81%) | 65% to 91% | roll 5, splitover 1 |
| 0.7 | 32 | 25/32 (78%) | 61% to 89% | roll 5, splitover 2 |
| 1 | 32 | 30/32 (94%) | 80% to 98% | legs_together 1, roll 1 |
| 1.3 | 32 | 26/32 (81%) | 65% to 91% | legs_together 1, roll 4, switch 1 |
| 1.6 | 32 | 27/32 (84%) | 68% to 93% | roll 1, switch 4 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 11/32 | 11 | roll 14, legs_together 3, splitover 3, switch 1 |
| 101 | 26/32 | 26 | roll 5, splitover 1 |
| 102 | 25/32 | 25 | roll 5, splitover 2 |
| 103 | 30/32 | 30 | legs_together 1, roll 1 |
| 104 | 26/32 | 26 | roll 4, legs_together 1, switch 1 |
| 105 | 27/32 | 27 | switch 4, roll 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 4311 s of simulation.
