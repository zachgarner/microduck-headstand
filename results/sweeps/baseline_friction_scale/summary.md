# baseline_friction_scale

Issue #13: the v16 routine with friction_scale pinned per environment, 32 attempts per value, every other parameter on its training draw.

256 attempts, 197 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## friction_scale

Scales the BAM actuator's friction budget: Coulomb, Stribeck and load-dependent.

Unit: x nominal. Nominal 1. Training range 0.9 to 1.1.

| friction_scale | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0.5 | 32 | 29/32 (91%) | 76% to 97% | legs_together 1, roll 2 |
| 0.7 | 32 | 32/32 (100%) | 89% to 100% | none |
| 0.9 | 32 | 29/32 (91%) | 76% to 97% | legs_together 1, roll 2 |
| 1 | 32 | 27/32 (84%) | 68% to 93% | legs_together 1, roll 3, splitover 1 |
| 1.1 | 32 | 28/32 (88%) | 72% to 95% | legs_together 1, roll 3 |
| 1.3 | 32 | 28/32 (88%) | 72% to 95% | roll 3, switch 1 |
| 1.6 | 32 | 18/32 (56%) | 39% to 72% | roll 12, switch 1, splitover 1 |
| 2 | 32 | 6/32 (19%) | 9% to 35% | fold 1, legs_together 1, roll 15, settle 1, switch 1, splitover 7 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 29/32 | 29 | roll 2, legs_together 1 |
| 101 | 32/32 | 32 | none |
| 102 | 29/32 | 29 | roll 2, legs_together 1 |
| 103 | 27/32 | 27 | roll 3, legs_together 1, splitover 1 |
| 104 | 28/32 | 28 | roll 3, legs_together 1 |
| 105 | 28/32 | 28 | switch 1, roll 3 |
| 106 | 18/32 | 18 | roll 12, splitover 1, switch 1 |
| 107 | 6/32 | 6 | roll 15, settle 1, splitover 7, switch 1, legs_together 1, fold 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5855 s of simulation.
