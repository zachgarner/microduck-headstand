# baseline_armature_scale

Issue #13: the v16 routine with armature_scale pinned per environment, 32 attempts per value, every other parameter on its training draw.

224 attempts, 188 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## armature_scale



Unit: x nominal. Nominal 1. Training range 0.9 to 1.1.

| armature_scale | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0.5 | 32 | 25/32 (78%) | 61% to 89% | legs_together 1, roll 5, splitover 1 |
| 0.7 | 32 | 28/32 (88%) | 72% to 95% | roll 4 |
| 0.9 | 32 | 24/32 (75%) | 58% to 87% | roll 7, switch 1 |
| 1 | 32 | 32/32 (100%) | 89% to 100% | none |
| 1.1 | 32 | 28/32 (88%) | 72% to 95% | legs_together 2, roll 2 |
| 1.3 | 32 | 29/32 (91%) | 76% to 97% | legs_together 1, roll 1, switch 1 |
| 1.6 | 32 | 22/32 (69%) | 51% to 82% | legs_together 3, roll 3, split 1, splitover 3 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 25/32 | 25 | splitover 1, roll 5, legs_together 1 |
| 101 | 28/32 | 28 | roll 4 |
| 102 | 24/32 | 24 | roll 7, switch 1 |
| 103 | 32/32 | 32 | none |
| 104 | 28/32 | 28 | roll 2, legs_together 2 |
| 105 | 29/32 | 29 | switch 1, roll 1, legs_together 1 |
| 106 | 22/32 | 22 | legs_together 3, roll 3, split 1, splitover 3 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5157 s of simulation.
