# baseline_trunk_com_z

Issue #13: the v16 routine with trunk_com_z pinned per environment, 32 attempts per value, every other parameter on its training draw.

288 attempts, 210 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## trunk_com_z



Unit: m. Nominal 0. Training range -0.015 to 0.015.

| trunk_com_z | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| -0.03 | 32 | 12/32 (38%) | 23% to 55% | legs_together 18, split 1, switch 1 |
| -0.02 | 32 | 25/32 (78%) | 61% to 89% | legs_together 6, roll 1 |
| -0.015 | 32 | 18/32 (56%) | 39% to 72% | legs_together 12, roll 1, switch 1 |
| -0.008 | 32 | 25/32 (78%) | 61% to 89% | legs_together 2, roll 2, switch 2, splitover 1 |
| 0 | 32 | 26/32 (81%) | 65% to 91% | legs_together 2, roll 4 |
| 0.008 | 32 | 30/32 (94%) | 80% to 98% | roll 1, switch 1 |
| 0.015 | 32 | 26/32 (81%) | 65% to 91% | roll 5, splitover 1 |
| 0.02 | 32 | 26/32 (81%) | 65% to 91% | roll 4, splitover 2 |
| 0.03 | 32 | 22/32 (69%) | 51% to 82% | roll 8, splitover 2 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 12/32 | 12 | legs_together 18, split 1, switch 1 |
| 101 | 25/32 | 25 | legs_together 6, roll 1 |
| 102 | 18/32 | 18 | legs_together 12, roll 1, switch 1 |
| 103 | 25/32 | 25 | switch 2, legs_together 2, roll 2, splitover 1 |
| 104 | 26/32 | 26 | roll 4, legs_together 2 |
| 105 | 30/32 | 30 | switch 1, roll 1 |
| 106 | 26/32 | 26 | roll 5, splitover 1 |
| 107 | 26/32 | 26 | roll 4, splitover 2 |
| 108 | 22/32 | 22 | roll 8, splitover 2 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5166 s of simulation.
