# baseline_trunk_com_y

Issue #13: the v16 routine with trunk_com_y pinned per environment, 32 attempts per value, every other parameter on its training draw.

288 attempts, 135 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## trunk_com_y



Unit: m. Nominal 0. Training range -0.015 to 0.015.

| trunk_com_y | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| -0.03 | 32 | 1/32 (3%) | 1% to 16% | legs_together 19, split 4, switch 8 |
| -0.02 | 32 | 3/32 (9%) | 3% to 24% | legs_together 1, split 1, switch 27 |
| -0.015 | 32 | 9/32 (28%) | 16% to 45% | legs_together 1, roll 3, switch 19 |
| -0.008 | 32 | 22/32 (69%) | 51% to 82% | legs_together 2, roll 2, switch 6 |
| 0 | 32 | 26/32 (81%) | 65% to 91% | legs_together 2, roll 4 |
| 0.008 | 32 | 30/32 (94%) | 80% to 98% | legs_together 1, roll 1 |
| 0.015 | 32 | 24/32 (75%) | 58% to 87% | roll 6, split 2 |
| 0.02 | 32 | 20/32 (62%) | 45% to 77% | legs_together 1, roll 3, split 1, switch 5, splitover 1, settle2 1 |
| 0.03 | 32 | 0/32 (0%) | 0% to 11% | legs_together 17, roll 4, split 5, switch 6 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 1/32 | 1 | legs_together 19, switch 8, split 4 |
| 101 | 3/32 | 3 | switch 27, split 1, legs_together 1 |
| 102 | 9/32 | 9 | switch 19, roll 3, legs_together 1 |
| 103 | 22/32 | 22 | switch 6, roll 2, legs_together 2 |
| 104 | 26/32 | 26 | roll 4, legs_together 2 |
| 105 | 30/32 | 30 | roll 1, legs_together 1 |
| 106 | 24/32 | 24 | split 2, roll 6 |
| 107 | 20/32 | 22 | switch 5, settle2 1, roll 3, split 1, splitover 1, legs_together 1 |
| 108 | 0/32 | 0 | roll 4, legs_together 17, switch 6, split 5 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5144 s of simulation.
