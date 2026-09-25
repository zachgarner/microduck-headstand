# baseline_imu_misalignment

Issue #13: the v16 routine with imu_misalignment pinned per environment, 32 attempts per value, every other parameter on its training draw.

192 attempts, 100 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## imu_misalignment

A fixed mounting error of this angle about a random axis.

Unit: deg. Nominal 0. Training range 0 to 6.

| imu_misalignment | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0 | 32 | 30/32 (94%) | 80% to 98% | roll 2 |
| 3 | 32 | 28/32 (88%) | 72% to 95% | roll 3, switch 1 |
| 6 | 32 | 24/32 (75%) | 58% to 87% | legs_together 2, roll 3, switch 2, splitover 1 |
| 9 | 32 | 13/32 (41%) | 26% to 58% | legs_together 5, roll 1, split 2, switch 10, splitover 1 |
| 12 | 32 | 5/32 (16%) | 7% to 32% | legs_together 18, roll 3, settle 1, split 2, switch 1, splitover 2 |
| 18 | 32 | 0/32 (0%) | 0% to 11% | fold 1, legs_together 25, settle 1, split 5 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 30/32 | 30 | roll 2 |
| 101 | 28/32 | 28 | switch 1, roll 3 |
| 102 | 24/32 | 24 | legs_together 2, splitover 1, switch 2, roll 3 |
| 103 | 13/32 | 13 | switch 10, legs_together 5, splitover 1, split 2, roll 1 |
| 104 | 5/32 | 5 | legs_together 18, split 2, roll 3, splitover 2, switch 1, settle 1 |
| 105 | 0/32 | 0 | split 5, legs_together 25, fold 1, settle 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 3508 s of simulation.
