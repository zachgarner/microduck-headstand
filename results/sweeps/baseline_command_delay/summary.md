# baseline_command_delay

Issue #13: the v16 routine with command_delay pinned per environment, 32 attempts per value, every other parameter on its training draw.

288 attempts, 138 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## command_delay

Training resamples the lag every physics step. A pinned lag is constant.

Unit: physics steps (5 ms). Nominal 4.5. Training range 3 to 6.

| command_delay | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0 | 32 | 0/32 (0%) | 0% to 11% | fold 2, legs_together 21, roll 6, switch 1, splitover 2 |
| 2 | 32 | 17/32 (53%) | 36% to 69% | legs_together 6, roll 8, splitover 1 |
| 3 | 32 | 22/32 (69%) | 51% to 82% | legs_together 3, roll 4, switch 2, splitover 1 |
| 4 | 32 | 30/32 (94%) | 80% to 98% | legs_together 2 |
| 5 | 32 | 29/32 (91%) | 76% to 97% | roll 3 |
| 6 | 32 | 28/32 (88%) | 72% to 95% | roll 1, switch 3 |
| 8 | 32 | 12/32 (38%) | 23% to 55% | roll 17, splitover 3 |
| 10 | 32 | 0/32 (0%) | 0% to 11% | legs_together 2, roll 25, split 2, splitover 3 |
| 12 | 32 | 0/32 (0%) | 0% to 11% | legs_together 6, roll 24, split 1, splitover 1 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 0/32 | 0 | legs_together 21, roll 6, splitover 2, fold 2, switch 1 |
| 101 | 17/32 | 17 | legs_together 6, roll 8, splitover 1 |
| 102 | 22/32 | 22 | legs_together 3, roll 4, switch 2, splitover 1 |
| 103 | 30/32 | 30 | legs_together 2 |
| 104 | 29/32 | 29 | roll 3 |
| 105 | 28/32 | 28 | switch 3, roll 1 |
| 106 | 12/32 | 12 | splitover 3, roll 17 |
| 107 | 0/32 | 0 | roll 25, split 2, splitover 3, legs_together 2 |
| 108 | 0/32 | 0 | split 1, legs_together 6, roll 24, splitover 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 6401 s of simulation.
