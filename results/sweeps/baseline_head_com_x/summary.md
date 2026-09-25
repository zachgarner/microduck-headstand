# baseline_head_com_x

Issue #13: the v16 routine with head_com_x pinned per environment, 32 attempts per value, every other parameter on its training draw.

224 attempts, 197 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## head_com_x



Unit: m. Nominal 0. Training range -0.01 to 0.01.

| head_com_x | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| -0.02 | 32 | 25/32 (78%) | 61% to 89% | legs_together 2, roll 1, switch 4 |
| -0.01 | 32 | 32/32 (100%) | 89% to 100% | none |
| -0.005 | 32 | 27/32 (84%) | 68% to 93% | legs_together 1, roll 1, switch 3 |
| 0 | 32 | 29/32 (91%) | 76% to 97% | roll 2, switch 1 |
| 0.005 | 32 | 28/32 (88%) | 72% to 95% | legs_together 1, roll 3 |
| 0.01 | 32 | 29/32 (91%) | 76% to 97% | legs_together 1, roll 2 |
| 0.02 | 32 | 27/32 (84%) | 68% to 93% | legs_together 1, roll 2, splitover 2 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 25/32 | 25 | roll 1, switch 4, legs_together 2 |
| 101 | 32/32 | 32 | none |
| 102 | 27/32 | 27 | switch 3, roll 1, legs_together 1 |
| 103 | 29/32 | 29 | roll 2, switch 1 |
| 104 | 28/32 | 28 | roll 3, legs_together 1 |
| 105 | 29/32 | 29 | roll 2, legs_together 1 |
| 106 | 27/32 | 27 | splitover 2, roll 2, legs_together 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5092 s of simulation.
