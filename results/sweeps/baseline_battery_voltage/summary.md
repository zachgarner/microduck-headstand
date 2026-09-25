# baseline_battery_voltage

Issue #13: the v16 routine with battery_voltage pinned per environment, 32 attempts per value, every other parameter on its training draw.

224 attempts, 194 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## battery_voltage

Above 8.2 V the MuJoCo force range, set from the top of the training range, clips the torque.

Unit: V. Nominal 7.4. Training range 6.5 to 8.2.

| battery_voltage | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 5.5 | 32 | 25/32 (78%) | 61% to 89% | legs_together 1, roll 4, switch 2 |
| 6 | 32 | 29/32 (91%) | 76% to 97% | roll 3 |
| 6.5 | 32 | 27/32 (84%) | 68% to 93% | legs_together 1, roll 3, switch 1 |
| 7 | 32 | 28/32 (88%) | 72% to 95% | legs_together 1, roll 1, switch 2 |
| 7.4 | 32 | 29/32 (91%) | 76% to 97% | roll 3 |
| 7.8 | 32 | 27/32 (84%) | 68% to 93% | legs_together 2, roll 2, switch 1 |
| 8.2 | 32 | 29/32 (91%) | 76% to 97% | legs_together 1, roll 2 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 25/32 | 25 | roll 4, switch 2, legs_together 1 |
| 101 | 29/32 | 29 | roll 3 |
| 102 | 27/32 | 27 | roll 3, switch 1, legs_together 1 |
| 103 | 28/32 | 28 | switch 2, legs_together 1, roll 1 |
| 104 | 29/32 | 29 | roll 3 |
| 105 | 27/32 | 27 | switch 1, legs_together 2, roll 2 |
| 106 | 29/32 | 29 | roll 2, legs_together 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5115 s of simulation.
