# baseline_voltage_drop_gain

Issue #13: the v16 routine with voltage_drop_gain pinned per environment, 32 attempts per value, every other parameter on its training draw.

192 attempts, 162 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## voltage_drop_gain

Voltage sag per newton-metre of total servo torque.

Unit: V/Nm. Nominal 0.1. Training range 0 to 0.2.

| voltage_drop_gain | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0 | 32 | 27/32 (84%) | 68% to 93% | legs_together 2, roll 1, switch 1, settle2 1 |
| 0.1 | 32 | 28/32 (88%) | 72% to 95% | roll 3, switch 1 |
| 0.2 | 32 | 28/32 (88%) | 72% to 95% | legs_together 1, roll 2, switch 1 |
| 0.3 | 32 | 29/32 (91%) | 76% to 97% | legs_together 1, roll 2 |
| 0.4 | 32 | 25/32 (78%) | 61% to 89% | legs_together 1, roll 6 |
| 0.6 | 32 | 25/32 (78%) | 61% to 89% | legs_together 2, roll 1, switch 3, splitover 1 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 27/32 | 28 | roll 1, settle2 1, legs_together 2, switch 1 |
| 101 | 28/32 | 28 | switch 1, roll 3 |
| 102 | 28/32 | 28 | roll 2, switch 1, legs_together 1 |
| 103 | 29/32 | 29 | roll 2, legs_together 1 |
| 104 | 25/32 | 25 | roll 6, legs_together 1 |
| 105 | 25/32 | 25 | switch 3, splitover 1, legs_together 2, roll 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 3446 s of simulation.
