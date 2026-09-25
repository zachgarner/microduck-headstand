# baseline_head_com_z

Issue #13: the v16 routine with head_com_z pinned per environment, 32 attempts per value, every other parameter on its training draw.

224 attempts, 190 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## head_com_z



Unit: m. Nominal 0. Training range -0.01 to 0.01.

| head_com_z | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| -0.02 | 32 | 21/32 (66%) | 48% to 80% | legs_together 10, split 1 |
| -0.01 | 32 | 30/32 (94%) | 80% to 98% | legs_together 2 |
| -0.005 | 32 | 28/32 (88%) | 72% to 95% | legs_together 2, switch 2 |
| 0 | 32 | 29/32 (91%) | 76% to 97% | roll 2, switch 1 |
| 0.005 | 32 | 30/32 (94%) | 80% to 98% | roll 2 |
| 0.01 | 32 | 29/32 (91%) | 76% to 97% | roll 2, switch 1 |
| 0.02 | 32 | 23/32 (72%) | 55% to 84% | roll 6, switch 3 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 21/32 | 21 | legs_together 10, split 1 |
| 101 | 30/32 | 30 | legs_together 2 |
| 102 | 28/32 | 28 | legs_together 2, switch 2 |
| 103 | 29/32 | 29 | roll 2, switch 1 |
| 104 | 30/32 | 30 | roll 2 |
| 105 | 29/32 | 29 | switch 1, roll 2 |
| 106 | 23/32 | 23 | roll 6, switch 3 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 4236 s of simulation.
