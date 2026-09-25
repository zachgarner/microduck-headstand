# baseline_encoder_bias

Issue #13: the v16 routine with encoder_bias pinned per environment, 32 attempts per value, every other parameter on its training draw.

192 attempts, 170 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## encoder_bias

Each joint's bias is drawn from plus or minus this value. Readback is the largest joint bias.

Unit: rad. Nominal 0. Training range 0 to 0.015.

| encoder_bias | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0 | 32 | 30/32 (94%) | 80% to 98% | legs_together 1, roll 1 |
| 0.01 | 32 | 30/32 (94%) | 80% to 98% | roll 2 |
| 0.015 | 32 | 25/32 (78%) | 61% to 89% | legs_together 1, roll 3, switch 2, splitover 1 |
| 0.02 | 32 | 32/32 (100%) | 89% to 100% | none |
| 0.03 | 32 | 26/32 (81%) | 65% to 91% | legs_together 2, roll 4 |
| 0.05 | 32 | 27/32 (84%) | 68% to 93% | legs_together 1, roll 2, switch 1, fell after finishing 1 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 30/32 | 30 | roll 1, legs_together 1 |
| 101 | 30/32 | 30 | roll 2 |
| 102 | 25/32 | 25 | roll 3, switch 2, legs_together 1, splitover 1 |
| 103 | 32/32 | 32 | none |
| 104 | 26/32 | 26 | roll 4, legs_together 2 |
| 105 | 27/32 | 27 | switch 1, roll 2, fell after finishing 1, legs_together 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 4435 s of simulation.
