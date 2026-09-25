# baseline_trunk_com_x

Issue #13: the v16 routine with trunk_com_x pinned per environment, 32 attempts per value, every other parameter on its training draw.

288 attempts, 186 successes. Mode: routine. Seed 100, 32 attempts per grid point.

## trunk_com_x



Unit: m. Nominal 0. Training range -0.015 to 0.015.

| trunk_com_x | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| -0.03 | 32 | 6/32 (19%) | 9% to 35% | fold 5, legs_together 3, roll 6, settle 3, fold2 1, switch 4, splitover 2, settle2 2 |
| -0.02 | 32 | 24/32 (75%) | 58% to 87% | fold 1, roll 2, split 1, switch 2, splitover 1, settle2 1 |
| -0.015 | 32 | 24/32 (75%) | 58% to 87% | roll 6, switch 2 |
| -0.008 | 32 | 26/32 (81%) | 65% to 91% | legs_together 1, roll 3, switch 2 |
| 0 | 32 | 26/32 (81%) | 65% to 91% | legs_together 2, roll 4 |
| 0.008 | 32 | 29/32 (91%) | 76% to 97% | legs_together 1, roll 1, switch 1 |
| 0.015 | 32 | 20/32 (62%) | 45% to 77% | legs_together 8, roll 2, splitover 2 |
| 0.02 | 32 | 21/32 (66%) | 48% to 80% | legs_together 7, roll 1, fold2 1, splitover 2 |
| 0.03 | 32 | 10/32 (31%) | 18% to 49% | legs_together 12, roll 4, splitover 5, settle2 1 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 100 | 6/32 | 8 | roll 6, settle 3, fold 5, fold2 1, switch 4, legs_together 3, splitover 2, settle2 2 |
| 101 | 24/32 | 25 | split 1, fold 1, switch 2, roll 2, settle2 1, splitover 1 |
| 102 | 24/32 | 24 | roll 6, switch 2 |
| 103 | 26/32 | 26 | roll 3, switch 2, legs_together 1 |
| 104 | 26/32 | 26 | roll 4, legs_together 2 |
| 105 | 29/32 | 29 | switch 1, legs_together 1, roll 1 |
| 106 | 20/32 | 20 | roll 2, legs_together 8, splitover 2 |
| 107 | 21/32 | 21 | roll 1, legs_together 7, splitover 2, fold2 1 |
| 108 | 10/32 | 11 | splitover 5, legs_together 12, roll 4, settle2 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5364 s of simulation.
