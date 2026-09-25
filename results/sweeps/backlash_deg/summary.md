# backlash_deg

Issue #14: the v16 routine on the all-collisions backlash model, total gear play pinned per environment, 32 attempts per value. The policies trained without backlash; the -Backlash- tasks train with 2 degrees.

224 attempts, 198 successes. Mode: routine. Seed 200, 32 attempts per grid point.

## backlash_deg

Needs the backlash robot model. The headstand policies trained without backlash; the -Backlash- tasks use 2 degrees.

Unit: deg total. Nominal 0. Training range 0 to 0.

| backlash_deg | Attempts | Success | 95% interval | First unfinished stage |
| --- | ---: | ---: | --- | --- |
| 0 | 32 | 27/32 (84%) | 68% to 93% | roll 5 |
| 0.5 | 32 | 28/32 (88%) | 72% to 95% | roll 1, switch 2, settle2 1 |
| 1 | 32 | 32/32 (100%) | 89% to 100% | none |
| 1.5 | 32 | 31/32 (97%) | 84% to 99% | switch 1 |
| 2 | 32 | 26/32 (81%) | 65% to 91% | legs_together 2, roll 2, splitover 2 |
| 3 | 32 | 28/32 (88%) | 72% to 95% | roll 2, switch 2 |
| 4 | 32 | 26/32 (81%) | 65% to 91% | legs_together 1, roll 4, switch 1 |

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 200 | 27/32 | 27 | roll 5 |
| 201 | 28/32 | 29 | switch 2, roll 1, settle2 1 |
| 202 | 32/32 | 32 | none |
| 203 | 31/32 | 31 | switch 1 |
| 204 | 26/32 | 26 | splitover 2, legs_together 2, roll 2 |
| 205 | 28/32 | 28 | switch 2, roll 2 |
| 206 | 26/32 | 26 | roll 4, switch 1, legs_together 1 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 5484 s of simulation.
