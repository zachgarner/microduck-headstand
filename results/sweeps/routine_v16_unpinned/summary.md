# routine_v16_unpinned

Issue #12 acceptance: the v16 routine with nothing pinned, 32 attempts at each of seeds 0, 1 and 2, to reproduce the verified 87/96.

96 attempts, 87 successes. Mode: routine. Seed 0, 96 attempts per grid point.

## By seed

| Seed | Success | Final standing | First unfinished stage of failures |
| --- | ---: | ---: | --- |
| 0 | 27/32 | 27 | roll 3, legs_together 1, switch 1 |
| 1 | 31/32 | 31 | legs_together 1 |
| 2 | 29/32 | 29 | legs_together 1, roll 2 |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 494 s of simulation.
