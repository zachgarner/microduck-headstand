# handover_roll_pitch_offset

Issue #15: the roll stage started from the states the routine handed it in routine_v16_unpinned, with handover_pitch_offset applied, 32 attempts per value. Each repeat uses the same source state at every value.

224 attempts, 193 successes. Mode: handover. Seed 300, 32 attempts per grid point.

## handover_pitch_offset

Rotate the trunk by `value` degrees about its own pitch axis.

Unit: deg. Nominal 0. Training range 0 to 0.

| handover_pitch_offset | Attempts | Success | 95% interval |
| --- | ---: | ---: | --- |
| -20 | 32 | 24/32 (75%) | 58% to 87% |
| -10 | 32 | 31/32 (97%) | 84% to 99% |
| -5 | 32 | 29/32 (91%) | 76% to 97% |
| 0 | 32 | 29/32 (91%) | 76% to 97% |
| 5 | 32 | 28/32 (88%) | 72% to 95% |
| 10 | 32 | 27/32 (84%) | 68% to 93% |
| 20 | 32 | 25/32 (78%) | 61% to 89% |

## Checks

The swept axes perturb start states, so there is no readback to check.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 2039 s of simulation.
