# handover_switch_pitch_offset

Issue #15: the switch stage started from the states the routine handed it in routine_v16_unpinned, with handover_pitch_offset applied, 32 attempts per value. Each repeat uses the same source state at every value.

224 attempts, 224 successes. Mode: handover. Seed 300, 32 attempts per grid point.

## handover_pitch_offset

Rotate the trunk by `value` degrees about its own pitch axis.

Unit: deg. Nominal 0. Training range 0 to 0.

| handover_pitch_offset | Attempts | Success | 95% interval |
| --- | ---: | ---: | --- |
| -20 | 32 | 32/32 (100%) | 89% to 100% |
| -10 | 32 | 32/32 (100%) | 89% to 100% |
| -5 | 32 | 32/32 (100%) | 89% to 100% |
| 0 | 32 | 32/32 (100%) | 89% to 100% |
| 5 | 32 | 32/32 (100%) | 89% to 100% |
| 10 | 32 | 32/32 (100%) | 89% to 100% |
| 20 | 32 | 32/32 (100%) | 89% to 100% |

## Checks

The swept axes perturb start states, so there is no readback to check.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 1934 s of simulation.
