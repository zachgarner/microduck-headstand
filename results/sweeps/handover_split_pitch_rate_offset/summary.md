# handover_split_pitch_rate_offset

Issue #15: the split stage started from the states the routine handed it in routine_v16_unpinned, with handover_pitch_rate_offset applied, 32 attempts per value. Each repeat uses the same source state at every value.

160 attempts, 125 successes. Mode: handover. Seed 300, 32 attempts per grid point.

## handover_pitch_rate_offset

Add `value` rad/s to the trunk's pitch rate. MuJoCo's free-joint angular     velocity is in the body frame, so index 4 is the pitch axis.

Unit: rad/s. Nominal 0. Training range 0 to 0.

| handover_pitch_rate_offset | Attempts | Success | 95% interval |
| --- | ---: | ---: | --- |
| -3 | 32 | 31/32 (97%) | 84% to 99% |
| -1.5 | 32 | 32/32 (100%) | 89% to 100% |
| 0 | 32 | 23/32 (72%) | 55% to 84% |
| 1.5 | 32 | 17/32 (53%) | 36% to 69% |
| 3 | 32 | 22/32 (69%) | 51% to 82% |

## Checks

The swept axes perturb start states, so there is no readback to check.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 1551 s of simulation.
