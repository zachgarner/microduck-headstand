# handover_split_joint_noise

Issue #15: the split stage started from the states the routine handed it in routine_v16_unpinned, with handover_joint_noise applied, 32 attempts per value. Each repeat uses the same source state at every value.

128 attempts, 90 successes. Mode: handover. Seed 300, 32 attempts per grid point.

## handover_joint_noise

Add Gaussian noise with standard deviation `value` rad to every servo     position. Plain-model layout: the 14 servos follow the 7 base values.

Unit: rad. Nominal 0. Training range 0 to 0.

| handover_joint_noise | Attempts | Success | 95% interval |
| --- | ---: | ---: | --- |
| 0 | 32 | 21/32 (66%) | 48% to 80% |
| 0.05 | 32 | 26/32 (81%) | 65% to 91% |
| 0.1 | 32 | 24/32 (75%) | 58% to 87% |
| 0.2 | 32 | 19/32 (59%) | 42% to 74% |

## Checks

The swept axes perturb start states, so there is no readback to check.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 1271 s of simulation.
