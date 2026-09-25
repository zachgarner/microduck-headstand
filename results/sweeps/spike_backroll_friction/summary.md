# spike_backroll_friction

Issue #11: joint friction pinned per environment for the back roll, in one 256-environment batch.

256 attempts, 251 successes. Mode: policy. Seed 0, 16 attempts per grid point.

## friction_scale

Scales the BAM actuator's friction budget: Coulomb, Stribeck and load-dependent.

Unit: x nominal. Nominal 1. Training range 0.9 to 1.1.

| friction_scale | Attempts | Success | 95% interval |
| --- | ---: | ---: | --- |
| 0.5 | 16 | 16/16 (100%) | 81% to 100% |
| 0.6 | 16 | 16/16 (100%) | 81% to 100% |
| 0.7 | 16 | 16/16 (100%) | 81% to 100% |
| 0.8 | 16 | 16/16 (100%) | 81% to 100% |
| 0.9 | 16 | 16/16 (100%) | 81% to 100% |
| 1 | 16 | 16/16 (100%) | 81% to 100% |
| 1.1 | 16 | 16/16 (100%) | 81% to 100% |
| 1.2 | 16 | 16/16 (100%) | 81% to 100% |
| 1.3 | 16 | 16/16 (100%) | 81% to 100% |
| 1.4 | 16 | 16/16 (100%) | 81% to 100% |
| 1.5 | 16 | 16/16 (100%) | 81% to 100% |
| 1.6 | 16 | 16/16 (100%) | 81% to 100% |
| 1.7 | 16 | 16/16 (100%) | 81% to 100% |
| 1.8 | 16 | 16/16 (100%) | 81% to 100% |
| 1.9 | 16 | 15/16 (94%) | 72% to 99% |
| 2 | 16 | 12/16 (75%) | 51% to 90% |

## Checks

Pinned values held for the whole rollout in every attempt: yes.
Resets inside a rollout: 0.

Chart: [map.png](map.png). Records: `records.jsonl`. States: `states.npz`. Provenance: `provenance.json`, 453 s of simulation.
