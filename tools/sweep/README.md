# Parameter sweeps

The sweep harness pins a physics parameter to a chosen value in each simulated environment, runs the headstand routine or one of its policies, and maps where each stage fails. Every other parameter keeps its random training draw. The work is tracked in [zachgarner/microduck_rl#10](https://github.com/zachgarner/microduck_rl/issues/10).

## Run a sweep

On the Anyscale workspace, spread the batches over the Ray cluster:

```bash
cd microduck_rl
python ../tools/sweep/ray_sweep.py ../tools/sweep/configs/baseline/friction_scale.json
```

`ray_sweep.py` runs with the cluster's Python, which has Ray. Each batch becomes a one-CPU Ray task. The first task on a node builds the training environment with uv and downloads the evaluated checkpoints, and later tasks on that node reuse them. Several configs can go in one call, and the autoscaler adds CPU nodes as tasks queue. Each sweep merges into `results/sweeps/<name>/` when its last batch returns.

For a small sweep, run it directly in the training environment. `--workers` runs batches as parallel processes:

```bash
cd microduck_rl
uv run ../tools/sweep/run_sweep.py ../tools/sweep/configs/spike_backroll_friction.json --workers 4
```

Then draw the map:

```bash
uv run --with matplotlib ../tools/sweep/report.py ../results/sweeps/<name>
```

Everything steps on Warp's CPU backend, the backend the verified evaluators use, so sweep results stay comparable with `results/verified/`. Warp steps a batch on one core. On CPU, throughput comes from running many batches at once, not from larger batches.

## The config

A config is a JSON file:

| Key | Meaning |
| --- | --- |
| `name` | The output directory under `results/sweeps/`. |
| `mode` | `policy` runs one policy from one start. `routine` runs the full v16 routine from standing. `handover` runs one routine stage from the states a routine sweep recorded when it handed over to that stage. |
| `robot` | `plain` (the default) or `backlash`, the all-collisions model with a passive backlash hinge after every servo. |
| `axes` | Each axis name and its grid values. The grid is the product of all axes' values. |
| `attempts` | Environments per grid point. |
| `seed`, `max_envs` | Environments split into batches of at most `max_envs`. Batch b runs at seed `seed + b`. |

`policy` mode also takes `policy.task`, `policy.checkpoint` (`run:model_N.pt`), `policy.start` (a spawn bucket) and `policy.steps`. `routine` and `handover` modes take the `routine` block of checkpoints and hold times from `configs/routine_v16_baseline_unpinned.json`. `handover` mode also takes `source` (a routine sweep's directory), `stage` and `stage_seconds`.

## Axes

Physics axes are defined in `axes.py`. Each one pins its value after the reset, through the same function or tensor the task's randomization uses, and restores the nominal value first so nothing accumulates.

| Axis | Unit | Training range |
| --- | --- | --- |
| `friction_scale` | × nominal | 0.9 to 1.1 |
| `battery_voltage` | V | 6.5 to 8.2 |
| `voltage_drop_gain` | V/Nm | 0 to 0.2 |
| `command_delay` | physics steps of 5 ms | 3 to 6, resampled every step |
| `trunk_com_x`, `_y`, `_z` | m | ±0.015 at the curriculum's end |
| `head_com_x`, `_z` | m | ±0.010 at the curriculum's end |
| `trunk_mass_scale` | × nominal | 0.95 to 1.05 |
| `armature_scale` | × nominal | 0.9 to 1.1 |
| `encoder_bias` | rad, each joint drawn from ±value | ±0.015 |
| `imu_misalignment` | deg about a random axis | 0 to 6 |
| `contact_friction_scale` | × nominal, every geom | not randomized |
| `backlash_deg` | deg total play, with `"robot": "backlash"` | not trained |

The model sets the MuJoCo force range from the top of the voltage training range. Above 8.2 V, that range clips the servo torque.

State axes are defined in `rollout.py` and apply in `handover` mode. They perturb each recorded handover state before the next policy starts from it:

| Axis | Unit |
| --- | --- |
| `handover_pitch_offset` | deg about the trunk's pitch axis |
| `handover_pitch_rate_offset` | rad/s added to the trunk's pitch rate |
| `handover_joint_noise` | rad, the standard deviation of Gaussian noise on every servo |

Each repeat uses the same source state at every value, so values compare on paired states.

`check_axes.py` verifies every physics axis. It pins a low and a high value at the same seed and checks three things: the readback matches, two runs at the same value are identical, and the two values produce different motion.

## Output

| File | Contents |
| --- | --- |
| `records.jsonl` | One line per attempt: grid point, pinned values, readback of every axis at the start and the end, outcome, and stage times. |
| `states.npz` | The state at every handover and at the end, in the handover-set layout (qpos and qvel), indexed by `attempt`. |
| `provenance.json` | The config, checkpoint and tool hashes, code revisions, package versions, and simulation and wall time. |
| `map.png`, `summary.md` | The report: success rate with its 95% Wilson interval per value, the training range shaded, and for routine sweeps the first unfinished stage of each failure. |

## Tests

```bash
uv run --with pytest --with matplotlib pytest ../tools/test_sweep.py
uv run ../tools/sweep/check_axes.py
```

The first is quick and needs no simulation. The second builds 45 small environments and belongs on the workspace.
