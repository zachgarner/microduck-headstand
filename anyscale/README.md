# Training on Anyscale

Each run is one L40S GPU node on Anyscale, submitted with `submit.py` from this repo's root. `train.sh` runs on the node: it restores the compiled MuJoCo Warp kernels from the artifact bucket, trains, and syncs the checkpoints back every five minutes. The Dockerfile builds the image with the fork's environment already installed, so a job starts training within a minute.

## Runs of record

Every policy in the routine, the run that produced it, and what it started from. The knobs column records environment variables used by the historical training code. The current factories take explicit keyword arguments; those environment variables no longer configure a run. All runs: 4,096 envs, one L40S, checkpoints every 250 iterations in the wandb project `zachgarner-ai/mjlab_microduck`.

| Policy | Task | Started from | Knobs | Iterations | wandb run | In the routine |
| --- | --- | --- | --- | --- | --- | --- |
| Fold, standing to the pike | `Mjlab-HeadstandFold-Flat-MicroDuck` | scratch | none | 1500 | `vorty4kb` | checkpoint 1000 |
| Split kick-up, snap (the root of the kick-up family) | `Mjlab-HeadstandKickup-Flat-MicroDuck` | scratch, pike spawns | none | 1000 | `076n5wpa` | no, the ancestor |
| Straight kick-up, snap | `Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck` | `076n5wpa` 999 | none | 1500 | `geexqesa` | no, the ancestor |
| Slow split kick-up | `Mjlab-HeadstandKickup-Flat-MicroDuck` | `076n5wpa` 999 | `HEADSTAND_RAMP_S=2.0 HEADSTAND_OMEGA_MAX=1.5 HEADSTAND_OVERSPEED_W=-1.0 HEADSTAND_POLISH_AT=0` | 1500 | `y21jqomx` | no, the ancestor |
| Slow straight kick-up | `Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck` | the snap | rotation cap 3 rad/s | 1500 | `exlaizfb` | no, the ancestor |
| Straight kick-up from the fold's handovers | `Mjlab-HeadstandKickupLegsTogether-Flat-MicroDuck` | `exlaizfb` 1000 | `HEADSTAND_OMEGA_MAX=2.0 HEADSTAND_OVERSPEED_W=-0.5 HEADSTAND_BANK_PROB=0.5 HEADSTAND_POLISH_AT=0` | 1500 | `gx4v9lcz` | checkpoint 1499 |
| Split kick-up from the fold's handovers | `Mjlab-HeadstandKickup-Flat-MicroDuck` | `y21jqomx` 1499 | same as above | 1500 | `y2fllvgj` | checkpoint 1499 |
| Back roll from the straight hold | `Mjlab-HeadstandBackrollLegsTogether-Flat-MicroDuck` | `44wneb8l` 1000 (run 1, feet-only collision model) | none | 2000 | `ax1vgv8z` | checkpoint 1999 |
| Split switch | `Mjlab-HeadstandSplitSwitch-Flat-MicroDuck` | `y21jqomx` 1499 | `HEADSTAND_OMEGA_MAX=2.0 HEADSTAND_OVERSPEED_W=-0.5 HEADSTAND_BANK_PROB=0.3 HEADSTAND_POLISH_AT=0` | 1500 | `xikztubv` | no, the ancestor |
| Fast split switch | `Mjlab-HeadstandSplitSwitch-Flat-MicroDuck` | `xikztubv` 1499 | same plus `HEADSTAND_SWITCH_RAMP_S=0.3` | 1000 | `whv1lcu7` | checkpoint 500 (750 and 999 flop in the chain) |
| Back roll from the split hold (tucks) | `Mjlab-HeadstandBackrollSplit-Flat-MicroDuck` | `ax1vgv8z` 1999 | none | 1500 | `bhcxmnvs` | no, the ancestor |
| Split over into the roll | `Mjlab-HeadstandSplitOver-Flat-MicroDuck` | `bhcxmnvs` 1499 | none (mirrored spawns are in the cfg) | 1500 | `2v46kg6g` | checkpoint 1499 |

Runs that did not make it: the tucked two-leg hop (`7m3uadlz`, learned a backbend), the slow split-over (`vnq2ka0n`, `zzd8uhag`, slowed to 5 rad/s but tucked the knees), the split exit to the pike (`fxhauoka`, replaced by the split over), and the single end-to-end headstand task (`Mjlab-Headstand-Flat-MicroDuck`, runs 1 to 4, which slammed or fell backward and led to the split into fold and kick-up).

## Training with the current code

Run from `microduck_rl/`. The wrapper constructs the task with explicit factory arguments and forwards the arguments after `--` to mjlab. Inspect a configuration before training:

```bash
uv run ../tools/train_variant.py \
  --task Mjlab-HeadstandSplitSwitch-Flat-MicroDuck \
  --factory-json '{"handover_prob": 0.3, "polish_at": 0, "switch_ramp_s": 0.3}' \
  --inspect
```

Continue the split-switch policy from its earlier checkpoint, resetting the training iteration and curricula:

```bash
MICRODUCK_WARM_START=1 uv run ../tools/train_variant.py \
  --task Mjlab-HeadstandSplitSwitch-Flat-MicroDuck \
  --factory-json '{"handover_prob": 0.3, "polish_at": 0, "switch_ramp_s": 0.3}' -- \
  --env.scene.num-envs 4096 --agent.max_iterations 1000 --agent.resume True \
  --wandb-run-path zachgarner-ai/mjlab_microduck/xikztubv \
  --wandb-checkpoint-name model_1499.pt
```

For the final kick-up runs, use the relevant kick-up task and
`--factory-json '{"handover_prob": 0.5, "polish_at": 0, "omega_max": 2.0, "overspeed_w": -0.5}'`,
with the starting checkpoint listed in the table. These commands configure a new
run using the current implementation. They do not reproduce historical runs bit
for bit: the training code changed, and some ancestor tasks are no longer registered.

The old environment-variable names in the table are provenance, not executable
instructions. `HEADSTAND_BANK_PROB` corresponds to `handover_prob`;
`HEADSTAND_POLISH_AT`, `HEADSTAND_OMEGA_MAX`, `HEADSTAND_OVERSPEED_W`, and
`HEADSTAND_SWITCH_RAMP_S` correspond to the lowercase factory argument names.
The old `HEADSTAND_RAMP_S` has no current equivalent.

For an Anyscale job, pass the same JSON object through `submit.py --factory-json`.
The submitter's `--dry-run` prints the job without submitting or starting a paid run.
