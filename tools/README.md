# Evaluation and training tools

Run these scripts from `microduck_rl/` using its Python environment. The task definitions come from the `headstand-pr` branch of the training repository.

## Verify the checkpoints

The verification battery runs 32 full routines at each of three seeds, then checks the six trained policies separately:

```bash
uv run ../tools/run_verification.py --routine-seeds 0 1 2 --individual --workers 2
```

Each run writes a log and JSON report to `results/verified/`. Reports include the seed, checkpoint hashes, per-attempt outcomes, and evaluation-source hashes. The routine reports also include software versions and stage completion times. Automatic resets are disabled.

The checkpoints download from the W&B project `zachgarner-ai/mjlab_microduck` when they are not already cached. This requires W&B access. Download Pollen's `alpha_stand.onnx` from `pollen-robotics/microduck-policies` on Hugging Face to `microduck_rl/policies/pollen/` before running the battery.

## What counts as success

A pike requires head contact, both feet down, no other monitored body contact, a downward-facing trunk, and trunk pitch between 60° and 95°. Standing requires both feet down, no head or other monitored body contact, and orientation within 30° of upright.

A headstand requires head-only support and orientation within 35° of inverted. The kick-ups also require both knees within 0.3 rad of straight. The legs-together hold allows at most 0.2 rad of hip separation; the split hold requires at least 1.0 rad. Separation is `abs(left_hip_pitch + right_hip_pitch)` under the model's joint sign convention. A completed switch must satisfy the split hold and be closer to the requested pose than the opposite pose.

The routine must complete every hold and still satisfy the standing criterion at the final frame. A completed routine followed by a fall is a failure. These checks measure the held positions; the separate split-over diagnostic measures leg shape during the exit itself.

Head forces are sampled at 50 Hz. They are not maximum forces across the 200 Hz physics substeps. Exit tasks whose sensors lack a force field report force as unmeasured.

## Individual tools

| Script | Purpose |
| --- | --- |
| `routine_full.py` | Run the full sequence, optionally recording video. Use `--seed` and `--json` to preserve evaluation conditions and results. |
| `check32.py` | Evaluate one policy. Set `HEADSTAND_TASK` to the task ID; use `--bucket standing`, `handover`, `pike`, or `hold`. |
| `check_switch.py` | Start in the split hold, change the command flag, and measure the resulting hold. |
| `check_splitover.py` | Measure leg shape during the exit from the original split; add `--mirrored` for mirrored-only starts. |
| `eval_checkpoint.py` | Shared success predicates and an older exploratory evaluator. Use the tools above for no-reset evidence. |
| `collect_handover.py` | Historical collection script. Use `microduck_rl/scripts/headstand/collect_handover.py` for the current contact checks. |
| `train_variant.py` | Construct a task using explicit factory arguments and launch mjlab training. See `anyscale/README.md`. |

Run the regression checks for evaluation and factory overrides:

```bash
uv run --with pytest pytest ../tools/test_evaluation.py
```
