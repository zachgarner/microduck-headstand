# Verified simulation results

Evaluation date: September 22, 2026.

The routine succeeded in **87/96 attempts** across seeds 0, 1 and 2. Median completion time among successful attempts was **12.30 s**.

Success requires every held stage, the specified leg shape in the headstand holds, and strict standing at the final frame. Each rollout lasts 18 seconds. Automatic resets are disabled. These are simulation results, not hardware validation.

| Seed | Success | Completed stages | Final standing | Resets |
| --- | --- | --- | --- | --- |
| 0 | 28/32 | 28 | 28 | 0 |
| 1 | 29/32 | 29 | 30 | 0 |
| 2 | 30/32 | 30 | 30 | 0 |

Failures by the first unfinished stage: legs_together: 3, roll: 4, settle2: 1, switch: 1.

## Individual policies

Each policy was checked in 32 environments at seed 0. Times are medians of the first qualifying goal state among attempts that also succeeded at the end. The switch time begins at the command change. Forces are the median and maximum of each attempt's largest 50 Hz sample, including initial settling. Shorter physics impacts can be missed.

| Policy | Success | Time to goal | Sampled head force, median / max |
| --- | --- | --- | --- |
| fold | 32/32 | 0.40 s | 11.3 / 34.8 N |
| legs-together | 32/32 | 0.40 s | 9.0 / 17.7 N |
| split | 32/32 | 0.42 s | 9.1 / 17.9 N |
| switch | 32/32 | 0.32 s | Not measured |
| backroll | 32/32 | 0.52 s | Not measured |
| splitover | 32/32 | 0.54 s | Not measured |

The exit counts assess final standing. They do not certify leg shape throughout the exit trajectory. The split-over check uses the task's mixed original/mirrored starts.

## Reproduction and evidence

From `microduck_rl/`, download the public checkpoints and standing ONNX, then run:

```bash
uv run ../tools/download_policies.py
uv run ../tools/run_verification.py --routine-seeds 0 1 2 --individual --workers 2
uv run ../tools/summarize_verification.py
```

Each rollout JSON file has a matching raw `.log`. Routine reports record the exact arguments, software versions, code revisions, evaluator-source hashes, checkpoint hashes and per-attempt stage times. Individual reports include the seed, checkpoint hash, evaluator hashes and per-attempt outcomes. The regression tests are in `tools/test_evaluation.py` and `tools/test_training_cli.py`.

The handover contact audit is a separate CPU `mj_forward` diagnostic of the 512 stored states. It found 511 with head and both feet contacting, and one with head and one foot. It does not replay the original Warp contact history. The historical handover set was kept unchanged for these checkpoint evaluations.
