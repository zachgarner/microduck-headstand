# Microduck headstand

Teaching Pollen Robotics' [Microduck](https://github.com/pollen-robotics/microduck) a headstand routine, in simulation, by reinforcement learning. Sim only so far; the robot ships before Christmas 2026.

The routine, from standing: **fold → straight-legs kick-up → hold → back roll to standing → fold → split kick-up → hold → switch the split in the air → split over into a back roll → standing.**

Routine v15, 32 of 32 episodes complete every stage, 11.8 s end to end. Video: [`results/routine/routine_full_v15_splitover.mp4`](results/routine/routine_full_v15_splitover.mp4).

![routine v15](results/routine/routine_full_v15_splitover_strip.png)

## How it is built

Eight policies, each a PPO policy trained in [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) on Pollen's `microduck_rl` stack, chained at runtime on held state conditions (in the pike, inverted, standing), the way Pollen's runtime hot-swaps ONNX policies on a shared 61-D observation. The task code lives on the `headstand` branch of the fork in [`microduck_rl/`](microduck_rl/) (a submodule); this repo holds the tooling, the physics probes, the training jobs and the results.

| Piece | Policy (wandb run, checkpoint) | Standalone check, 32 episodes, resets disabled |
| --- | --- | --- |
| Fold, standing to the resting pike | `vorty4kb` 1000 | 32/32, head lands at 13 N |
| Straight-legs kick-up, from the pikes the fold hands over | `gx4v9lcz` 1499 | 32/32, 0.38 s to the headstand, 8.9 N |
| Split kick-up, same start | `y2fllvgj` 1499 | 32/32, 0.40 s, 9.0 N |
| Back roll, straight hold to standing | `ax1vgv8z` 1999 | 32/32 |
| Split switch in the hold, on a flag | `whv1lcu7` 500 | 32/32 mirrored, 0.2 s |
| Split over into a back roll, from either split | `2v46kg6g` 1499 | 32/32 standing, knees 0.26 rad and hips 0.9 rad split before the lead foot lands |
| Stand from the pike (Pollen's published `alpha_stand.onnx`) | — | 31/32 |

Physics: the `allcollisions` model, BAM voltage-limited actuators. Every count comes from `tools/check32.py`, `tools/check_switch.py` or `tools/check_splitover.py` with terminations disabled, and the frames around each state change were looked at before the count was believed (three "successes" in the first two days were resets, a mislabelled fall and a respawn).

## Layout

- `microduck_rl/` — the fork, `headstand` branch: task cfgs, rewards, tests (submodule).
- `tools/` — evaluation and the routine runner (`routine_full.py`), the handover-bank collector.
- `anyscale/` — training jobs (single L40S node per run), `train.sh`, the image Dockerfile.
- `results/routine/` — the routine videos and frame strips, by version. `results/policies/` — per-policy checks. `results/strips/` — the early frame strips.

## Lessons that survived

- Pay the swing as a frontier (never charge falling back), gate it on support and flatness, cap it with a ramped setpoint; the hold jackpot buys slams otherwise.
- Terminate a flop during kick-up discovery; bill it per step and the policy freezes in the tripod.
- Train the next stage from the states the previous stage actually hands over (a recorded bank), not from the designed spawn: the routine went from 23/32 to 30/32 on that alone.
- A checkpoint that is 32/32 from its own spawn can be 0/32 in the chain. Promote into the routine only after the routine says so.
- Slow entries need a rate cap plus an ahead-of-ramp penalty; the ramp alone defers pay and the policy snaps up and waits.

## Running it

```
cd microduck_rl && uv sync
uv run ../tools/routine_full.py --fold vorty4kb:model_1000.pt --straight gx4v9lcz:model_1499.pt \
  --roll ax1vgv8z:model_1999.pt --split y2fllvgj:model_1499.pt --splitexit fxhauoka:model_750.pt \
  --switch whv1lcu7:model_500.pt --switch-hold-s 1.0 --single-switch --splitover 2v46kg6g:model_1499.pt \
  --stand policies/pollen/alpha_stand.onnx --hold-s 2 --settle-s 1.0 --pike-settle-s 0 \
  --episodes 32 --seconds 18 --orbit-deg-s 60 --video routine.mp4
```

Checkpoints download from the wandb project `zachgarner-ai/mjlab_microduck` on first use. Pollen's standing policy comes from the Hub repo `pollen-robotics/microduck-policies`.
