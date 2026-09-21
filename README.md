# Microduck headstand

The Microduck does a full headstand routine in simulation, from standing back to standing, in 32 of 32 attempts (routine version 15, September 21 2026). The Microduck is Pollen Robotics' 25 cm biped: 14 servos, 800 g, no arms. Pollen targets deliveries before Christmas 2026, and until the robot arrives all of this is simulation.

From standing, the duck folds forward until its head is on the floor, kicks up to a headstand with the legs together, holds, rolls backward onto its feet, folds again, kicks up to a headstand with the legs in a split (one leg forward, one back), swaps which leg leads while inverted, and continues the split over into a roll back to standing. The routine takes 11.8 s. The video is [`results/routine/routine_full_v15_splitover.mp4`](results/routine/routine_full_v15_splitover.mp4), and [`results/README.md`](results/README.md) lists every version before it.

![the routine, one frame per stage](results/routine/routine_full_v15_splitover_strip.png)

## How it is built

The routine is a chain of seven policies, six trained here plus Pollen's published stand-up, each handing over once the duck has held a pose for a set time. A policy is a small neural network that reads the duck's 61 sensor and command values and writes the 14 servo targets, 50 times a second. I trained each one with PPO, a reinforcement-learning algorithm, in [mjlab](https://github.com/mujocolab/mjlab), a physics simulator built on MuJoCo that runs 4,096 ducks at once on one GPU. The training code is Pollen's `microduck_rl`. The handovers fire on held poses: head and feet on the floor for 0.3 s, inverted for 2 s, standing for 1 s. The split switch policy also reads a command flag, 0 or 1, that says which leg leads. Pollen's robot software switches policies from the same 61 values, so the chain should port as is. That is untested, since the robot has not arrived.

Each row below is one policy: its training run on wandb (the training log service) and the checkpoint iteration the routine uses, then a check of the policy from its own start state, 32 attempts, with the simulator's automatic resets off so a fall stays a fall. "N" is the peak force between the head and the floor. The duck weighs about 8 N.

| Policy | wandb run, checkpoint | Check from its own start, 32 attempts |
| --- | --- | --- |
| Fold, standing to the pike (head and both feet on the floor) | `vorty4kb` 1000 | 32/32, 13 N |
| Kick-up to the legs-together headstand, from the pikes the fold hands over | `gx4v9lcz` 1499 | 32/32, 0.38 s to the headstand, 8.9 N |
| Kick-up to the split headstand, same start | `y2fllvgj` 1499 | 32/32, 0.40 s, 9.0 N |
| Back roll, legs-together headstand to standing | `ax1vgv8z` 1999 | 32/32 |
| Split switch, on the flag | `whv1lcu7` 500 | 32/32 mirrored in 0.2 s |
| Split over into a back roll, from either split | `2v46kg6g` 1499 | 32/32 standing, knees straight (0.26 rad) and hips split (0.9 rad) until the lead foot lands |
| Stand up from the pike (Pollen's `alpha_stand.onnx`) | not trained here | 31/32 |

The simulation uses Pollen's robot model with every part able to collide, and their BAM actuator model, a fitted model of the XL330 servo that caps each motor at the voltage the real one gets. Before counting a success I read the simulation frames around every state change. Three early passes were a reset, a mislabelled fall and a respawn, not headstands. The check scripts and the routine runner are in [`tools/`](tools/), and [`anyscale/runs.md`](anyscale/runs.md) lists every training run with what it started from.

## Layout

`microduck_rl/` is my fork of Pollen's training code, on its `headstand` branch, with the task definitions, rewards and tests, checked out as a submodule. `probes/` has the physics measurements the task numbers came from. `anyscale/` has the training jobs for Anyscale, the hosted Ray service the runs used, one L40S GPU per run. `tools/` has the check scripts and the routine runner. `results/` has the videos and frame strips, the routine by version and each policy on its own.

## Reward rules that worked

Reward each new best angle of the swing toward the headstand, and nothing else about the swing, so a fall back costs nothing and rocking earns nothing. Pay it only while the weight is on the head and the trunk is level above it. A reward for reaching the hold produced head slams until I conditioned it on a soft landing.

End the training attempt the moment the duck flops during kick-up training. Charging a per-step penalty for the flop instead made the cheapest strategy freezing in the tripod, head and both feet on the floor.

Train each policy from the states the previous policy hands over. I trained the kick-ups from 512 pike states recorded from the fold policy, and an earlier routine went from 23 of 32 to 30 of 32 on that change alone.

A checkpoint that is 32 of 32 from its own start can be 0 of 32 in the routine. The split switch passed every check from its own start at iteration 999 and flopped in the routine, where it takes over from the kick-up's hold. Iteration 500 passed the routine at 30 of 32, so that is the checkpoint in the chain.

## Running it

This runs the routine 32 times and writes the first attempt to `routine.mp4`.

```
cd microduck_rl && uv sync
uv run ../tools/routine_full.py --fold vorty4kb:model_1000.pt --straight gx4v9lcz:model_1499.pt \
  --roll ax1vgv8z:model_1999.pt --split y2fllvgj:model_1499.pt \
  --switch whv1lcu7:model_500.pt --switch-hold-s 1.0 --single-switch --splitover 2v46kg6g:model_1499.pt \
  --stand policies/pollen/alpha_stand.onnx --hold-s 2 --settle-s 1.0 --pike-settle-s 0 \
  --episodes 32 --seconds 18 --orbit-deg-s 60 --video routine.mp4
```

The checkpoints download on first use from the wandb project `zachgarner-ai/mjlab_microduck`, which needs a wandb login with access to it. Pollen's stand-up policy is `alpha_stand.onnx` in the Hugging Face repo `pollen-robotics/microduck-policies`. Download it to `microduck_rl/policies/pollen/`.
