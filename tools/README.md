# Tools

Run these from inside `microduck_rl/` with `uv run ../tools/<script>`, so the fork's environment and task registry are loaded. Each script prints a count out of 32 and, with `--video`, writes the first attempt as an mp4.

`check32.py` runs one policy 32 times from one start state with the simulator's automatic resets off, and reports the end states, the time to reach the headstand angle, the peak head force and how many stand at the end. `--bucket` picks the start: `tripod` (the resting pike), `bank` (the pikes the fold policy hands over), `hold`, `standing`.

`check_switch.py` holds the split for 1.5 s, flips the switch flag for every attempt, and counts who ends inverted in the mirrored split and how fast.

`check_splitover.py` runs an exit from the split hold (`--mirrored` for the other split) and measures whether the legs stay split and the knees straight while the trunk goes over, plus the peak rotation rate.

`routine_full.py` chains the policies on held state conditions and reports how many attempts reach each stage. The record command is in the top-level README.

`collect_handover.py` records the states the fold policy leaves the duck in and writes the bank the kick-ups train from (`microduck_rl/src/mjlab_microduck/tasks/handover_pike_from_fold.npz`).

`eval_checkpoint.py` is the older per-bucket eval the others import from (the spawn switch and the end-state classifier).
