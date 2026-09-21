# Physics measurements

One-off measurements from before and during training, kept because the task cfg's numbers come from them. Run from inside `microduck_rl/` with `uv run ../physics/<script>`.

`settle_sweep.py` dropped the duck into a grid of inverted poses and found the ones that rest on the head. Result: the hold pose (left hip 1.2, right hip 0.8, neck 1.0, head 1.25) balances, and the trunk rests at z = 0.117 m.

`tripod_check.py` measured the resting pike, head and both feet down, that the kick-ups start from. `pike_start.json` is that state.

`scripted_kickup.py` played hand-written kick-ups from the pike under the plain servo model and showed the servos can swing the trunk to 180° in 0.8 s, before any training.

`entry_modes.py` compared Zach's two entries, the two-leg hop and the deep split with a small push, the same way.

`head_hop.py` answered whether the duck can hop on its head: scripted neck snaps and leg pumps under the BAM actuator model never lift the head off the floor, and the trunk rises 3 mm at most.

`show_pose.py` renders one pose to a PNG.
