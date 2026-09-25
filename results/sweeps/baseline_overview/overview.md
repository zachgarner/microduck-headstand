# Routine success by pinned parameter

Each panel pins one parameter and runs every grid value. The shaded band is the training range, the dashed line the nominal value, and the grey line the baseline, 87/96 (91%), the success rate of `routine_v16_unpinned`.

![overview](overview.png)

The failure edge is the first value, moving outward from nominal, whose 95% interval lies entirely below the baseline.

| Axis | Training range | Lower failure edge | Upper failure edge | Worst success inside the training range | All attempts | Most common first unfinished stages |
| --- | --- | --- | --- | ---: | ---: | --- |
| armature_scale (x nominal) | 0.9 to 1.1 | 0.9 | 1.6 | 75% | 188/224 | roll 22, legs_together 7, splitover 4 |
| battery_voltage (V) | 6.5 to 8.2 | 5.5 | none in grid | 84% | 194/224 | roll 18, switch 6, legs_together 6 |
| command_delay (physics steps (5 ms)) | 3 to 6 | 3 | 8 | 69% | 138/288 | roll 88, legs_together 40, splitover 11 |
| contact_friction_scale (x nominal) | not randomized | 0.7 | none in grid | 94% | 145/192 | roll 30, splitover 6, switch 6 |
| encoder_bias (rad) | 0 to 0.015 | none in grid | 0.015 | 78% | 170/192 | roll 12, legs_together 5, switch 3 |
| friction_scale (x nominal) | 0.9 to 1.1 | none in grid | 1.6 | 84% | 197/256 | roll 40, splitover 9, legs_together 5 |
| head_com_x (m) | -0.01 to 0.01 | -0.02 | none in grid | 84% | 197/224 | roll 11, switch 8, legs_together 6 |
| head_com_z (m) | -0.01 to 0.01 | -0.02 | 0.02 | 88% | 190/224 | legs_together 14, roll 12, switch 7 |
| imu_misalignment (deg) | 0 to 6 | none in grid | 6 | 75% | 100/192 | legs_together 50, switch 14, roll 12 |
| trunk_com_x (m) | -0.015 to 0.015 | -0.015 | 0.015 | 62% | 186/288 | legs_together 34, roll 29, splitover 12 |
| trunk_com_y (m) | -0.015 to 0.015 | -0.008 | 0.015 | 28% | 135/288 | switch 71, legs_together 44, roll 23 |
| trunk_com_z (m) | -0.015 to 0.015 | -0.008 | 0.03 | 56% | 210/288 | legs_together 40, roll 26, splitover 6 |
| trunk_mass_scale (x nominal) | 0.95 to 1.05 | none in grid | 1.2 | 84% | 197/224 | roll 14, legs_together 7, switch 5 |
| voltage_drop_gain (V/Nm) | 0 to 0.2 | none in grid | 0.4 | 84% | 162/192 | roll 15, legs_together 7, switch 6 |
