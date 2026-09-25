# Stage success by handover pitch_rate_offset

Each panel pins one parameter and runs every grid value. The shaded band is the training range, the dashed line the nominal value, and the grey line the baseline, 90%.

![overview](overview.png)

The failure edge is the first value, moving outward from nominal, whose 95% interval lies entirely below the baseline.

| Axis | Training range | Lower failure edge | Upper failure edge | Worst success inside the training range | All attempts | Most common first unfinished stages |
| --- | --- | --- | --- | ---: | ---: | --- |
| fold2 (rad/s) | not randomized | -3 | none in grid | 100% | 129/160 |  |
| legs_together (rad/s) | not randomized | none in grid | none in grid | 100% | 160/160 |  |
| roll (rad/s) | not randomized | none in grid | none in grid | 94% | 145/160 |  |
| settle2 (rad/s) | not randomized | none in grid | none in grid | 100% | 160/160 |  |
| settle (rad/s) | not randomized | none in grid | none in grid | 100% | 160/160 |  |
| split (rad/s) | not randomized | none in grid | 1.5 | 72% | 125/160 |  |
| splitover (rad/s) | not randomized | none in grid | none in grid | 94% | 155/160 |  |
| switch (rad/s) | not randomized | none in grid | none in grid | 100% | 160/160 |  |
