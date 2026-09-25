# Stage success by handover pitch_offset

Each panel pins one parameter and runs every grid value. The shaded band is the training range, the dashed line the nominal value, and the grey line the baseline, 90%.

![overview](overview.png)

The failure edge is the first value, moving outward from nominal, whose 95% interval lies entirely below the baseline.

| Axis | Training range | Lower failure edge | Upper failure edge | Worst success inside the training range | All attempts | Most common first unfinished stages |
| --- | --- | --- | --- | ---: | ---: | --- |
| fold2 (deg) | not randomized | -10 | none in grid | 100% | 160/224 |  |
| legs_together (deg) | not randomized | none in grid | 10 | 100% | 188/224 |  |
| roll (deg) | not randomized | -20 | 20 | 91% | 193/224 |  |
| settle2 (deg) | not randomized | none in grid | none in grid | 100% | 222/224 |  |
| settle (deg) | not randomized | none in grid | none in grid | 100% | 224/224 |  |
| split (deg) | not randomized | -20 | 5 | 81% | 146/224 |  |
| splitover (deg) | not randomized | none in grid | none in grid | 100% | 218/224 |  |
| switch (deg) | not randomized | none in grid | none in grid | 100% | 224/224 |  |
