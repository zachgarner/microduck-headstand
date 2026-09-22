# Historical results

Current seeded evaluations are in [verified/](verified/). The counts below were recorded with earlier evaluators and are retained as experiment history. They are not evidence under the current contact, leg-shape, and final-standing checks.

Videos show one simulated attempt from each historical batch. Frame strips sample that attempt around the transitions. The tables retain the original counts and descriptions; evaluator behavior varied during development.

## The routine, version by version

| Version | What changed | Complete | Time | Files |
| --- | --- | --- | --- | --- |
| v1 | First chain of seven policies: fold, straight kick-up, hold, back roll, stand, fold, split kick-up, split exit to the pike, stand | 11 of 16 | 12.4 s | `routine/routine_full_v1.mp4` |
| v6 | Settle stages: the standing policy holds 1 s after the roll before the second fold | 23 of 32 | | `routine/routine_full_v6.mp4` |
| v7 | Kick-ups retrained from the pikes the fold actually hands over | 30 of 32 | 10.7 s | `routine/routine_full_v7.mp4`, `_strip.png` |
| v9 | Split switch added, there and back, before the split exit | 30 of 32 | 14.8 s | `routine/routine_full_v9_switch.mp4`, `routine_full_v8_switch_strip.png` |
| v13 | The split back roll replaces the split exit and the stand-up, the switch is faster, and the camera orbits through the switches | 30 of 32 | 11.9 s | `routine/routine_full_v13_orbit.mp4`, `routine_full_v13_strip.png` |
| v14 | One switch instead of there and back, with the exit from the mirrored split | 23 of 32 (a high-variance run, the same policies scored 30 to 32 on other seeds) | 11.7 s | `routine/routine_full_v14_oneswitch.mp4`, `_strip.png` |
| **v15** | Split-over exit added. Historical evaluation. | **32 of 32** | 11.8 s | `routine/routine_full_v15_splitover.mp4`, `_strip.png` |

Versions not listed (v2 to v5, v8, v10 to v12) were settle-time and checkpoint experiments. Their numbers are in the training notes on the fork's issue tracker.

## Each policy on its own

Checks from the policy's own start state. Counts are out of 32. "s" is the time to reach the headstand angle and "N" the peak force between the head and the floor.

| Piece | File | Result |
| --- | --- | --- |
| Fold, standing to the pike, iteration 250 | `policies/fold2_iter250_standing_start.mp4` | 32/32, 13 N |
| Straight kick-up, snap, iteration 250 | `policies/straight1_iter250_pike_start.mp4` | 32/32, 0.2 s |
| Split kick-up, snap, iteration 1000 | `policies/kickup3_iter1000_pike_start.mp4` | 32/32, 0.20 s, 11.7 N |
| Slow split kick-up, iteration 1500 | `policies/split_slow3_iter1500_pike_start.mp4` | 32/32, 1.63 s, 8.7 N |
| Slow straight kick-up, iteration 1500 | `policies/straight_slow5_iter1500_pike_start.mp4` | 31/32, 1.54 s |
| Straight kick-up from the fold's handovers, iteration 1499 (in the routine) | `policies/straight_chain1_iter1499_bank_start.mp4`, strip at 250 | 32/32, 0.38 s, 8.9 N |
| Split kick-up from the fold's handovers, iteration 1499 (in the routine) | `policies/split_chain1_iter1499_bank_start.mp4`, strip at 250 | 32/32, 0.40 s, 9.0 N |
| Split switch, iteration 1499 | `policies/splitswitch1_iter1499_hold_start.mp4`, strip at 750 | 32/32 mirrored, 0.5 s |
| Fast split switch, iteration 999 | `policies/splitswitch_fast1_iter999_hold_start.mp4` | 32/32 mirrored in 0.16 s alone, and flops in the routine, which uses iteration 500 |
| Back roll from the straight hold, iteration 1000 | `policies/backroll1_iter1000_headstand_start.mp4` | 32/32 standing |
| Back roll from the split hold, iteration 1499 (tucks) | `policies/backroll_split1_iter1499_splithold_start.mp4`, strip at 250 | 32/32 standing |
| Split over into the roll, iteration 500 (in the routine at 1499) | `policies/splitover2_iter500_splithold_start.mp4`, `_strip.png` | 32/32 standing |
| Tucked two-leg hop, iteration 750 (a backbend, not kept) | `policies/tucked1_iter750_pike_start.mp4` | 31/32 |
| The resting pike the kick-ups start from | `policies/tripod_start_pike.png` | |

## Early strips

`strips/` holds the frame strips from the first two days (the end-to-end headstand task, runs 1 to 4, and the first kick-ups), kept for the record. The tracked findings are in the issue tracker on the fork.
