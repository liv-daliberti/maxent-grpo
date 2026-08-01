# E72 B3a — replay-gradient ablation

`#modes` is distinct@8 at the terminal pass-12 checkpoint. Reference arms
are the frozen published values, not recomputed here.

| domain | seeds | Dr.GRPO #modes | B3a #modes | xGRPO #modes | B3a-xGRPO | B3a-Dr.GRPO |
| --- | --- | --- | --- | --- | --- | --- |
| Graph coloring | 5/5 | 0.325 | 0.429 | 2.406 | -1.977 | 0.104 |
| Countdown | 5/5 | 0.627 | 0.630 | 1.893 | -1.263 | 0.003 |
| Python factors | 2/5 | _withheld: fewer than five terminal seeds_ | | | | |
| MathIR action menu | 5/5 | 0.666 | 0.550 | 0.926 | -0.375 | -0.116 |
| PantryPlan | 5/5 | 0.634 | 0.564 | 2.186 | -1.622 | -0.070 |

## Integrity

All summarized runs observed `replay_compute_only=1`, zero applied replay gradient on every logged update, and active novelty credit.
