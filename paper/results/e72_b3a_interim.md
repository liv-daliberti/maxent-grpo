# E72 B3a — replay-gradient ablation (INTERIM)

**Interim: not a result, and biased against B3a.** The protocol reports a
domain only when all five seeds are terminal. Rows below may include seeds
still training, whose most recent evaluation is at an *earlier* pass than
the pass-12 references they are differenced against. Breadth rises through
training in every arm measured so far, so a partial row understates B3a by
an unknown amount. Read these only as evidence that the pipeline is
producing numbers, never as a direction.

`#modes` is distinct@8 at the terminal pass-12 checkpoint. Reference arms
are the frozen published values, not recomputed here.

| domain | seeds | Dr.GRPO #modes | B3a #modes | xGRPO #modes | B3a-xGRPO | B3a-Dr.GRPO |
| --- | --- | --- | --- | --- | --- | --- |
| Graph coloring | 5/5 | 0.325 | 0.429 | 2.406 | -1.977 | 0.104 |
| Countdown | 5/5 | 0.627 | 0.630 | 1.893 | -1.263 | 0.003 |
| Python factors | 0/5 (+3 partial) | 0.172 | 0.087 | 1.676 | -1.589 | -0.085 |
| MathIR action menu | 0/5 (+5 partial) | 0.666 | 0.425 | 0.926 | -0.501 | -0.241 |
| PantryPlan | 0/5 (+5 partial) | 0.634 | 0.566 | 2.186 | -1.620 | -0.068 |

## Integrity

All summarized runs observed `replay_compute_only=1`, zero applied replay gradient on every logged update, and active novelty credit.
