# E16 canonical MaxEnt replication: completed sampled endpoints

All eighteen eligible Stage-R V3 jobs completed the frozen five-pass budget
with Slurm state `COMPLETED`, exit code `0:0`, and no watchdog alert. Across
the complete cohort, the training logs record zero invalid canonical actions.
The values below are the registered terminal sampled evaluation at temperature
1, averaged over paired training seeds 43/44/45. They are not the still-pending
exact full-support evaluation of every held-out policy.

| Domain | Method | pass@8 | mean@8 | coverage@8 | distinct@8 | pass@1 |
|---|---|---:|---:|---:|---:|---:|
| Graph coloring | fixed | 0.913 | 0.370 | 0.360 | 2.177 | 0.465 |
| Graph coloring | proportional | 0.892 | 0.357 | 0.338 | 2.042 | 0.361 |
| Graph coloring | Haarnoja dual | 0.899 | 0.369 | 0.352 | 2.132 | 0.497 |
| Countdown | fixed | 0.609 | 0.348 | 0.255 | 1.120 | 0.500 |
| Countdown | proportional | 0.651 | 0.440 | 0.287 | 1.263 | 0.552 |
| Countdown | Haarnoja dual | 0.651 | 0.442 | 0.267 | 1.156 | 0.539 |

The common initialization is identical across methods within each domain.
On graph coloring it has pass@8/mean@8/coverage@8/distinct@8/pass@1 of
0.757/0.274/0.243/1.576/0.337; on Countdown the corresponding values are
0.589/0.091/0.158/0.698/0.148. Thus every registered method improves every
reported sampled metric over initialization in both domains.

Relative to paired fixed MaxEnt, Countdown proportional control changes
pass@8 by +4.2 points, mean@8 by +9.1 points, coverage@8 by +3.3 points,
distinct@8 by +0.143 modes, and pass@1 by +5.2 points. The dual controller
changes the same outcomes by +4.2, +9.4, +1.2, +0.036, and +3.9 points/modes.
Graph coloring is already strong under fixed MaxEnt: the adaptive endpoints
are within 2.3 coverage points and 0.14 distinct modes of fixed, while the
dual arm has the highest mean greedy accuracy.

These are three-seed replication effect sizes, not asymptotic significance
claims. The result establishes stable finite-action direct MaxEnt training and
encouraging sampled-answer behavior at 0.5B. Exact held-out action entropy,
valid probability, and valid-mode effective support remain unreported until
the frozen terminal checkpoints receive the promised full-support audits.

## Provenance

- watchdog: `var/artifacts/e16_v3_full_watchdog_status.json`, SHA-256
  `a16fd624be2273d048b510af60cbef5b1573763488eca1c10de7c65e07dd81e9`;
- graph-coloring curve: `var/artifacts/gce16_canonical_maxent_05b_v2_scaling_curve.json`,
  SHA-256 `d61878c7ed6c025fe88914de485a1e0203f851700d1be7ade74bfe566e395c6f`;
- Countdown curve: `var/artifacts/cde16_canonical_maxent_05b_v2_scaling_curve.json`,
  SHA-256 `c76b0d707ad17bd3e2bf82882a218245583b30bdd4f3f7f886c2e03f9aae8216`.
