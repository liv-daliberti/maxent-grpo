# E15 canonical-action dose calibration: retrospective outcome

This record was written after E15 completed. It reports one seed (`9005`) of
an engineering calibration on graph coloring with
Qwen2.5-0.5B-Instruct. It is not a replication, a main-paper result, or
evidence about Countdown, larger models, free-text MaxEnt, or adaptive
controllers.

Both 128-update treatments and their exact 27-action endpoint audits completed
successfully. Endpoint values below are exact means over the 96 frozen eval
prompts at `step_00128`; reward is the mean over updates 97--128.

| Arm | Train / audit job | $\alpha$ | $H(A)$ | $P_{\mathrm{valid}}$ | $H_{\mathrm{valid}}$ | $N_{\mathrm{eff,valid}}$ | Retention / support ratio vs C0 | Final-32 reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 30010773 / 30010861 | 0 | 1.345021 | 0.314074 | 0.747250 | 2.326380 | 1.000000 / 1.000000 | 0.330078 |
| M075 | 30012428 / 30012432 | 0.075 | 2.251622 | 0.281698 | 1.114829 | 3.310869 | 0.896915 / 1.423185 | 0.291016 |
| M10 | 30012427 / 30012431 | 0.10 | 2.797474 | 0.261672 | 1.418360 | 4.437708 | 0.833155 / 1.907559 | 0.271484 |

The frozen safety gates required exact mean valid probability above `0.05`,
retention of at least `0.80` relative to C0, and positive final-32 reward.
The diversity gates required an exact action-entropy gain of at least
`log(1.25)=0.22314355131420976` nats and a valid-mode effective-support ratio
of at least `1.25`. Both arms passed every gate. M075 gained `0.906601` nats
of exact action entropy and 42.32% valid-mode support; M10 gained `1.452453`
nats and 90.76% valid-mode support.

The frozen rule selects the larger-support arm when both are viable unless
their support values are within 5%, in which case it selects M075. M10's
support was 34.03% larger than M075's, so E15 selects **M10**. M10 retained
83.32% of C0 valid probability and positive reward, but the margin above the
80% retention floor was only 3.32 percentage points. That is calibration
evidence, not an estimate of robustness across seeds.

This result remains firewalled to one 0.5B graph-coloring seed. It authorizes
only drafting and reviewing a separately frozen three-seed 0.5B replication
protocol—not its automatic submission, scale/domain expansion, adaptive
control, analytical-grid insertion, or a main-paper claim.

Machine-readable metrics, jobs, gates, evidence paths, and SHA-256 identities
are in
[`e15_canonical_dose_calibration.json`](e15_canonical_dose_calibration.json).
