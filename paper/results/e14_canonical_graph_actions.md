# E14 canonical graph actions: retrospective outcome

This record was written after the E14 runs. It reports a single-seed
engineering calibration on graph coloring with Qwen2.5-0.5B-Instruct; it is
not a paper result or evidence about Countdown, larger models, or free-text
MaxEnt.

All three 128-update runs and their exact 27-action endpoint audits completed
successfully. The values below are evaluation-set means at `step_00128`,
except rollout reward, which is the mean over updates 97--128.

| Arm | Train / audit job | $\alpha$ | $H(A)$ | $P_{\mathrm{valid}}$ | $H_{\mathrm{valid}}$ | $N_{\mathrm{eff,valid}}$ | Final-32 reward |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | 30010773 / 30010861 | 0 | 1.345021 | 0.314074 | 0.747250 | 2.326380 | 0.330078 |
| M01 | 30010871 / 30010874 | 0.01 | 1.428254 | 0.310019 | 0.794647 | 2.442240 | 0.318359 |
| M05 | 30010870 / 30010875 | 0.05 | 1.817039 | 0.295782 | 0.931797 | 2.774941 | 0.292969 |

The frozen safety gates required mean $P_{\mathrm{valid}}>0.05$, at least
80% retention relative to C0, and positive final-32 reward. The diversity
gates required an action-entropy gain of at least $\log(1.25)=0.223144$ nats
and a valid-mode effective-support ratio of at least 1.25.

M01 was safe but not diversity-effective: its entropy gain was 0.083233 nats
and its support ratio was 1.049803. M05 was also safe and cleared the entropy
gate with a 0.472018-nat gain. It was a near miss on the only remaining gate:
its support ratio was 1.192815, a 19.28% gain rather than the required 25%
(5.72 percentage points short). Under the frozen binary rule, neither arm is
viable and E14 selects no fixed canonical MaxEnt dose.

The scientifically useful signal is narrow: in this finite action space, the
canonical entropy actuator genuinely raised exact action entropy, and M05
also raised entropy among valid actions, while retaining 94.18% of C0's exact
valid probability. The preregistered decision nevertheless remains negative
because that redistribution did not reach the required valid-mode-support
gain. With one seed, these measurements authorize neither a comparative claim
nor expansion to other seeds, scales, domains, or adaptive controllers.

Machine-readable values and evidence paths are in
[`e14_canonical_graph_actions.json`](e14_canonical_graph_actions.json).
