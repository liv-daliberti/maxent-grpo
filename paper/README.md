# Paper

[`main.tex`](main.tex) is the NeurIPS-format source for **ModeBench:
Executable Outcome Discovery for Maximum-Entropy Reasoning**.
[`main.pdf`](main.pdf) is the built manuscript.

The paper centers the final long-horizon design:

- four ModeBench environments with execution-bound correctness and mode keys;
- an online bank containing only policy-generated, validator-positive outcomes;
- an open-set predictive advantage with a structural unseen bucket, plus a
  one-time discovery credit of `0.50`;
- one globally scheduled verified replay bank per optimizer update, under
  split verified-mass and known-mode-balance terms;
- three projection-free controllers referenced only to their own warmup means;
- a support-only, entropy-gated singleton actuator whose proposals reach the
  replay layer but never the on-policy advantage; and
- identical passive discovery telemetry in the matched Dr.GRPO control.

Graph coloring, Countdown, executable Python factors, and executable MathIR
action menus are the four ModeBench domains. MathIR keys the exact rational
equation-state trajectory produced by a successful execution. **It is not
MATH-500.**

The held-out MATH-500 track is a realism/generalization test, not a fifth
multi-mode ModeBench row. It trains on a frozen 384-row MATH12K subset and
evaluates on all 500 disjoint MATH-500 problems. All verifier-positive
completions for a prompt share one `correct` canonical key, so verified-mass
replay is eligible but known-mode balance must remain structurally inactive.
This supports an honest correctness-transfer test without calling answer
formatting or free-form prose a reasoning mode.

## Evidence boundary

The reported campaign is 54 registered runs — 24 method-versus-control
ModeBench runs, 6 MATH-500 runs, 12 same-plumbing actuator-off controls, and
12 separated-support actuator treatments — at exactly 12 training passes,
seeds 43--45, on Qwen2.5-0.5B-Instruct. It is **still executing**.

Every number in the manuscript is an **interim fixed-checkpoint** reading at
the latest registered checkpoint that all three seeds of both compared arms
have landed. Domains stand at different passes and are not pooled. No terminal
value, no AUC, and no interpretation gate is reported as decided: all four
frozen gates (cross-domain positive, actuator repair, plumbing sensitivity,
held-out realism) are `pending`, and a pending gate is evidence of neither
success nor failure.

Two earlier cohorts are excluded from the 54-run denominator by
machine-readable invalidation audits, not by inspecting their results: one
whose runtime forced the novelty credit to `0.0`, and one whose shared
proposal/on-policy bank was rejected before any optimizer step. Both remain
archived as engineering evidence. Earlier experiments inform the method
rationale and decision table but are not pooled into the treatment estimate.

## Reproduce

The live surface — audits, curves, the results markdown and CSV, and both
paper figures — is regenerated on a fixed interval by the campaign monitor:

```bash
ops/exp_scaling/watch_e65_entropy_gated_singleton_confirmation.sh
```

To refresh only the figures used by the manuscript:

```bash
python ops/exp_scaling/plot_e64_math500_realism.py
python ops/exp_scaling/plot_e61r1_e58_vs_grpo_12pass.py
python ops/exp_scaling/plot_e65_all_epoch_diagnostic.py
```

Build or audit the Python environment from the repository root:

```bash
var/seed_paper_eval/paper310/bin/python \
  ops/make_python_factor_mode_data.py \
  --output-root var/data/python_factor_modebench_v1

var/seed_paper_eval/paper310/bin/python \
  ops/verify_python_factor_mode.py \
  --candidate 'lambda n: 2 if n % 2 == 0 else 3' \
  --reference \
  '{"verifier":"python_factor_function","python_version":"factor-v1","cases":[6,10,15]}'
```

Build the PDF:

```bash
make -C paper
```

## Primary artifacts

- Live five-domain figure (paper Figure 1):
  [`figures/e61r1_e58_vs_grpo_05b_12ep_live.pdf`](figures/e61r1_e58_vs_grpo_05b_12ep_live.pdf)
- Held-out MATH-500 realism figure (paper Figure 2):
  [`figures/e64_math500_realism_05b_12ep_live.pdf`](figures/e64_math500_realism_05b_12ep_live.pdf)
- All-epoch diagnostic (appendix; MATH display surface only):
  [`figures/e61r1_e58_vs_grpo_05b_12ep_all_epoch_diagnostic_live.pdf`](figures/e61r1_e58_vs_grpo_05b_12ep_all_epoch_diagnostic_live.pdf)
- Live confirmation report:
  [`results/e65_five_domain_confirmation_live.md`](results/e65_five_domain_confirmation_live.md)
- Seed-level fixed-checkpoint surface:
  [`results/e65_five_domain_confirmation_fixed_checkpoints_live.csv`](results/e65_five_domain_confirmation_fixed_checkpoints_live.csv)
- Machine-readable results:
  [`../var/artifacts/e65_five_domain_confirmation_results_latest.json`](../var/artifacts/e65_five_domain_confirmation_results_latest.json)
- Final fail-closed claim gate:
  [`../ops/exp_scaling/audit_e65_legitimate_result_readiness.py`](../ops/exp_scaling/audit_e65_legitimate_result_readiness.py)
- Campaign protocol:
  [`preregistration/e65_five_domain_terminal_confirmation_05b.md`](preregistration/e65_five_domain_terminal_confirmation_05b.md)
- Method-versus-control protocol:
  [`preregistration/e61r1_e58_vs_grpo_05b_12pass.md`](preregistration/e61r1_e58_vs_grpo_05b_12pass.md)
- Objective definition:
  [`preregistration/e58_global_verified_replay_canonical_05b.md`](preregistration/e58_global_verified_replay_canonical_05b.md)
- Same-plumbing actuator-off control:
  [`preregistration/e66_same_plumbing_actuator_ablation_05b.md`](preregistration/e66_same_plumbing_actuator_ablation_05b.md)
- Separated-support actuator:
  [`preregistration/e68_separated_support_actuator_ablation_05b.md`](preregistration/e68_separated_support_actuator_ablation_05b.md)
- MATH12K-to-MATH-500 realism protocol:
  [`preregistration/e64_math500_realism_transfer_05b.md`](preregistration/e64_math500_realism_transfer_05b.md)
- MathIR dataset identity:
  [`../var/data/mathir_action_menu_v1/identity.json`](../var/data/mathir_action_menu_v1/identity.json)
- MathIR executable task contract:
  [`../src/oat_drgrpo/mathir.py`](../src/oat_drgrpo/mathir.py)
- Python dataset identity:
  [`results/python_factor_modebench_v1_identity.json`](results/python_factor_modebench_v1_identity.json)
- Python task and syntax contract:
  [`../src/oat_drgrpo/python_modebench.py`](../src/oat_drgrpo/python_modebench.py)
- External process boundary:
  [`../src/oat_drgrpo/python_modebench_process.py`](../src/oat_drgrpo/python_modebench_process.py)
- Deterministic generators:
  [`../ops/make_python_factor_mode_data.py`](../ops/make_python_factor_mode_data.py),
  [`../ops/make_mathir_action_menu_data.py`](../ops/make_mathir_action_menu_data.py)
- External audit CLI:
  [`../ops/verify_python_factor_mode.py`](../ops/verify_python_factor_mode.py)
