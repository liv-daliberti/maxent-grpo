# Independent E118 / frozen E120 numerical review

Reviewed 2026-09-06 UTC. No manuscript, result artifact, scheduler, or training state was changed by this review.

## E118 Qwen2.5-3B Python

Independently parsed all 20 raw arm endpoints (four methods, registered seeds 70–74), requiring exact step 3072, evaluation kind `fixed_seed_sampled_k_neutral`, sample count 8, draws 0–3, and finite endpoint metrics. There were no repeated or conflicting terminal metric payloads. All values matched `paper/figures/e118_all_scale_factorial_progress.json` and the four-row `paper/results/e118_qwen3b_python_terminal_table_body.tex`.

| Method | pass@8 | raw distinct@8 |
|---|---:|---:|
| Dr.GRPO | 0.068750 | 0.068750 |
| ReplayDr.GRPO | 0.65859375 | 0.85234375 |
| MaxRL | 0.356250 | 0.356250 |
| ReplayMaxRL | 0.6328125 | 0.74765625 |

Independent paired Student-t calculation: ReplayMaxRL minus MaxRL is +0.2765625 on pass@8, interval [-0.036915277, +0.590040277], and +0.39140625 on raw distinct@8, interval [+0.078679823, +0.704132677]. All five raw distinct effects are positive; pass effects have three positive values and two zeros. The correctness interval includes zero. These results alone do not establish correctness-adjusted breadth, individual-mode survival, or model-size generality.

Independently counted 11 complete five-seed MaxRL pair blocks and 10 complete five-seed four-arm blocks. Falcon Countdown retains four Dr.GRPO-track seeds and five MaxRL-track seeds. The other four Qwen3B MaxRL domains are incomplete.

## E120 frozen Qwen2.5-0.5B primary contrast

Input: `paper/results/e120_frequency_progress.json`, generated 2026-09-04T15:40:26.896956+00:00, SHA256 `92304ed9ac70f6ebc4dd38e12801750175b96703bf657459dafed3c390bf2bab`.

Independently parsed all 50 raw endpoints (25 frequency treatment and 25 uniform comparator), with the same endpoint requirements above. Every endpoint matched the frozen input, with no repeated or conflicting terminal metric payloads. Current registered ledger SHA256 matches the frozen source hash `5302ba26da3f67d0b57e9eef1e8f528c6a56aef5faeb543a3adbfe32e721d8e7`. This numerical review did not repeat the training-telemetry audit.

Let B = distinct@8 - pass@8. Positive effects below mean uniform minus frequency. Bootstrap numbers were calculated independently from the proposed builder: enumerate all 5^5 ordered resamples of the five paired seed differences, then take linearly interpolated 2.5th and 97.5th percentiles. For the cross-domain result, first average all five domains within each seed and resample those five seed averages.

| Domain | B effect | Paired percentile 95% interval | Positive / zero / negative seed effects |
|---|---:|---|---|
| Graph | +0.596484375 | [+0.512890625, +0.687109375] | 5 / 0 / 0 |
| Countdown | +0.638671875 | [+0.544140625, +0.733203125] | 5 / 0 / 0 |
| Python | +0.037109375 | [-0.01171875, +0.123046875] | 1 / 3 / 1 |
| MathIR | +0.019140625 | [-0.00078125, +0.037890625] | 4 / 0 / 1 |
| Pantry | +0.29765625 | [+0.173828125, +0.400390625] | 5 / 0 / 0 |
| Five-domain mean | +0.3178125 | [+0.272265625, +0.3578125] | 5 / 0 / 0 |

The corresponding five-domain pass@8 effect is +0.00984375, interval [-0.119375, +0.1390625]. Its mean is close to zero but this does not establish equal accuracy, noninferiority, or comparable correctness. The raw distinct effect is +0.32765625, interval [+0.192890625, +0.462421875].

The registered estimand, contrast direction, paired bootstrap family, and five-domain mean are specified in `paper/preregistration/e120_frequency_weighted_replay_ablation_20260902.md`. The exact percentile algorithm, linear quantile convention, joint resampling of same-numbered seeds across domains, and lack of multiplicity adjustment are analysis implementation choices; they must not be described as explicitly preregistered details. The domain-specific breadth result is strongest on Graph, Countdown, and Pantry; Python and MathIR intervals include zero. Inferential limits must accompany any claim that key balancing explains the replay mechanism.

## Builder and generated-output review

Reviewed `ops/exp_scaling/build_paper_e120_primary_breadth.py` and `tests/test_paper_e120_primary_breadth.py`; no blocking scientific or implementation findings. The tests cover sign direction and removal of the success component, shared-seed aggregation, invalid source metadata/seeds/values, unchanged input, and a known discrete bootstrap distribution. The implementing agent reports all 13 focused tests passing.

Independently checked all 18 generated summaries, 90 individual seed contrasts, and 36 interval limits in `paper/results/e120_primary_breadth.json`. The second verification used unordered count vectors with multinomial weights, rather than the builder's ordered-resample enumeration. Every value agrees within absolute tolerance 1e-12. The rendered summary table agrees with the independently computed values.

The new bootstrap pass@8 intervals for Countdown [+0.004296875, +0.040234375] and Pantry [+0.005859375, +0.09140625] are positive even though their old Student-t intervals crossed zero. This is a change in the estimation procedure, not new training data. Manuscript prose must label the procedure accurately and avoid combining conclusions from one procedure with intervals from the other. The pooled correctness bootstrap interval still spans zero.
