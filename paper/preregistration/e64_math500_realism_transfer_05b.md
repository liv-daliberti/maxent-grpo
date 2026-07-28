# E64: held-out MATH-500 realism transfer for literal E58

**Status: FROZEN BEFORE GPU SMOKE — 2026-07-27**

## Question and scope

E64 asks whether the literal E58 controller transfers beyond finite
ModeBench tasks to ordinary free-form mathematical reasoning. It is an
external-validity and generalization track, not a fifth multi-mode ModeBench
domain.

MATH supplies one answer-equivalence class per problem. Ordinary final-answer
grading cannot certify that two free-form derivations use distinct reasoning
strategies. E64 therefore forbids treating surface form, answer formatting,
or unverified prose as canonical modes.

## Frozen train/test boundary

Use the existing immutable
`var/data/math12k_384_math500/MATERIALIZATION_MANIFEST.json`:

- training: exactly the 384 frozen MATH12K rows in source order;
- evaluation: exactly all 500 held-out MATH-500 rows in source order;
- training Arrow SHA-256:
  `359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8`;
- evaluation Arrow SHA-256:
  `2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7`;
- ordered training-row SHA-256:
  `051baa5571a1865518ef200c414178f1e50decea261bcd46ec0328e5837c0f36`;
- ordered evaluation-row SHA-256:
  `1576fd11df21dc705a7c85000f232031212225cd9c00520faa26f6bdfc751166`.
- normalized train/evaluation problem overlap: exactly zero.

MATH-500 is evaluation-only. No MATH-500 prompt, answer, reward, or evaluation
result may enter training, controller state, checkpoint selection, or
hyperparameter selection.

## Verified-answer canonical contract

Both arms use the repository's full `math_verify` reward and the neutral
`qwen_math` prompt. In the new `math_verified_answer` key mode, every active
reward-positive completion for one prompt maps to the same prompt-local key:

`math_verified_answer:correct`.

Every reward-zero completion maps to no canonical key. Thus:

- equivalent final-answer renderings cannot masquerade as distinct modes;
- free-form reasoning text is never claimed to be validated strategy identity;
- verified support per prompt is structurally at most one;
- verified-mass replay may anchor a model-generated correct solution;
- known-mode balance is structurally ineligible and must emit zero loss and
  zero controller observations;
- no reference solution is seeded into a bank.

The success-conditioned semantic tracker uses the same collapsed key for
reward-positive rows. Literal E58's structural unseen bucket and
projection-free semantic coefficient remain unchanged. This is deliberately
a stress test: on a single-answer task, open-set pressure may be neutral,
helpful through exploration before discovery, or harmful after the only
verified outcome is known.

## Arms and matched training

Use
`Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.

- `grpo`: matched Dr.GRPO with passive verified-answer tracking only.
- `verified_first_global_replay_canonical`: literal E58 with one
  checkpointed global verified bank per update, split verified-mass and
  known-mode-balance replay, and success-conditioned open-set pressure.

The full cohort uses seeds 43, 44, and 45; 16 samples per prompt; learning
rate `2e-7`; one PPO epoch; and 12 prompt passes (4,608 optimizer updates per
run). All three E58 coefficients remain projection-free. The balance
coefficient may retain its finite initialized value, but its observation
count and applied balance gradient must remain zero because support two is
impossible under this contract.

## Evaluation

Evaluate all 500 MATH-500 rows at passes 0, 2, 4, 6, 8, 10, and 12.
At each checkpoint report:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- response length and formatting rate;
- seed-level trajectories and the three-seed mean/range.

Sampled evaluation uses one fixed K=8 draw with seed 640100. `distinct@8`
is not a reasoning-route endpoint and must not be described as mode
coverage.

Training telemetry reports first verified discovery, verified-mass replay
activity, mass coefficient, open-set entropy and semantic coefficient,
balance eligibility/observations/loss, finite gradients, and resume state.

## Information firewall

Training and all controllers are forbidden from reading:

- MATH-500 evaluation output;
- gold support or a gold mode count;
- a desired entropy, accuracy, or discovery count;
- reference solutions as replay exemplars;
- an LLM judge or inferred reasoning-strategy label.

The fixed one-bank replay schedule and capacity 16 are compute bounds, not
semantic targets.

## Fail-closed smoke and advancement

Before the six-run cohort, run one seed-43 E58 smoke for 96 optimizer updates
on the first 96 frozen training rows. K-sampled evaluation is disabled for the
smoke; ordinary greedy initialization/terminal diagnostics may run.

The smoke passes only if:

1. at least one task-reward-positive completion is admitted under the single
   verified-answer key;
2. every admitted prompt has support exactly one;
3. verified-mass replay becomes active after discovery;
4. known-mode balance has zero eligible groups, zero loss, zero score
   gradient, and zero controller observations;
5. no coefficient projection is active;
6. no MATH-500 value enters training state;
7. all rewards, losses, coefficients, gradients, and checkpoints are finite;
8. the exact source, execution surface, data manifest, and job identity are
   frozen before release.

Only a passing smoke authorizes the matched three-seed, 12-pass cohort.
