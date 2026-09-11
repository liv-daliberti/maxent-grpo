# E48: 50-pass matched normalized canonical Haarnoja at 0.5B

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Motivation and scope

E48 is a user-directed long-horizon rerun of the two E46 ModeBench conditions
after inspecting the incomplete ten-pass dashboard. It does not extend, resume,
overwrite, or pool with E44 or E46. It launches a fresh contemporaneous
two-arm cohort so both the matched Dr.GRPO control and normalized canonical
Haarnoja treatment use the same current source, placement, wall-clock regime,
and fifty-pass budget.

The only intended scientific changes relative to the matched E46 comparison
are:

1. both arms are freshly and contemporaneously launched;
2. the training budget is fifty complete prompt-pool passes rather than ten;
3. ordinary Dr.GRPO passively records validator-bound discoveries and verified
   support without adding any reward, advantage, loss, or gradient term.

All conclusions are exploratory because the rerun was requested after viewing
the earlier learning curves.

## Frozen cohort

- Model:
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Tasks:
  - graph coloring: 192 training prompts and 96 neutral evaluation prompts;
  - Countdown easy3: 384 training prompts and 128 neutral evaluation prompts.
- Arms:
  - `grpo`: ordinary matched Dr.GRPO;
  - `online_canonical_haarnoja`: normalized online-canonical Haarnoja.
- Seeds: `43,44,45`.
- Group size: `G=16`.
- Budget: exactly 50 complete prompt-pool passes.
- Learning rate: `2e-7`.
- One PPO epoch, `beta=0`, maximum norm `1`.
- Rollout temperature `1`, top-p `1`, maximum response length `192`.
- Prompt template `qwen_boxed`, executable fast verifier, and
  `multi_answer` evaluation split.
- One node302 A100, 8 CPUs, and 64 GiB per job.
- Slurm wall-time limit: seven days.
- Checkpoint and resume interval: one complete prompt-pool pass.

Run prefixes:

- `gce48_normalized_canonical_haarnoja_05b_50ep_v1`;
- `cde48_normalized_canonical_haarnoja_05b_50ep_v1`.

## Treatment contract

The treatment is identical to E46:

- verified prompt-local canonical bank;
- novelty coefficient `beta=0.50`;
- pseudocount `1`;
- surprisal clip `5`;
- normalized controller sensor
  `rho_x = H(q_x) / log |B_x^+|` for `K_x >= 2`;
- normalized target `rho*=0.80`;
- initial coefficient `alpha_0=0.10`;
- projection `alpha in [0.10,0.50]`;
- Adam learning rate `0.003`, betas `(0.9,0.999)`, epsilon `1e-8`;
- EMA decay `0.90`;
- the controller update applies first to the next optimizer round;
- bank and controller states are independently checkpointed and restored.

Admission remains fail-closed for the treatment: the learner revalidates the
exact generated output and aborts if validator admission disagrees with the
actor reward. Countdown keys come from the executed exact AST and graph keys
from the constraint-checked coloring vector.

## Dr.GRPO discovery telemetry

The Dr.GRPO arm uses the same validator and prompt-local bank only as a passive
observer. Both exploration coefficients are exactly zero, the returned
exploration advantage is identically zero, and no discovery value enters task
reward centering or the policy loss.

Mandatory telemetry for both arms includes:

- cumulative verified discoveries;
- number of prompts with verified support;
- mean verified support per tracked prompt;
- per-round newly verified outcomes;
- validator-positive and canonicalizable fractions.

The passive bank is checkpointed so cumulative quantities survive ordinary
job recovery. Treatment-only controller and exploration-advantage diagnostics
remain absent or zero for Dr.GRPO as appropriate.

## Evaluation and interpretation

Evaluation remains the E46 protocol at initialization and every quarter pass:
greedy pass@1 plus four deterministic temperature-1, `K=8` draws with seeds
`440100--440103`. Raw outputs, rewards, and canonical keys are retained.

Primary reporting is the complete paired-seed trajectory through pass 50 for
pass@8, mean@8, valid coverage@8, distinct-correct@8, cumulative verified
discoveries, and mean verified support per prompt. Early collapse, recovery,
or null effects remain reportable. Thin curves are individual seeds; a heavy
mean is drawn only where all three seeds have reached the same evaluation
boundary.
