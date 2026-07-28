# E50: Uncapped-alpha normalized canonical Haarnoja at 0.5B

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Motivation and scope

E50 is the user-directed from-scratch replacement for the incomplete E48
ModeBench cohort after inspecting its live trajectories. E48 remains immutable
historical evidence and is neither resumed nor pooled with E50. E50 freshly
launches matched Dr.GRPO and normalized online-canonical Haarnoja for Countdown
and graph coloring at seeds 43/44/45.

The sole scientific change from E48 is removal of Haarnoja alpha's upper
projection. The coefficient starts at `0.10`, retains the lower projection
`alpha >= 0.10`, and has no configured upper bound (`alpha_max = +inf`).
Therefore alpha may grow beyond E48's former `0.50` ceiling whenever the
normalized entropy sensor remains below target. This removes an upper bound on
the entropy pressure; the normalized entropy itself remains mathematically
bounded by its finite discovered support.

The matched Dr.GRPO arm is unchanged and remains contemporaneous so placement,
source, data, evaluation, and elapsed cluster conditions are controlled.

## Frozen cohort

- Model:
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Tasks:
  - graph coloring: 192 training prompts and 96 neutral evaluation prompts;
  - Countdown easy3: 384 training prompts and 128 neutral evaluation prompts.
- Arms:
  - `grpo`: ordinary matched Dr.GRPO;
  - `online_canonical_haarnoja`: normalized online-canonical Haarnoja with
    uncapped upper alpha.
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

- `gce50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1`;
- `cde50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1`.

## Treatment contract

The treatment retains E48's complete objective and controller contract except
for its upper alpha projection:

- verified prompt-local canonical bank;
- novelty coefficient `beta=0.50`;
- pseudocount `1`;
- surprisal clip `5`;
- normalized controller sensor
  `rho_x = H(q_x) / log |B_x^+|` for `K_x >= 2`;
- normalized target `rho*=0.80`;
- initial and minimum coefficient `alpha_0=alpha_min=0.10`;
- no maximum coefficient and no upper projection: `alpha_max=+inf`;
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

Mandatory telemetry for both arms includes cumulative verified discoveries,
the number of prompts with verified support, mean verified support per tracked
prompt, per-round newly verified outcomes, and validator-positive and
canonicalizable fractions. The passive bank is checkpointed so cumulative
quantities survive ordinary job recovery.

## Evaluation and interpretation

Evaluation remains the E48 protocol at initialization and every quarter pass:
greedy pass@1 plus four deterministic temperature-1, `K=8` draws with seeds
`440100--440103`. Raw outputs, rewards, and canonical keys are retained.

Primary reporting is the complete paired-seed trajectory through pass 50 for
pass@8, mean@8, valid coverage@8, distinct-correct@8, cumulative verified
discoveries, mean verified support per prompt, normalized entropy, and alpha.
Alpha is plotted with data-dependent upper limits so growth above `0.50` is
not visually clipped. Early collapse, runaway pressure, recovery, or null
effects remain reportable.
