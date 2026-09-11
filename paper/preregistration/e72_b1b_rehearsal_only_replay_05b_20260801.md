# E72 B1b: does ordinary verified replay preserve support by rehearsal alone?

**Status: FROZEN BEFORE SUBMISSION — 2026-08-01**

## Question

What happens if Dr.GRPO is simply trained with replay of previously verified
responses, at the same replay-token and gradient budget, but with no rarity
weighting and no uniform mode balancing?

This is the plainest reading of "ordinary verified-response replay," and the
manuscript lists it as untested. It is a live hypothesis rather than a
formality: rehearsal of correct responses could preserve useful modes on its own,
and if it does, the treatment's rarity and balance machinery is not what
retains support.

## Why the existing arms do not answer it

Three replay-bearing arms already exist and none is this one.

| arm | rarity credit | balance KL | replay gradient |
| --- | --- | --- | --- |
| matched Dr.GRPO (published control) | no | no | **zeroed** (`compute_only`) |
| B3a (replay-gradient ablation) | yes | yes | **zeroed** |
| B1a (discovery-credit ablation) | no | **yes** | live |
| **B1b (this arm)** | **no** | **no** | **live** |

B1a is the closest and is still not it: it retains
`split_mass_balance_per_rollout`, whose balance term is exactly the uniform mode
balancing the question excludes. Every replay variant in the repository before
this one used that objective. B1b is therefore the first arm in which a replay
gradient acts without any balancing pressure.

Stated the other way: **B1b is the published compute-matched control with its
replay gradient switched on, and nothing else changed.** That makes it the
minimal edit distance from the paper's own baseline to a rehearsal method.

## Arm

Runtime variant `verified_first_replay_rehearsal_only`. Replay objective
`verified_likelihood_per_rollout`, under which the learner forms the verified-
likelihood term alone and never constructs the balance loss, so balancing is
absent by construction rather than down-weighted. Rarity coefficient, novelty
credit, bank-entropy actor channel, token entropy, and the separate-advantage,
success-conditioned, and open-set adaptation switches are all off.

**Budget matching.** Bank capacity 16, one scheduled bank per optimizer update
in the same deterministic global round robin, the same $w_{\mathrm{rep}} =
(G-1)/G^2$ weighting, and the replay coefficient at the treatment's `.10`. The
arm therefore rehearses the same exemplars, on the same schedule, at the same
gradient weight as the treatment: it differs in what the replay loss *is*, not
in how much replay it gets.

## Cohort

Five domains x seeds 43--47 = 25 runs, trained from the pinned
Qwen2.5-0.5B-Instruct initialization on the common design: 12 passes, 4,608
optimizer updates, $G = 16$, $\beta_{KL} = 0$, rollout temperature 1, the fixed
128-prompt evaluation split, greedy decoding plus four deterministic
temperature-one $K = 8$ replicates. Per-domain data, template, response budget,
and evaluation draw seeds are inherited from the published runs. Each run is
pinned to the GPU model that trained its paired reference seed.

Comparators are the published seed-matched arms: matched Dr.GRPO and xGRPO,
plus B3a and B1a from this campaign. No coefficient is tuned; the replay
coefficient is the treatment's own.

## Analysis, fixed in advance

Primary quantity: `distinct@8` at terminal pass 12, per domain, five paired
seeds, no pooling. Reported with paired per-seed rows and a seed-level paired
bootstrap interval (10,000 resamples). Equivalence margin $\delta = 0.15$, as
in the B3a and confirmation protocols.

## Registered interpretations

- **P1.** B1b within $\delta$ of xGRPO in at least three domains -> rehearsal
  alone preserves support. The treatment's rarity and balance terms are then not
  what retains modes, Section 4.2 and Section 4.3 are both re-scoped, and the
  reviewer's hypothesis is confirmed against us.
- **P2.** B1b materially above matched Dr.GRPO but below xGRPO in most domains
  -> rehearsal contributes and balancing adds more; report the decomposition
  across all four arms, claiming neither component alone is sufficient.
- **P3.** B1b at or near matched Dr.GRPO -> rehearsal alone does not preserve
  support, and combined with B3a this isolates the balance term as the active
  ingredient. This is the outcome that would most strengthen the manuscript, and
  it is therefore the one to state most carefully.
- **P4.** B1b above xGRPO anywhere -> reported as-is.

Because B1a is already known to sit near xGRPO on its discovery sample, P3 would
imply the effect lives specifically in balancing rather than in replay as such.
That inference is registered here so it cannot be presented later as though it
had been anticipated all along.

## Failure policy

CUDA OOM, non-finite loss or coefficient, traceback, malformed checkpoint,
identity mismatch, missing seed, or failure to reach the terminal budget is a
run failure; a failed run resumes only from its own source-bound checkpoint. A
domain missing any of its five terminal seeds is unreported.

Arm-specific integrity, checked on every logged update rather than assumed:
`canonical_replay_compute_only` is 0 and the applied replay score gradient is
nonzero; the balance loss is never formed (`canonical_replay_balance_loss`
absent or exactly zero with no balance score gradient); and rarity and novelty
advantages are exactly zero. A run violating any of these is discarded, not
reinterpreted.
