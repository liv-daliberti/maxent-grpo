# E73 Falcon3-1B cross-family replication of the five-domain design

**Status: FROZEN BEFORE LAUNCH (2026-07-31).**

## Question

Every headline cell in the manuscript is produced by one base model,
Qwen2.5-0.5B-Instruct. A reader may reasonably ask whether the reported
collapse of matched Dr.GRPO's metric band, and its reopening under xGRPO, is a
property of the objective or a property of that one model family's tokenizer
and instruction tuning. This experiment answers that directly by re-running the
entire preregistered five-domain design on a second, unrelated instruction-tuned
base model and asking whether the same qualitative result appears.

This is an external-validity replication. It is not a new method, a new
benchmark, or a tuning study, and no coefficient is re-selected for the new
model.

## Model

`tiiuae/Falcon3-1B-Instruct` at pinned revision
`28ba2251970a01dd1edc7ba7dad2eb71216ccfdf`.

Falcon3-1B-Instruct is chosen because it is independent of the Qwen line in
tokenizer, pretraining corpus, and instruction tuning, while remaining a small
instruction-tuned decoder in the same regime. It is a Llama-architecture model
with 18 layers, hidden size 2048, and a 131,072-token vocabulary, giving
approximately 1.67B parameters against Qwen2.5-0.5B-Instruct's 0.5B. The
parameter count is deliberately not matched: no claim in this experiment is a
comparison of the two base models to each other. Every comparison is
within-family, between the two arms.

## What is held fixed

The design is the manuscript's, unchanged:

- five domains: Graph coloring, Countdown, Python factors, MathIR action menu,
  and PantryPlan;
- two arms: matched Dr.GRPO (`grpo_compute_matched`) and xGRPO
  (`verified_first_global_replay_canonical`);
- seeds 43, 44, 45, 46, 47;
- 384 training prompts and a fixed 128-prompt evaluation split per domain,
  12 epochs, 4,608 optimizer updates per run;
- group size 16, `beta_KL = 0`, rollout temperature 1, learning rate 2e-7;
- evaluation by greedy decoding plus four deterministic replicates of `K = 8`
  temperature-one samples, four times per training pass, reported at the fixed
  checkpoints 0, 3, 6, 9, 12;
- every xGRPO coefficient at its manuscript value, with no per-model retuning.

Within every domain the two arms share data, model, optimizer, rollout count,
training budget, evaluation seed, checkpoint cadence, and recovery rules. Only
the xGRPO mechanism differs.

## What necessarily changes, and why it is not a confound

**Chat surface.** Falcon3-Instruct was instruction-tuned on
`<|system|>` / `<|user|>` / `<|assistant|>` role markers, not Qwen's ChatML
`<|im_start|>` / `<|im_end|>`. Each domain therefore uses the `falcon_*` twin of
its prompt contract. A twin shares its system instruction and its canonical
answer rewrite with the Qwen original verbatim and differs only in role markers;
the rendering reproduces the model's own published chat template byte for byte.
Prompt content is therefore held fixed and only the surface the base model was
trained to read is swapped. `tests/test_falcon_prompt_surface.py` enforces both
properties.

**Optimizer state placement.** Falcon3-1B does not fit a 24 GB A5000 alongside a
collocated vLLM engine with optimizer state resident on device, so Adam state and
the fp32 master copy are offloaded to host memory. This is applied identically to
both arms and changes where optimizer state lives, not the update it computes.

**Prompt token budgets are unchanged.** Falcon's role markers are ordinary
multi-token strings rather than single special tokens, which adds a small fixed
prompt overhead. Measured over the full train and evaluation pools, the longest
rendered prompt per domain is 213 (Graph), 246 (Countdown), 169 (Python), 226
(MathIR), and 609 (PantryPlan) tokens, all inside the manuscript's existing
budgets, so no *prompt* limit is relaxed for this cohort.

**Amendment 1 (2026-07-31, before any outcome was read): Python response
budget.** The paragraph above verified prompt lengths. The binding constraint
turned out to be the *response* budget, which is a property of how verbosely a
model answers rather than of the data. Measured on the first ~30 optimizer steps
of this cohort against the corresponding Qwen runs:

| Domain | Falcon mean response | Falcon at 192-token cap | Qwen mean | Qwen at cap |
|---|---|---|---|---|
| Graph coloring | 20.9 | 6% | 4.3 | 0% |
| Countdown | 24.0 | 2% | 8.8 | 0% |
| Python factors | 136.4 | 44% | 9.8 | 0% |

Falcon3-1B is uniformly more verbose than Qwen2.5-0.5B. On Python that makes the
manuscript's 192-token response budget bind on 44% of rollouts, truncating them
before any parseable answer is emitted, whereas the same budget never bound the
Qwen cohort (20 of 4,609 responses). A cell measured through a budget that
truncates almost half its rollouts reports the interaction of the model's
verbosity with the instrument, not the behaviour of the objective, and the
design's intent is that the response budget not be the binding constraint for
either family.

The Python domain's response budget is therefore raised to 512 generate and 512
evaluate tokens with `max_model_len` 768, applied identically to both arms and
to all five seeds, and its ten runs are relaunched from step zero on the same
frozen source and execution snapshots. Graph coloring (6%) and Countdown (2%)
are left at the manuscript values: they are close enough to the non-binding
regime that changing them would add deviation without removing a confound.
MathIR and PantryPlan are unaffected, the latter because its canonical
fixed-shape sampler emits exactly six action tokens by construction.

This amendment was written and applied before any evaluation checkpoint of this
cohort was read; the Python arms were at optimizer step ~30 of 4,608 and their
mean reward was 0.0000 in both arms, the same value the Qwen Python cohort
showed at the same point (0.0016). No outcome ordering between the arms was
known, and none of the compared quantities was used to choose the new budget:
it was chosen to make truncation negligible, not to move a result. Because the
change alters response length, Python's Falcon rows are not comparable to a
192-token Falcon Python run, and the reported cohort uses the 512-token runs
throughout.

**Canonical action geometry is unchanged.** Every canonical action string
resolves to exactly one round-tripping token under Falcon's tokenizer, and the
three canonical tasks resolve to the same horizon, sequence count, and maximum
sequence entropy as under Qwen: Graph 3/27, Countdown 3/108, PantryPlan 6/64.
The objective is therefore defined over the same action geometry in both
families.

## Placement

All jobs run on the `cs` partition under account `allcs` on A5000 nodes. The
`mltheory` partition is deliberately excluded so this cohort cannot contend with
the concurrently running decoding-control and replay-ablation campaigns.

## Prediction

The registered prediction is qualitative and directional, stated before launch:

1. Matched Dr.GRPO's four metrics will again coincide at the terminal epoch, with
   modes-per-success near 1, in at least four of the five domains.
2. xGRPO will raise `distinct@8` over matched Dr.GRPO at the terminal epoch in at
   least four of the five domains.
3. xGRPO will not reduce `pass@8` below the matched baseline in any domain.

Absolute values are **not** predicted to reproduce the Qwen numbers, and no
threshold on their agreement is registered. A different base model has different
task competence; the claim under test is the direction and the presence of the
band collapse, not the magnitude.

## Analysis and fail-closed rules

The headline row is the deepest checkpoint reached by all five seeds of both arms
in each domain, matching the manuscript's reporting rule. Domains are reported
individually and are not pooled.

The following are fixed now and may not be chosen after seeing outcomes: seed
replacement, early stopping, best-checkpoint selection, result-dependent
extension, evaluation feedback, and post-hoc coefficient changes. A run that
fails is requeued from its last checkpoint under the existing watchdog rules or
is reported as incomplete; it is not replaced by a fresh seed. If a domain
cannot reach a common terminal epoch across all ten of its runs, that domain is
reported at its deepest common checkpoint and the shortfall is stated.

A negative or mixed result is reportable and will be reported. If the manuscript's
effect does not reproduce on Falcon3-1B, the correct conclusion is that the
result as stated is contingent on the base model, and the manuscript's claim will
be narrowed accordingly rather than this cohort being excluded.

## Companion measurements

Two manuscript appendices are replicated on the same cohort once its terminal
checkpoints exist:

- the decoding control, re-measuring all 50 terminal Falcon checkpoints at
  `T` in {0.5, 0.7, 1.0, 1.3, 1.6, 2.0} with no rollout, optimizer step, or
  export, giving 300 evaluation-only cells; and
- the MathIR terminal causal comparison, which changes only separated replay
  support on otherwise identical plumbing.

Both inherit this protocol's seeds, splits, evaluation path, and reporting rule.
