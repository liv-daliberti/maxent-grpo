# ICLR readiness audit

Evidence refresh: 2026-08-26. The canonical 750-cell registry and DAPO state
are audited through 12:06 EDT. This is a working decision document,
not manuscript prose.
It separates already supported claims from experiments that are merely running
or proposed.

## Bottom line

The five-domain, five-seed Falcon3-1B block is the primary cross-family
result: verified replay improves terminal solution breadth in all 25 paired
cells. The completed Qwen2.5-0.5B and Qwen2.5-3B experiments supply two more
five-domain uncertainty results, each with five terminal paired seeds per
domain. The
manuscript is now organized around the Falcon result while retaining the Qwen
inferential core, titled for Dr.GRPO, and fits the nine-page initial
submission limit. The direct-comparator evidence is now substantial on its
terminal surface: UCPO has all ten smaller-model five-seed blocks, and sparse
RLEP-Dr has nine complete five-seed blocks plus Falcon Python at two seeds.
The new Qwen RLEP-Dr MathIR block has a positive pass interval and neutral
adjusted breadth; the new Countdown block remains inconclusive. The complete
Qwen Countdown UCPO block retains positive pass and adjusted-breadth intervals. The full three-scale direct-comparator matrix is
registered, but many extension cells remain nonterminal. Custom DAPO-R3 is
retired and excluded because its sampling algorithm is not the published
multi-prompt DAPO sampler; the official-verl R4 smokes failed before training
on a Ray socket-path limit, and R4-R1 fixed Ray but failed before rollout on a
vLLM scheduler-cap invariant. The R4-R2 gate passed and released all 50 science
cells; four Qwen Graph cells are training-terminal, 46 are nonterminal, and
none has failed. Their exact upstream `acc@1` diagnostics are not the
paper's standardized pass@8/breadth endpoints. The corrected
semantic-MaxEnt mechanism gate is complete. E112-R1 is 53/75 operationally and
now has an author-requested 49-pair smaller-model terminal disclosure. Remaining
risks are incomplete Qwen2.5-3B coverage, one excluded historical comparator,
and interrupted confirmatory outcome blindness for E112-R1.

Treat verified mode replay as the paper's contribution. Treat fixed semantic
MaxEnt as a completed, domain-heterogeneous factorial mechanism ablation. Treat
the legacy adaptive MaxEnt outcomes as appendix evidence with an explicit
mechanism-gate failure. Keep the repaired estimators scientifically separate:
the v6 diagnostic is complete, but E105 was retired outcome-blind because v6 is
structurally inactive on singleton verified support. The v7 E111 mechanism gate
is 15/15 terminal and passes its frozen audit; E109 is a valid 15/15
repaired-Python comparator. E112-R1 is 53/75 terminal and 22 nonterminal at the
operational refresh. User-requested looks were frozen at 14, 33, and 50 terminal
cells; the public result reports 49 integrity-valid smaller-model pairs after
excluding one conflicting comparator. It is exploratory, excludes Qwen2.5-3B,
and cannot affect execution. No confirmatory component-isolated claim is made.

## What the completed core actually says

All entries below are replay minus matched control at pass 8 for
Qwen2.5-0.5B, with five paired seeds. “Excess modes” is
distinct-mode@8 minus pass@8; its change removes the one-for-one increase in
distinct correct modes that can arise solely from higher correctness.

| Domain | Delta distinct | Paired 95% interval | Delta pass | Delta excess modes |
|---|---:|---:|---:|---:|
| Graph Coloring | +2.112 | [2.001, 2.224] | +0.646 | +1.467 |
| Countdown | +1.150 | [0.959, 1.341] | +0.191 | +0.959 |
| Python Factors | +0.399 | [-0.101, 0.899] | +0.356 | +0.043 |
| MathIR | +0.315 | [0.220, 0.411] | +0.285 | +0.030 |
| PantryPlan | +1.018 | [0.781, 1.254] | +0.207 | +0.811 |

This supports two deliberately different conclusions:

1. Replay preserves breadth beyond accuracy on Graph Coloring, Countdown, and
   PantryPlan.
2. On Python Factors and MathIR, most of the raw distinct-mode improvement is an
   accuracy rescue. Those domains are positive outcomes, but not strong evidence
   for independent diversity retention.

The manuscript should make that distinction itself instead of inviting a
reviewer to discover it.

The newly completed Qwen2.5-3B block repeats that domain split. Mean paired
pass@8 effects are +.154, +.137, +.590, +.409, and +.106 on Graph, Countdown,
Python, MathIR, and Pantry; all five paired 95% intervals exclude zero.
Correctness-adjusted breadth is +.423 [.331, .515], +.197 [.150, .244],
+.194 [-.106, .494], +.019 [-.002, .041], and +.567 [.366, .768]. Thus
Graph, Countdown, and Pantry retain breadth beyond accuracy, while Python and
MathIR are chiefly accuracy rescues. These remain five separate domain
estimates, not a pooled scale effect.

## Canonical experiment matrix

`ops/exp_scaling/paper_matrix.py` is the source of truth for the declared
10-method × 3-scale × 5-domain × 5-seed design. Repairs and historical cohorts
map into these cells rather than becoming extra methods.

Current regenerated snapshot: 750 target cells, 560 registered, 505 terminal,
and 190 not registered. Figures may show any available scientific checkpoint
with its exact seed count; display is never gated on five seeds.

| Method | Target | Registered | Terminal | Active | Blocked | Failed | Partial | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 75 | 75 | 73 | 2 | 0 | 0 | 0 | 0 |
| Dr.GRPO | 75 | 75 | 75 | 0 | 0 | 0 | 0 | 0 |
| UCPO | 75 | 75 | 50 | 25 | 0 | 0 | 0 | 0 |
| RLEP-Dr | 75 | 75 | 47 | 25 | 3 | 0 | 0 | 0 |
| ReplayDr.GRPO | 75 | 75 | 75 | 0 | 0 | 0 | 0 | 0 |
| Adaptive ReplayDr.GRPO | 75 | 25 | 25 | 0 | 0 | 0 | 0 | 50 |
| Semantic MaxEnt | 75 | 50 | 50 | 0 | 0 | 0 | 0 | 25 |
| Adaptive Semantic MaxEnt | 75 | 0 | 0 | 0 | 0 | 0 | 0 | 75 |
| ReplayDr.GRPO + Semantic MaxEnt | 75 | 55 | 55 | 0 | 0 | 0 | 0 | 20 |
| Adaptive Semantic MaxEnt + ReplayDr.GRPO | 75 | 55 | 55 | 0 | 0 | 0 | 0 | 20 |

Active combines scheduler-running and scheduler-pending cells. Blocked cells
cannot satisfy their frozen gate without a separate repair. Failed means a
released scientific job ended before its terminal endpoint; partial means
recorded work exists without the registered terminal endpoint.

Refresh this snapshot rather than hand-editing counts:

```bash
python ops/exp_scaling/paper_matrix.py --view all --markdown
```

`paper_launch_plan.py` partitions unfinished registered cells;
`paper_comparator_batches.py` partitions missing GRPO, UCPO, and RLEP-Dr cells.

## Missing experiments, in reviewer-value order

### 1. Plain-GRPO bridge: 73 terminal, full matrix registered

The ordinary-GRPO bridge has complete five-seed blocks on all five Qwen2.5-0.5B
and all five Falcon3-1B domains. Qwen2.5-3B Graph, Countdown, and Python are
complete at five seeds; MathIR and Pantry are exact four-seed prefixes. Report intervals only
for complete blocks; do not average the prefixes or infer a model-size trend.

### 2. Closest diversity-preserving policy-optimization baseline

UCPO is now registered across all three models, five domains, and five seeds.
Fifty cells are terminal: all five Qwen2.5-0.5B and all five Falcon domains are
complete at paired $n=5$. Qwen Countdown has a positive adjusted-breadth
interval, while Qwen MathIR is neutral. The direct matched-baseline forest gives
intervals only to the balanced five-seed blocks. The remaining 25 registered
cells are active rather than silently absent.

### 3. Isolate mode balancing from generic experience replay

Compare x-Mode replay against a verified-success replay buffer without one-slot
canonical keys or uniform-over-mode sampling, such as FIFO or
frequency-proportional success replay. This is the cleanest test that executable
canonical identity and balanced memory—not merely extra supervised updates on
successful samples—cause the effect. Use two or three representative domains
with five paired seeds.

Status: the frequency-preserving RLEP-Dr collections completed, but their
audited pools failed the frozen requirement of two verified trajectories for
every prompt. That E98 feasibility failure remains frozen. The separately
preregistered E98-R1 repair reuses the immutable pools, applies replay only on
eligible prompts, and leaves other updates unchanged. Across E98-R1/E100/E116,
all five Qwen domains and Falcon Graph/Countdown/MathIR/Pantry are terminal at
paired $n=5$, while Falcon Python is terminal at $n=2$. E116 registers the
remaining Qwen-3B matrix cells; three Falcon Python cells remain pool-gated. The trajectory grid and direct-comparator
forest disclose every exact available block.

DAPO remains a protocol-level comparator rather than outcome evidence. Static
audit established that E113-R3's one-prompt retry loop is materially different
from the published multi-prompt filter-and-buffer sampler, so every R3 cell is
retired from named-DAPO efficacy regardless of its operational state. E113-R4
prospectively pins the official `verl-recipe/dapo` implementation, container,
controls, seeds, query accounting, and endpoints. Its first two smokes failed
before training on the Ray Unix-domain socket-path limit and their zero-runtime
science dependents were canceled. The R4-R1 amendment changed only the
node-local Ray temporary path and reached vLLM construction, where both
replacement smokes failed because the 448-token aggregate scheduler cap was
below the pinned 1,024-sequence limit. Their 50 dependents were canceled at
zero runtime. R4-R2 prospectively raises only the aggregate capacity and
validates inside the pinned container. Both fresh smokes passed and released
the 50 science cells. Qwen Graph seeds 43--46 completed all 24 accepted updates;
the other 46 cells are nonterminal and none has failed. The four exact upstream
`acc@1` diagnostics are retained per cell without a mean or interval. Because
no standardized pass@8 or executed-mode breadth evaluation exists yet, there
is no matched DAPO efficacy point.

### 4. Measure the proposed mechanism directly from existing logs

The terminal E78 logs now support and the appendix reports:

- bank occupancy and capacity saturation over all 76,800 replay updates;
- realized replay dose against the registered charged token budget;
- descriptive paired optimizer-update timing against the exact-zero control;
- the preregistered trajectory AUC estimand alongside terminal values.

The run directories do not retain bank-exemplar identity snapshots. Survival
of individual banked modes, later evaluation re-observation, and identity-level
coverage gain versus bank coverage therefore cannot be reconstructed from this
cohort. Treat those as a logging requirement for a future run, not as an
unreported analysis of existing data.

### 5. Add a small sensitivity surface

On two representative domains, vary replay-bank capacity and replay weight with
three seeds per setting. The bank-normalized arm partially addresses dose but
not capacity. This should establish a stable region, not optimize a coefficient.

### 6. Decide the external-validity story

ModeBench is a synthetic diagnostic suite. Either add one natural,
execution-verifiable task with meaningful canonical modes or explicitly frame
ModeBench as a diagnostic instrument and narrow generalization claims.

### 7. Add scale only after controls

A representative 7B slice would matter because the closest work reports much
larger models. It is lower priority than plain-GRPO, closest-method, and generic
replay controls. Do not spend the remaining budget on more adaptive-controller
families before these comparisons exist.

## Final paper organization: nine-page target

| Section | Target | Content |
|---|---:|---|
| Introduction | 1.0 page | Failure, executable identity, result, contributions |
| Measurement + ModeBench | 1.25 | Definitions, five domains, why pass and breadth differ |
| Collapse + x-Mode replay | 1.25 | Mean-flow intuition and algorithm; proofs in appendix |
| Experimental design | 0.5 | Paired seeds, budgets, estimands, uncertainty |
| Terminal results | 2.25 | Inference frontier, cross-family effects, accuracy-adjusted breadth |
| Mechanism + boundaries | 0.75 | Bank telemetry, fixed-MaxEnt factorial, adaptive evidence boundary |
| Related work + conclusion | 0.75 | Direct contrasts and scoped takeaway |
| Figures/tables/spacing reserve | 1.25 | Prevent accidental page-ten spill |

Move the semantic-MaxEnt derivation, full raw tables, adaptive dosing history,
evidence-policy discussion, and most training traces to the appendix. Keep
proofs, exact benchmark contracts, terminal paired-seed tables, and mechanism
audits. Remove historical experiment
archaeology that a reviewer is not required to read.

Implemented in the current draft: the end-of-training pass@8--distinct@8
frontier replaces the interim dashboard as the main empirical figure. Matched
distinct@8, pass@8, and mean@8 training grids, semantic MaxEnt, and
scale-progress detail move to the appendix; the completed fixed-MaxEnt factorial
is generated from all 100 cells with paired uncertainty. The 15-panel wall is
excluded; terminal Falcon and fixed-factorial trajectories are separate
appendix figures, while UCPO and RLEP-Dr occupy a separate matched-baseline
forest and direct-comparator trajectory grid.
Supporting historical controls remain labeled in the appendix rather than mixed
with final comparison families. The appendix now also exposes the balanced
ten-row Qwen/Falcon endpoint forest, the terminal adaptive-semantic slices,
the registered E88 dose-gate failure, and all 25 terminal Adaptive
ReplayDr.GRPO cells as an exact-paired terminal figure. The counted main text ends
with the conclusion on page 9; the page-limit-exempt AI, ethics, and
reproducibility statements begin page 10 immediately before the references.

## Final figure plan

1. Keep the opening model-backed collapse/replay trajectory. It now uses a
   completed registered pair and makes the failure concrete.
2. Keep a compact benchmark/examples panel; place the replay mechanism in the appendix.
3. Lead Results with the five-domain, five-seed terminal pass@8--distinct@8
   frontier for Qwen2.5-0.5B and Falcon3-1B. Put the matched distinct@8,
   pass@8, and mean@8 training grids in the appendix. The 15-panel wall stays
   excluded.
4. Use the invariant 3x5 trajectory grid everywhere: row 1 Qwen2.5-0.5B, row 2
   Falcon3-1B, row 3 Qwen2.5-3B; Graph/Countdown/Python/MathIR/Pantry columns.
   Cells are blank only when the requested method has no sampled checkpoint, never dropped or overlaid.
5. Keep the endpoint forest in the same three physical model rows. Populate every
   available method--domain block, label its exact seed count, and leave only
   unsupported cells blank.
6. Keep the completed fixed-MaxEnt factorial separate in the appendix. Apply
   the same 3x5 contract to fixed-MaxEnt, adaptive semantic,
   UCPO, and replay-dose trajectories. Keep each labelled by its realized
   terminal or progress status.
7. Use `FIGURE_MANIFEST.md` as the placement contract.

Figure rules:

- One comparison per visual: control versus x-Mode in the core figures.
- Use the shared method palette, typography, line weights, and redundant line
  styles from `ops/paper_method_style.py`.
- Show raw paired-seed effects wherever space permits.
- Put sample size beside every effect and distinguish terminal from partial
  cells.
- Do not use raw distinct modes alone when pass rate changes substantially.
- Do not include cohort IDs or scheduler state. Incomplete evidence must say
  “progress” or “interim”; terminal figures must not.

## Claim/evidence gate

| Claim | Current status | Submission gate |
|---|---|---|
| Dr.GRPO can lose execution-defined modes | Supported | Show paired trajectories and terminal effects |
| Verified mode replay prevents loss | Strong, domain-specific evidence at 0.5B, Falcon3-1B, and Qwen2.5-3B; five terminal domains and five paired seeds at each scale | Report scales and domains separately; do not infer a pooled model-size trend |
| Gain is not only accuracy | Mixed by domain | Report excess modes and inference frontiers |
| Canonical balancing matters | Direct sparse RLEP-Dr evidence has nine complete blocks plus Falcon Python `n=2`; Qwen MathIR has a positive pass interval but neutral adjusted breadth, while Countdown is inconclusive | Report exact domain results and the strict-parent feasibility failure |
| Better than current diversity methods | UCPO has ten paired `n=5` smaller-model blocks; Qwen Countdown's adjusted-breadth interval is positive, while Qwen MathIR is neutral; 50/75 cells are terminal | Report the domain-specific matched forest; do not claim incomplete extensions |
| Applies to plain GRPO | All 15 model--domain blocks are complete at paired `n=5`; 75/75 cells are terminal and all five Qwen2.5-3B effects are inconclusive | Report domain-specific results; avoid a broad cross-model claim |
| Mechanism is replayed mode survival | Plausible | Add bank/survival telemetry |
| Semantic MaxEnt is broadly helpful | Not supported | Present as mixed factorial ablation |

## Stop-doing list

Do not resume broad coefficient search, revive E105, or add more legacy
adaptive-controller variants. E109 and E111 are closed. Continue the released
E112-R1 Qwen2.5-3B cohort without using the exploratory smaller-model outcomes
for selection. Keep custom DAPO-R3 retired; official R4 requires standardized
terminal evidence before any efficacy claim. Finish registered comparator
extensions without converting partial trajectories into endpoint evidence.

## Submission checklist

- [x] Counted main text ends with the conclusion on page 9; exempt policy statements begin page 10 before references.
- [x] Terminal paired results for every main-text cell.
- [x] Paired uncertainty, raw seed effects, and AUC are generated for the core and fixed-MaxEnt factorial.
- [ ] Add measured throughput overhead.
- [x] Explicit separation of correctness and breadth.
- [x] No dated-interim language or partial scheduler state in the main text.
- [x] Closest contemporary work is cited and addressed by the UCPO/RLEP direct-comparator panel.
- [x] Reproduction commands cover every main-text figure and its JSON provenance.
- [x] Main claims match the models, algorithms, and domains actually tested.
- [x] The required AI Use Statement appears before the references and follows the ICLR 2027 author policy.
