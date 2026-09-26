# Three-experiment paper and execution plan

Status: active paper contract, September 2 evidence freeze.

The paper is organized around ModeBench as the measurement contribution and
ReplayMaxRL as the algorithmic contribution. Only three experiments belong in
the main narrative. This plan governs what is reported now and what should be
finished during the remaining two-week window.

## Evidence rules

- Estimate effects within model, domain, level, and seed; never pool those axes.
- The primary endpoints are `pass@8` and `distinct@8`; report
  `distinct@8 - pass@8` when separating breadth from correctness.
- A complete efficacy block requires all five registered paired terminal seeds
  at step 3,072 and four fixed evaluation draws.
- A smaller terminal seed intersection may be displayed as descriptive
  progress with exact `n`, but receives no five-seed interval or completed claim.
- Level 1 and Level 2 use disjoint prompts. Cross-level comparisons are
  descriptive comparisons of within-level effects, not paired-prompt tests.
- Semantic-MaxEnt is frozen: one no-replay row appears as a Figure 4
  benchmark, while its definition and one supporting comparison remain in the
  appendix. Do not launch further Semantic-MaxEnt, DAPO, adaptive-dose, or
  open-bank branches for this paper.

## Experiment 1 — Does verifier-only RL lose support, and does replay retain it?

Design: Level 1, Dr.GRPO versus ReplayDr.GRPO, three scales, five domains, five
paired seeds. Supporting alternatives are plain GRPO, UCPO, and sparse RLEP-Dr.

Current evidence: all 15 primary model--domain blocks are complete (75 paired
terminal comparisons). ReplayDr.GRPO has higher raw terminal `distinct@8` in
all 75 pairs. Correctness-adjusted effects are strongest on Graph, Countdown,
and Pantry; Python and MathIR are more often correctness rescues.

Paper treatment: Figure 4A is the compact cross-scale endpoint-effect grid.
Figure 4B holds Qwen-0.5B fixed and places ReplayDr.GRPO beside no-replay
MaxRL, UCPO, sparse RLEP-Dr, fixed no-replay Semantic-MaxEnt, and ordinary
GRPO. The main text closes with “What Experiment 1 establishes.” Seed-level
direct-alternative forests, trajectories, and the detailed semantic analysis
live in the appendix.

Remaining work: no new primary training. Freeze the 75-pair result, retain the
trajectory-AUC sensitivity analysis, and regenerate only if a source hash or
integrity check changes.

## Experiment 2 — Does replay add value beyond MaxRL?

Design: the Level-1 2x2 factorial crosses fresh objective (Dr.GRPO or binary
MaxRL) with canonical verified replay (off or on). The new scale extension is
150 MaxRL/ReplayMaxRL runs: three scales x five domains x two arms x five seeds.
Completed Dr.GRPO/ReplayDr.GRPO cells supply the other factorial axis.

Current evidence:

| Scale | Complete five-seed blocks | Exact terminal matched prefixes |
|---|---:|---|
| Qwen2.5-0.5B | 5/5 | all domains `n=5` |
| Falcon3-1B | 5/5 | all domains `n=5` |
| Qwen2.5-3B | 0/5 | Python `n=2`; no other terminal matched pair |

At both completed scales, ReplayMaxRL exceeds MaxRL on the within-seed
cross-domain averages for terminal `pass@8` and `distinct@8`. At Qwen it is
positive in every domain on both endpoints; at Falcon it is positive in every
domain on `pass@8` and four of five on raw `distinct@8`. This licenses a
two-scale nonredundancy claim, not a monotonic scaling law.

Paper treatment: Figure 5 contains only the within-seed cross-domain averages
for Qwen and Falcon. The appendix contains the complete per-domain panels for
both models and is linked from the main caption and findings. Qwen-3B is
retained in the machine-readable progress record but omitted from both figures
until all five blocks are terminal.

Remaining work, in order:

1. Repair/resume Qwen2.5-3B until every domain has five matched terminal pairs.
2. Freeze a three-scale result record containing all four arms, both replay
   contrasts, MaxRL minus Dr.GRPO, and the replay-by-objective interaction.
3. Add pass-6 and trajectory-AUC sensitivity analyses without selecting a
   favorable checkpoint.

## Experiment 3 — Does the factorial transfer to a harder matched level?

Design: ModeBench Level 2 preserves the verifier, canonical key, split sizes,
and valid-support histogram while changing problem structure. The treatment
experiment repeats all four arms for Qwen2.5-0.5B across five domains and five
seeds (100 runs).

Current evidence: the frozen admission study is complete. Level 2 is harder but
solvable in all five domains under both small model families. Graph has the
complete exact four-arm terminal intersection for seeds 43--47 (`n=5`). Both
replay effects are positive, and every replay seed exceeds every non-replay
seed on both endpoints. The other four domains remain incomplete.

Paper treatment: Figure 6A reports admission and Figure 6B reports the complete
Graph `n=5` factorial. The section foregrounds that difficulty turns support
preservation into a correctness bottleneck and that replay matters more than
the choice of fresh-rollout objective, while reserving cross-domain Level-2
claims until the remaining blocks finish.

Remaining work, in order:

1. Finish the other four Level-2 four-arm blocks at the registered terminal endpoint.
2. Build a fail-closed Level-2 record validating arm, seed, identity, checkpoint,
   and four-draw completeness.
3. Estimate ReplayMaxRL minus MaxRL, ReplayDr.GRPO minus Dr.GRPO, MaxRL minus
   Dr.GRPO, and the interaction separately per domain.
4. Compare Level-1 and Level-2 effect patterns with independent intervals; do
   not draw paired prompt lines or claim a paired cross-level test.
5. Add trajectories/AUC to the appendix only after the endpoint analysis is
   frozen.

## Two-week priority

| Priority | Work | Why it changes the paper |
|---|---|---|
| P0 | Complete Qwen-3B MaxRL/ReplayMaxRL pairs | Extends Experiment 2 from two completed scales to the intended three-scale claim |
| P0 | Complete the five Level-2 four-arm blocks | Converts Experiment 3 from benchmark admission to the paper's transfer test |
| P1 | Build frozen all-scale and Level-2 factorial records/figures | Prevents hand-copied or mixed-depth results |
| P1 | Run terminal, pass-6, and trajectory-AUC analyses | Tests endpoint dependence without adding new treatment branches |
| P2 | Report discovery/retention telemetry only if identity logging is uniform | Adds mechanism evidence without changing the causal design |
| Stop | New Semantic-MaxEnt, DAPO, adaptive-dose, or open-bank variants | They dilute the ModeBench + ReplayMaxRL story and cannot close the two primary gaps in time |
