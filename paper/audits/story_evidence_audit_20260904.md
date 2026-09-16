# Evidence and story audit — September 4, 2026

This report reviews the empirical argument shared by [the full paper](../main.tex) and [the MATH-AI workshop version](../mathai2026/main.tex), including their appendices. It records findings, evidence boundaries, and the reason for the September 4 source-integrity correction. It is an audit record written after outcomes were available. It does not amend the original experimental registrations, restore outcome blindness, or authorize selecting among conflicting evaluations.

The argument remains well supported: binary-reward training can lose sampled verified alternatives; canonical-key replay improves held-out correctness and sampled breadth; replay also adds value to MaxRL; and a completed harder-Graph comparison extends that finding beyond Level 1. The strongest defensible account combines those observations with a conditional explanation of why replay can preserve probability on already discovered outputs. It does not require a general scaling law, a claim of full support recovery, or a claim that increased breadth causes the correctness improvement.

## Audit records and scope

- [Terminal integrity audit, JSON](primary_terminal_integrity_20260904.json) and [brief report](primary_terminal_integrity_20260904.md): 445 unique run directories, 484 source logs, and 15,873,336,994 bytes read and hashed. The recorded audit time is September 4 at 20:48:16 UTC. Scope covers the E78/E79/E80-R1 core, E118, E119, E120-R1, and the E120-R1 historical comparators. The JSON retains source-ledger hashes, source-file hashes, exact duplicate row numbers, payload hashes, and endpoint vectors.
- [Initial-checkpoint and AUC source audit](initial_auc_source_integrity_20260904.md), with its [machine record](initial_auc_source_integrity_20260904.json): the companion audit covers all 150 core initial checkpoints, every registered half-pass checkpoint for the 50 Qwen2.5-0.5B core runs, and the five Qwen RLEP Pantry trajectories. Its authoritative-source checks also distinguish superseded jobs from registered replacements.
- [Supplemental comparator source audit](comparator_source_integrity_20260904.md), with its [machine record](comparator_source_integrity_20260904.json): 281 admitted files and 356 requested checkpoints, including all 75 plain-GRPO initial/terminal pairs and the reported UCPO, RLEP, fixed-semantic, and additional E112 comparator sources. Every requested checkpoint passes; there are no additional duplicate or source-binding conflicts.
- [Final paper validation record](paper_revision_validation_20260904.md) records the compiled artifacts, focused tests, and numerical proof checks.
- Numerical evidence was checked against the existing figure/result JSONs and their readers. This report reuses the completed source scans and adds no new training, evaluation, figure generation, or checkpoint selection.

The original figures used the September 4 analysis snapshot. Later progress reported below is a separately labeled observation during this audit. Active logs can continue to grow; recorded hashes identify the bytes actually read rather than promising that a live path remains unchanged.

## Numerical source index

- Baseline loss and model-size percentages: [baseline precheck JSON](../figures/baseline_collapse_precheck.json) and [table](../results/baseline_collapse_precheck_table_body.tex).
- Core terminal comparisons and exclusions: [core endpoints](../results/core_terminal_endpoints.json); [cross-scale endpoint effects](../figures/cross_scale_terminal_endpoint_effects.json) records the Countdown extra-mode interval.
- Absolute reference comparisons: [absolute support references](../results/absolute_support_references.json).
- MaxRL comparisons and track identities: [E118 figure record](../figures/e118_all_scale_factorial_progress.json).
- Harder Graph and partial Level-2 coverage: [level admission and treatment record](../figures/modebench_level_admission.json).
- Uniform versus frequency weighting: [E120 frozen results](../results/e120_frequency_progress.json). The separate latest counts came from read-only source-log inspection; they are not a refreshed efficacy artifact.
- Longitudinal evidence: [Qwen AUC record](../figures/sustained_auc_effects_qwen05b.json).
- Exploratory discovery comparison and its disclosures: [E112-R1 results](../results/e112r1_two_scale_exploratory_results.json).
- Theorems, protocols, and qualifications: [full manuscript](../main.tex) and [workshop appendix](../mathai2026/appendix.tex).

## Required integrity correction

The original core reader accepted all 75 registered Dr.GRPO/ReplayDr.GRPO pairs. One replay endpoint is ambiguous: Falcon3-1B, Countdown, seed 59, registered job `30269051`.

Its single recorded sampled-evaluation file contains two complete terminal sequences at update 3,072, with matching request identity but different responses and endpoint metrics. Draws 0–3 occur at lines 172–175 and again at 338–341. The file SHA-256 is `2f6e2ea3419c617dcedafd0ce2f7da7734ead2ad19c73056c0ae579252d64465`.

| Recorded sequence | Mean pass@8 | Mean distinct@8 |
|---|---:|---:|
| First four terminal draws | 0.64453125 | 0.83984375 |
| Last four terminal draws | 0.650390625 | 0.8515625 |
| Matched Dr.GRPO control | 0.337890625 | 0.4375 |

The old core reader assigned `records[draw] = metrics`, which silently selected the last sequence. Both sequences yield positive replay effects, but that agreement does not supply a legitimate rule for choosing one. No later unambiguous evaluation or registered superseding repair was found. The [August 28 E112 integrity amendment](../preregistration/e112r1_two_scale_49pair_terminal_integrity_amendment_20260828.md) already identified this same historical comparator and explicitly prohibited choosing the first, last, averaged, or privately inspected value. The September 4 correction consistently enforces that known source ambiguity in the core comparison as well. The terminal scan found no other duplicated terminal keys, including harmless identical duplicates, among its 445 directories.

The source-admissible core therefore contains **74 paired seeds: 14 complete five-seed blocks and one four-seed Falcon Countdown block**. All 74 admissible seed effects remain positive on both pass@8 and distinct@8. Falcon Countdown's descriptive four-seed mean effects are +0.30712890625 and +0.42333984375. It receives an explicit `n=4` label and no five-seed uncertainty interval.

This exclusion leaves all ten completed MaxRL/ReplayMaxRL pair blocks intact. The completed two-scale four-arm factorial has **nine complete five-seed blocks and one Falcon Countdown four-seed intersection**. A Falcon Dr.GRPO-track cross-domain average must use seeds 55–58 in every domain; its corresponding effects are +0.44169921875 pass@8 and +0.77392578125 distinct@8. Its initial, control, and replay references must all use that same subset. The MaxRL track retains its five seeds. Cross-track interactions require their common admissible seed intersection.

Any 15-complete-block or 75-admissible-pair claim, five-seed Falcon Dr.GRPO macro interval, or figure carrying the excluded value must be updated under this correction. The historical values remain auditable; the correction must not be presented as a newly preregistered rule. The original 15-block omnibus sign calculation is superseded. A calculation restricted to the 14 complete positive block means gives a two-sided exact sign probability of 0.0001220703125 and Holm adjustment across two endpoints of 0.000244140625, conditional on the sign-test assumptions. These remain post-hoc directional summaries, and the shared models/domains do not establish a new random population of independent tasks.

## Positive empirical findings and their scope

### The loss of alternatives is observed before comparing interventions

The baseline precheck evaluates 150 runs: three models × five domains × two objectives × five seeds, at training passes 0 and 8. Each endpoint averages four fixed eight-sample draws on the evaluation bank. The 150 count is not a count of blocks each containing five additional seeds.

“Extra modes” is `distinct@8 − pass@8 = E[max(D8 − 1, 0)]`, where `D8` is the number of distinct verified keys in eight samples. Reported percentage loss is one minus the terminal five-domain macro mean divided by the initial macro mean. It is not an average of domain-specific percentage losses and not the fraction of all valid mathematical solutions removed.

| Model | Dr.GRPO initial → final extra modes | Dr.GRPO reduction | GRPO reduction |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 0.328828125 → 0.001875 | 99.4% | 98.2% |
| Falcon3-1B | 0.33203125 → 0.094296875 | 71.6% | 59.4% |
| Qwen2.5-3B | 0.20515625 → 0.120078125 | 41.5% | 49.4% |

The excluded replay endpoint does not affect this verifier-only precheck. All Dr.GRPO initial checkpoints were clean in the companion audit. Correctness decreases in 10 of the 30 model–domain–objective comparisons; it need not rise in every instance of collapse. At Qwen3B, macro raw distinct@8 rises while extra-mode breadth falls, which makes the “beyond the first” qualification essential.

There are 15 absolute-reference panels, not 17. Six Dr.GRPO panels finish below their initial raw distinct@8; two ReplayDr.GRPO panels do so, both PantryPlan. The Qwen3B Dr.GRPO Pantry values are **1.445 → 0.841**, replacing the stale 1.412 → 0.734 text. These references establish longitudinal loss and incomplete recovery, rather than assuming that every increase over a learned baseline is strong absolute performance.

### Replay improves sampled candidates on held-out problems

The corrected core supports positive effects on both endpoints in every admissible paired seed, with the sample-size boundary stated above. At Qwen2.5-0.5B Countdown, pass@8 increases from 0.48046875 to 0.671875, and distinct@8 from 0.48671875 to 1.63671875. Extra-mode breadth increases by **+0.95859375**, with unadjusted paired 95% Student-t interval **[+0.86309493, +1.05409257]**.

Countdown keys identify normalized computation trees, making this especially useful evidence for different computation routes. Across domains, the interpretation varies: MathIR represents algebraic routes, Python tested input–output behavior, and Graph/Pantry valid configurations. Python's Qwen0.5B extra-mode effect is small and uncertain despite a large correctness improvement. The paper should not describe every canonical key as an independent reasoning strategy.

The bank contains training-prompt responses only; evaluation prompts do not enter replay. The results therefore show changes in the candidate distribution on held-out problems. Aggregate distinct-mode counts do **not** directly follow individual initial modes through time, identify which specific modes survived, or demonstrate that a particular training-bank mode transferred to a new prompt. “Retention” is operationally supported by the intervention and aggregate distributional comparisons; direct individual-mode survival would require an additional longitudinal identity analysis.

At Qwen0.5B, ReplayDr.GRPO has the largest mean pass@8 effect among **five** trained direct alternatives in each domain. The untrained Pantry reference can be higher, so it must remain visually and textually distinct from the trained-method comparison.

### Replay adds value to the stronger fresh objective

For ReplayMaxRL minus MaxRL, the equal-domain average is computed within each paired seed before summarizing seeds:

| Model | pass@8 effect, 95% interval | distinct@8 effect, 95% interval |
|---|---|---|
| Qwen2.5-0.5B | +0.307 [0.208, 0.406] | +0.991 [0.815, 1.167] |
| Falcon3-1B | +0.133 [0.043, 0.223] | +0.228 [0.094, 0.361] |

These ten complete MaxRL pair blocks and their five-seed intervals are unaffected by the historical Dr.GRPO-replay exclusion. The macro results are post-hoc descriptive summaries, with unadjusted paired Student-t estimation intervals. ReplayMaxRL improves both mean endpoints in every Qwen domain. Falcon pass@8 improves in all five domains, while raw breadth improves in four: Falcon Graph's raw distinct@8 effect is **−0.09296875**. A universal domain-level improvement claim for ReplayMaxRL would therefore be wrong.

The intervention comparison supports the contribution of replay under both fresh objectives. It does not uniquely separate preservation from all other consequences of supervised replay, or show that additional breadth mediates the correctness effect. Shared code, initialization, and prescribed schedules do not make realized banks, replay rows, or computational workloads identical after the policies diverge.

### Uniform weighting has a direct, completed ablation

E120-R1 changes the target weights within the replay bank. At the frozen cutoff, its five complete Qwen0.5B blocks contain 25 treatment runs. Fresh-frequency weighting minus uniform weighting changes raw distinct@8 by:

- Graph: −0.663 [−0.808, −0.517].
- Countdown: −0.661 [−0.838, −0.483].
- PantryPlan: −0.341 [−0.564, −0.117].
- MathIR: −0.014 [−0.160, +0.132].
- Python: +0.040 [−0.926, +1.006].

This directly supports uniform key weighting as an equal-retention design choice in three of five complete blocks. It does not establish superiority for every domain, larger model, or downstream utility. It is cleaner evidence for the weighting choice than RLEP, whose historical pool, collection decoding, and replay setup differ in additional ways. A brief main-text reference to this ablation would connect the implementation choice to evidence without enlarging the claim.

### Harder Graph has a complete five-seed comparison

Level-2 Graph preserves the validator, key rule, and valid-mode-count histogram while increasing the number of hidden vertices from three to four. “Four vertices” without “hidden” misdescribes the construction. Frozen-model pass@8 decreases from 0.4375 to 0.1484375.

At eight training passes, Dr.GRPO, MaxRL, ReplayDr.GRPO, and ReplayMaxRL reach pass@8 **0.2015625, 0.434375, 0.7625, and 0.739453125**, respectively. Every replay seed exceeds every non-replay seed on both endpoints: the minimum replay pass@8 is 0.7109375 versus maximum non-replay 0.4765625; corresponding raw breadth values are 1.08984375 versus 0.619140625. The replay pass@8 effects are +0.561 [0.445, 0.677] and +0.305 [0.250, 0.360].

This is strong within-domain evidence under a controlled difficulty increase. Other Level-2 domains remain incomplete. The comparison does not establish cross-domain Level-2 generality, or that output breadth itself is the causal bottleneck for correctness.

## Primary, supporting, exploratory, and unfinished evidence

| Evidence | Frozen source-admissible scope | Latest observed progress during this audit | Interpretation |
|---|---|---|---|
| Core Dr.GRPO/replay, Level 1 | 74 admissible pairs; 14 complete blocks + Falcon Countdown n=4 | Same terminal scope | Primary retention comparison after integrity correction |
| MaxRL/replay, Level 1 | 10 complete five-seed pair blocks | 104/150 arm-runs terminal; Qwen3B Python seeds 70–71 only | Primary two-scale MaxRL contrast; incomplete third scale |
| Full four-arm Level-1 factorial | Nine complete blocks + Falcon Countdown n=4 across the two completed MaxRL scales | Same | Match all four arms before comparing objective interactions |
| Level-2 factorial | Graph complete; MathIR four-arm n=1 | 25/100 arm-runs terminal; MathIR ReplayDr also has seed 44 | Primary Graph comparison; other domains are progress only |
| E120-R1 weighting ablation | 25/45 treatment runs; five complete Qwen blocks out of nine planned | 27/45, adding Falcon Graph seeds 55 and 58; still five complete blocks | Completed one-scale ablation; larger-scale prefixes are descriptive progress |
| Qwen core trajectory AUC | Five seeds per domain, all 17 half-pass checkpoints | Companion source audit finds no conflicts | Supporting longitudinal sensitivity analysis, not independent mechanism evidence |
| E112-R1 discovery bundle | 49 integrity-valid historical pairs: Qwen 25, Falcon 24; Falcon Countdown n=4 | Third scale excluded | Exploratory historical bundled comparison after prior outcome inspection |
| Further fresh-signal controls, larger-model extensions, and remaining hard domains | No additional completed efficacy claim established here | Execution, smoke, recovery, or checkpoint progress alone is insufficient | Planned/in-progress evidence until the relevant source and completion gates pass |

E112 combines proposal support, semantic pressure, and replay-related changes. Its source records previous private looks and a post-outcome integrity amendment. Its 49-pair means are descriptive; neither a component-isolated discovery effect nor confirmatory evidence should be inferred. Its exact planned trajectory analysis is unavailable because the historical comparator violates the retry contract. Do not pool its four Falcon Countdown seeds as a fifth complete seed or transfer its protocol-specific n=4 interval rule to the primary five-seed protocol without disclosure.

The older `paper/results/paper_program_status.json` covers a historical ten-method grid and omits the current MaxRL factorial structure. Its totals should not serve as the current primary study denominator. Incomplete MaxRL, E120, Level-2, or new control runs must not be promoted by selecting earlier checkpoints, extrapolating progress, or relabeling smoke-test success as efficacy.

## Initial checkpoints, AUC, and source identity

The companion initial/AUC audit found no missing grid keys or malformed JSON rows in its specified historical-source scan, no conflicting Qwen0.5B core AUC rows, and clean initial checkpoints for every Dr.GRPO control. The separate plain-GRPO cohort remains outside that companion audit's original scope, but the later [supplemental comparator audit](comparator_source_integrity_20260904.md) checks all 75 of its initial and terminal source checkpoints and finds no ambiguity; its precheck values were also checked against the frozen numerical artifact. It found one additional ambiguity at **step 0**, distinct from the terminal exclusion: Qwen3B MathIR ReplayDr.GRPO seed 71, job `30277405`. Duplicated draws differ in mean@8 while request identities and the other endpoint metrics agree. Under full-metric integrity checking, the ambiguous descriptive initial checkpoint is omitted; its unambiguous terminal endpoint remains admissible.

The RLEP Pantry seed-43/44 directory histories also contain older evaluations, but those jobs are explicitly superseded: jobs `30538126/30538127` were replaced by `30688944/30688945`. The August 14 action-surface and August 18 memory-recovery amendments, together with `replaced_job_ids` in the ledger, supply the authoritative source identities. Reading the authorized replacements preserves their valid initial, trajectory, and terminal measurements. This differs from silently choosing whichever conflicting row appears last in a directory scan. Exact identities, hashes, and amendment links are in the companion audit. A subsequent read-only check of all ten current Qwen/Falcon RLEP Pantry sources retained all 33 available quarter-pass checkpoints, including the valid initial and terminal observations; that check is broader than the companion JSON’s 17-checkpoint discovery grid.

## How the proofs support the story

1. **Categorical collapse.** For one prompt with finitely many correct and incorrect categories, independent softmax logits, zero reference KL, and exact infinitesimal expected on-policy updates, GRPO, Dr.GRPO, and binary MaxRL follow a positive multiple of the correctness gradient. A unique initially largest correct-mode probability leads to eventual single-mode concentration. This explains a structural vulnerability in that idealization; it does not prove collapse for every neural parameterization or at the finite experimental horizon.
2. **Conditional model-size mechanism.** The shared-correctness theorem assumes the specific geometry `K_lambda = I + lambda vv^T`, with `v` identifying correct categories, identical initial logits, and comparison at matched correct mass. Increasing lambda preserves more expected sampled modes at that accuracy; every fixed finite lambda still admits eventual collapse under a unique initial maximum. This is a theorem about a specified update geometry. Adding active redundant parameters can instead speed up, leave unchanged, or slow the original trajectory depending on normalization. Parameter count alone therefore supplies no guarantee.
3. **Fixed-bank retention.** With a nonempty fixed bank after time T, recurring positive effective replay dose, and the stated continuous gradient flow, an energy bound prevents a banked category's probability from tending to zero. The same energy argument extends to smooth parameterizations in the prompt-isolated Euclidean gradient setting. It gives a qualitative positivity result; the worst-case bound can be as weak as approximately `exp(-2560)` before further dilution and is not a practical sampling or floating-point guarantee.
4. **Full support requires full coverage.** Uniform conditional correct mass follows under the additional independent-logit and full-bank-coverage assumptions. The bank must fit and contain every correct category. Python factors has 16–3,600 valid execution modes per prompt, versus capacity 16, so this is not a domain-wide experimental guarantee. Replay cannot protect an undiscovered or unretained mode merely because the theoretical full-coverage limit exists.

The neural implementation optimizes a length-normalized teacher-forced exemplar score, whereas the categorical calculation assigns probability directly to an execution category. Stochastic finite updates, adaptive optimizer geometry, shared-prompt interference, changing banks, and acquisition of complete coverage require empirical investigation. The smooth-parameter energy argument does not by itself imply convergence to uniformity for a restricted or degenerate parameterization.

## Important empirical limits

**Model size is confounded.** Qwen0.5B uses constant learning rate 2e−7; Qwen3B uses cosine peak 1e−7, floor 1e−8, and 10% warmup. Equal training passes are not matched optimization time or matched accuracy. Initial pretrained distributions also differ; Falcon changes family and some token budgets. Pantry contributes 92.7% of the initial extra-mode denominator at Qwen0.5B but 59.8% at Qwen3B. Same-family Graph and Pantry results support smaller aggregate finite-horizon loss at 3B, without identifying width as its cause. No shared-update geometry or its dependence on model size was measured.

**Fresh policy-gradient signal can become sparse.** Qwen0.5B Python controls have mixed-reward groups on only about 1.5% of updates. Direct raw-output inspection finds all five control seeds become one sampled completion per prompt within roughly 0.75–1.25 training passes, persisting to the endpoint. Equal aggregate endpoints occur in 14/15 Python and 12/15 Pantry control seeds across scales, but that equality is only a screening signature; the raw per-prompt response audit supplies direct evidence for the Qwen Python case. The larger-model mixed-group anecdote is not a complete scaling experiment. Dynamic sampling, larger groups, KL, entropy regularization, and decoding sweeps remain substantive comparison opportunities.

**Repeated measurements are not independent experiments.** Four decoding draws improve measurement of each trained policy; they are not four training seeds. AUC revisits the same runs and partly measures how long an early-collapsed policy stays collapsed. Post-hoc cross-domain averages and sign summaries do not establish a universal task population or monotonic scaling law.

**Correctness and breadth remain coupled.** Raw distinct@8 can rise because success rises. Subtracting pass@8 counts additional modes beyond the first, but still does not hold success probability fixed. The matched-accuracy theorem concerns a theoretical controlled comparison that the cross-model experiments do not implement. No mediation experiment establishes that broader support caused the accuracy improvement.

**Task identity sets the claim's meaning.** The measured outputs come from small executable domains with task-defined canonicalization. These results do not establish free-form reasoning diversity, coverage of all valid solutions, equal value of all modes, or improved downstream selection utility. The finite action-uniform references for Graph and Pantry make the absolute performance boundary explicit.

## Recommended exposition

Present the observable problem and extra-mode measure before interpreting percentages. Introduce executable output identity before describing key-balanced memory. Connect replay to its fixed-bank conditional guarantee, then show the corrected held-out core comparison, the complete MaxRL pair comparison, the direct weighting ablation, and the completed harder-Graph test. Keep the size theorem explicitly conditional and the unfinished cohorts visible as progress. This order lets the reader understand each object before it carries a claim, while retaining a strong empirical argument without treating descriptive outcomes as a stronger mechanism or generalization result.
