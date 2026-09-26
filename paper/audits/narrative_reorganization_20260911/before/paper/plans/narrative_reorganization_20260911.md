# Mode Collapse in RLVR & ModeBench: proposed narrative and evidence organization

Status: editorial and analysis plan, based on the validated September 11 manuscripts. This plan does not replace either manuscript, alter frozen results, or launch evaluations. The accepted title stays **Mode Collapse in RLVR & ModeBench**.

## Update after the saved-output concentration analysis

The bridge analysis is now implemented, with its exact estimator and theory in `paper/results/conditional_concentration_20260911_appendix.tex`, a full result artifact in `paper/results/conditional_concentration_20260911.json`, and independent synthetic validation under `paper/audits/conditional_concentration_20260911/`. The initial analysis and a subsequent complete-census extension are preserved separately: the paper’s source census refreshed during the analysis, so the extension is explicitly documented after inspection of the initial results. No new model generation is required.

**The strongest supported story is now: RLVR can concentrate the distribution of verified outputs; ModeBench measures that concentration; canonical replay can reduce it in controlled comparisons.** Graph and Pantry provide the strongest longitudinal evidence: all twelve Dr.GRPO/GRPO × three-scale × two-domain contrasts increase concentration. All six ReplayDr.GRPO contrasts on those domains decrease it, with agreement in direction in both disjoint-stream orientations. Retain the single Falcon Graph GRPO orientation disagreement, sparse Python initial eligibility, already concentrated MathIR outputs, and heterogeneous MaxRL sensitivities alongside those claims. Replay’s lower conditional concentration can coincide with lower per-sample correctness on the jointly eligible prompts; full-test pass@8 and breadth are distinct outcomes.

Put the longitudinal concentration result in the main paper immediately after the measurement definition. Introduce the conditional theory as an explanation for a pressure observed in those runs, then the controlled replay intervention. Hosted systems demonstrate the broader relevance of measuring verified-output concentration; their snapshots do not identify how it arose. A main figure should display conditional collision changes with eligibility and sampling sensitivity; the broad raw-breadth precheck remains useful context in the appendix. This gives the collapse claim a direct measurement, and gives the replay claim a matched comparison using the same mathematical object.

The sampling audit changes the implementation plan: four recorded K=8 groups have overlapping child seeds, yielding eleven nominal streams under the audited runtime mapping. Use deterministic representatives and separate 5/6-stream orientations; never treat the concatenation as 32 iid observations. Original K=8 metrics retain their original groups. Distinct seed IDs do not certify statistical independence, and all random-eligible-population averages remain descriptive.

The current primary MaxRL comparison now has all 75 paired terminal runs, including five per 3B domain. Partial source histories and conditional eligibility must still be shown separately from this completed primary endpoint cohort. Level 2 remains a within-level replication, not an identified difficulty intervention.

## Recommended framing

**A binary reward records that a response succeeded; its execution identity records which success occurred. ModeBench makes those identities measurable, and canonical replay uses them during training.**

This gives the paper a single subject: the distribution of verified alternatives. The hosted evaluations describe that distribution in deployed systems. The controlled suite measures how it changes during training and when replay is added. The theory identifies a conditional mechanism and explains a sufficient replay barrier. These are complementary forms of evidence about the same object; the paper does not need to attribute hosted concentration to an unobserved training process.

The empirical claim now has two independently measured components: replay improves full-test success and sampled breadth in the reported comparisons, and it reduces conditional concentration in the demonstrated eligible populations. Universal concentration reduction, individual-mode survival, and downstream utility remain unsupported. Because collapse is in the title, the longitudinal diagnosis deserves space in the main paper, together with its success dependence and heterogeneous domains.

Suggested transition between the experimental types:

> Hosted evaluation measures which verified alternatives a deployed policy produces. Our controlled training study then examines how those distributions change during optimization and when canonical replay is added. The execution keys provide a common measurement across the two studies; their causal questions remain distinct.

## What to do and say about Level 2

### Immediate role in the paper

Treat Level 2 as a **replication of the replay comparison under a second task construction and evaluation protocol**. The four arms were trained at that level, so this is not zero-shot transfer of the Level-1 intervention. The existing within-level comparisons remain useful: they hold the evaluation prompts, training seed, fresh objective, and prescribed within-level budget fixed when adding replay.

Show the within-Level-2 replay effects and their uncertainty. Put development admission and cross-level absolute values with benchmark construction in the appendix. Retain the strong Graph result, the uncertainty in the other completed domains, and the incomplete Pantry cohort alongside the claim.

Suggested main-text wording:

> We repeat the objective-by-replay comparison on Level 2, a second task construction admitted as more demanding under frozen-model development evaluation. Within each level, replay and control use the same task population and prescribed protocol. Cross-level differences combine changes in problem structure, test population, and parts of the task interface and generation budget; they therefore do not isolate an effect of difficulty alone.

Suggested main caption:

> Replay effects within Level 2. Each point compares replay with its matched fresh-objective control on the same held-out prompts and training seeds. Four domains have complete factorials; PantryPlan is reported separately. Construction admission and cross-level reference populations are described in the appendix.

### Why a histogram repair alone is insufficient

The audited construction matches native Level-1 training histograms and designated development/confirmation reserves. Native Level-1 terminal test histograms differ from those reserves in Graph, Countdown, and Python. There are also protocol differences: the native MathIR train/evaluation cap is 64 tokens, versus 192 at Level 2; the Level-2 Countdown, Python, and MathIR profiles introduce domain-specific prompting/guided syntax. These apply to rollout generation and terminal sampled evaluation, not merely development admission. Graph has the cleanest unchanged profile among the four complete domains.

Matching the number of valid modes controls one property of a task, not the entire difficulty intervention. Re-evaluating on the construction reserve would address the population mismatch while leaving other differences to describe or control explicitly. The designated confirmation reserve has already been evaluated in the Level-3 construction work; reusing it would be a retrospective sensitivity analysis, not a new untouched confirmation test. Prior evaluation is distinct from optimizer training leakage.

### Additional work, ordered by value

1. **Proceed with the within-level interpretation and rewrite.** No new experiment is required to support the current within-level replay claim.
2. **Optional existing-data sensitivity:** compare within-level effects after standardizing over valid-mode-count strata on common support. Freeze outcome-independent target weights, use the same weights for all arms, retain seed pairing, and report the retained prompt fraction, empty strata, and weight concentration. Recover counts from the verified reference/specification: Python logs can carry a placeholder `answer_mode_count=1`; the actual `reference.num_modes` or validated manifest is required. This diagnoses sensitivity to one population difference; it does not identify difficulty. Do not extrapolate into absent strata or call the standardized cross-level difference causal.
3. **Optional evaluation repair:** evaluate the relevant frozen Level-1 checkpoints on the designated construction-reference test sets, with an explicit protocol. Checkpoint availability, prompt exposure, caps, syntax, seeds, and all four arms must be audited first. Compare with native results and report this as a retrospective reserve analysis. For scale, four domains × four arms × five seeds × 128 prompts × four eight-draw groups is 327,680 generated responses for one added complete evaluation pass; missing or inadmissible intersections must retain their actual sizes. The required archived model receipts exist for all 80 cells in each of the four-complete-domain Level-1 and Level-2 panels. Their terminal export names use `step_03073`, while the paper evaluations use step 3,072; establish the indexing correspondence before calling restored weights identical to the reported endpoint. This is not needed for the recommended narrative.
4. **Only if isolating difficulty becomes a headline claim:** define a paired task transformation, fix a policy checkpoint and decoding law, and evaluate both conditions on paired instances with documented verifier/key correspondence and support. For a claim about learning under difficulty, also control or randomize the training construction and keep optimization budgets fixed. Even this estimates the specified transformation, not an abstract universal quantity called difficulty.

My recommendation is step 1 now. The saved-output concentration analysis below is a higher priority than a new difficulty campaign.

## The most useful additional bridge: conditional concentration

Retain pass@8 and distinct@8 as the existing primary outcomes. Add a clearly labeled retrospective analysis of the conditional distribution over verified keys, using the existing hosted correct-pair collision measure on saved training outputs where available.

For a fixed prompt, let P be correct mass and q the conditional distribution over valid keys. The existing identity is

\[
S_K=1-(1-P)^K,\qquad B_K=\sum_c[1-(1-Pq_c)^K].
\]

Correct-pair collision is

\[
C(q)=\sum_c q_c^2.
\]

It has an especially useful relation to the theory. In the ideal categorical collapse flow after its positive time change,

\[
\frac{dC}{d\tau}=2\left[\sum_c q_c^3-\left(\sum_c q_c^2\right)^2\right]
=2\operatorname{Var}_{c\sim q}(q_c)\ge0.
\]

For two draws the occupancy identity also gives \(B_2=2P-P^2C(q)\). This connects the success dependence of raw breadth, the theory's concentration measure, and an already implemented hosted statistic. These identities do not establish the same monotonicity for the neural optimizer.

With R≥2 correct saved outputs and per-key counts n_c,

\[
\widehat C=\frac{\sum_c n_c(n_c-1)}{R(R-1)}
\]

has conditional expectation C(q) given R under iid sampling from the same policy. Use it for the prespecified initial/final and replay/control checkpoint comparisons; first verify the availability of per-response keys and consistent decoding. Keep the existing four K=8 training groups intact for the primary metrics. The source audit found overlapping child seeds, so direct concatenation is not an iid sample. The implemented secondary statistic instead selects eleven distinct nominal streams and checks disjoint 5/6-stream orientations; neither view changes the original pass@8 definition.

Important analysis choices:

- Ineligible prompts have an undefined conditional estimate, not zero diversity.
- Use equal prompt weights on a common eligible intersection for each paired contrast, and show its coverage and correctness. The conclusion then concerns that selected observable population; it does not identify concentration on all prompts.
- Preserve hosted pair-pooled summaries as a distinct estimand. At a common sampling budget their population weights are proportional to P², so pooled changes can reflect changing correctness weights even if each prompt's q is unchanged.
- Respect training-seed pairing and prompt clustering. Pairs sharing responses are not independent observations, and local checkpoints are not interchangeable with independent hosted deployments.
- Freeze the retrospective analysis definition before computing it. Report all admitted domains and signs. A decrease in C does not establish majorization, full support recovery, or individual-mode survival.

The completed analysis supplies both the strongest bridge and its boundaries: Graph/Pantry training concentration, matched replay reductions, visible stream sensitivities, and sparse or undefined results elsewhere. Use the direct concentration evidence in the main story while retaining the independent success-and-sampled-breadth claim.

## Recommended ICLR main: about 8.7 pages including figures

| Order and budget | Section question | What belongs in main |
| --- | --- | --- |
| 1. Introduction — 1.0 page | What can accuracy hide? | One concrete verified-output example; a compact longitudinal observation; the shared question; three linked contributions: measurement, mechanism, intervention. |
| 2. ModeBench and measurement — 1.0 page | What is an alternative, and what do our metrics measure? | Five-domain/key/alias table, P and q, the two sampling identities, and the distinction between sampled breadth and conditional concentration. Full prompts are unnecessary here. |
| 3. Verified-output concentration — 1.15 pages | What do we observe during training and in hosted systems? | Promote the completed before/after collision analysis with eligibility and sensitivity; retain the raw-breadth precheck in the appendix. Present the hosted landscape after the measurement definition, with accurate-and-concentrated cases and broader counterexamples. Keep longitudinal changes and deployment snapshots visibly distinct. |
| 4. Conditional mechanism and canonical replay — 1.25 pages | Why can this happen, and what does memory change? | The idealized collapse mechanism, actual replay loss, categorical mass/balance identity, short retention statement, and the bank-coverage/neural-transfer boundary. |
| 5. Controlled intervention — 1.85 pages | Does replay help under both fresh objectives? | Make the Dr.GRPO/MaxRL × replay factorial the centerpiece. Include complete scale results, the completed five-seed 3B cohort and explicit conditional-analysis eligibility, and compact domain-specific effects. The Dr-only experiment becomes part of this design. |
| 6. What matters and where it weakens — 1.55 pages | Does key weighting matter, and does the result recur on another construction? | Promote uniform-versus-frequency replay. Present within-Level-2 effects. Keep the Falcon Graph exception, Graph-driven Level-2 pattern, and declining exemplar-score tail adjacent to the positive claims. |
| 7. Related work and discussion — 0.9 page | What have we established? | Position the measurement and intervention; distinguish conditional theory, practical gains, and untested downstream utility. Refer once to the detailed claim map. |

These are planning budgets, not layout guarantees. If the page fills, reduce repeated figures and prose before shrinking figures or moving decisive caveats out of view. A small neutral-prompt result can fit into section 6 only after completed evidence is admitted; it should replace a weaker sensitivity paragraph rather than create an extra main storyline.

### Five primary figure roles

| Proposed slot | Content and purpose | Current source and required change |
| --- | --- | --- |
| Figure 1 | Worked mode identity plus a compact training diagnosis | Recompose `modecollapse_story` with a compact view from `conditional_concentration_overview.pdf`; keep the full concentration panel, selected example, and raw-breadth precheck in the appendix. If using replay in the teaser, use matched objectives or show the factorial. Preserve explicit illustrative selection. |
| Figure 2 | Hosted correctness and verified breadth across domains/deployments | Move `hosted_verified_breadth` earlier. Retain all domains, including broad Pantry outputs. Keep the changed Opus Python prompt condition visible; the panel is not a fair common-protocol ranking. Add any newly admitted conditional-concentration view as a labeled secondary analysis. |
| Figure 3 | Central objective-by-replay result | Rework `e118_all_scale_factorial_progress` with enough per-domain information to expose heterogeneity. Preserve actual paired cohorts and the distinction between complete primary endpoints and incomplete conditional eligibility. The older broad comparator matrix moves to the appendix. |
| Figure 4 | Uniform versus fresh-frequency key weighting | Promote `e120_primary_breadth` and current weighting results into a readable plot of correctness and extra modes, with the same seed/block scope. This directly tests the memory design choice. |
| Figure 5 | Within-Level-2 replay effects | Recompose the terminal portion of `modebench_level_admission` as paired effects by domain/objective. Move frozen development admission to the construction appendix. A compact table is acceptable if a plot crowds the page. |

Move the full `modebench_examples` figure, standalone `verified_support_story` schematic, direct-comparator catalogue, and full GPT temperature curve to the appendix. The temperature result still earns a main sentence: in the measured no-reasoning profile, decoding changes can improve observed breadth and success. It cannot be used to claim that sampling changes never help.

Use the same names and primary metrics throughout. Where the hosted overview currently uses per-response accuracy, either retain and label that distinction plainly or derive a companion empirical pass@8 view from the saved groups. Do not relabel mean@8 as pass@8 or merge heterogeneous populations into a single leaderboard. Keep differences in sampled budget and uncertainty visible.

## Main-text mathematical spine

Distribute about one page of mathematical content across measurement and method, rather than hiding the reasoning entirely in the supplement or inserting all proofs into main.

1. **Identity beside measurement:** the promptwise success/occupancy equations above, with one sentence about fixed-P concentration and changing correctness.
2. **Conditional mechanism beside the training diagnosis:**
   \(\frac{d}{dt}\log(q_c/q_d)=c_G(P)P(1-P)(q_c-q_d)\).
   State the isolated prompt, independent canonical-outcome logits, common length normalization, Euclidean infinitesimal on-policy updates, and absent competing regularizer in the same paragraph. A unique initial winner concentrates asymptotically under those assumptions. Include the C derivative to link this conditional prediction directly to the completed saved-output diagnostic.
3. **Replay design beside the weighting ablation:** for the equal-weight categorical surrogate,
   \(R_{\mathcal B}=-\log P_{\mathcal B}+\log|\mathcal B|+\mathrm{KL}(U_{\mathcal B}\|q_{\mathcal B})\).
   Keep the actual mean-token exemplar loss separate. State the fixed-bank, positive-dose retention result and the fact that unbanked modes may vanish in the same model.

Keep exact coefficients, full proofs, quantitative floors, equality/tie cases, the shared-parameter/exemplar extension, and discrete-energy conditions in the appendix. The fact that those conditions are not established for AdamW, and that unequal lengths/aliases separate exemplar scores from key probabilities, remains visible in main. The empirical score tail is an informative boundary, not a validation of the probability floor.

## Appendix: organized to answer the main paper's questions

Open with a one-page navigational claim/evidence map. Put the scientific material before operational provenance.

| Appendix | Contents |
| --- | --- |
| A. Measurement and mathematical results | Full occupancy/concentration arguments; categorical update and collapse proofs; replay decomposition, retention, weighted/full/partial-bank limits; exemplar bridge and finite-sampling visibility; entropy and neural-optimization scope. Keep the proof content identical across both papers. |
| B. Benchmark construction and task contracts | Full domain examples, alias rules, support counts, exact prompts, validation, split identities, and construction-reference versus native-test table. Include development admission here. |
| C. Training design and algorithm | Pseudocode, real length-normalized loss, admission/capacity/schedule, optimizer and syntax settings, seed intersections, computational controls, and comparator definitions. Include an explicit Level-1/Level-2 prompt/token/syntax table. |
| D. Controlled results | All domain/scale factorial endpoints and trajectories; weighting effects; within-Level-2 estimates and any standardized sensitivities; exact partial cohorts; direct alternative methods. |
| E. Longitudinal and mechanism diagnostics | Full precheck, completed conditional-concentration analysis, with both source-census phases and all sensitivities, absolute references, fresh-gradient availability measurements, fixed-bank score distributions including deterioration. |
| F. Hosted deployments and sensitivity | Full original and normalized results, prompt-specific alternatives, collision eligibility/reference weights, uncertainty, provider outcomes, temperature/retry studies, and the prompt-hint ablation only when complete. |
| G. Reproducibility and disclosures | Source admissibility, receipts/hashes, missing or excluded endpoints, code/data/model release, continuation details, and required statements. Historical experiment chronology stays in the research archive. |

Keep a concise boundary beside each main result; the appendix carries the full qualification and reproducibility detail. A statement that changes what a main figure means belongs in that figure's caption or adjacent paragraph.

## Workshop adaptation

Use the same claim and shared theory, with three main figures rather than compressing seven into four pages:

- Page 1: one worked execution-key example, the measurement distinction, and a compact hosted/training observation.
- Page 2: conditional mechanism, actual replay objective, and the central factorial design.
- Page 3: factorial results with visible cohort and domain exceptions.
- Page 4: key-weighting ablation, a short Level-2 robustness result, and the scope of preservation/utility claims.

The full hosted landscape, level construction, baseline catalogue, temperature sweeps, and proofs remain in the supplement. Reuse visual encodings and numerical snapshots while allowing different main/appendix placements.

## Execution order and acceptance criteria

1. Freeze the current validated papers and the revised claim spine. This plan is based on the [completed correction record](../audits/claim_theory_alignment_20260911/README.md).
2. **Completed:** audit saved keys, freeze estimands and stream selection, compute both documented source-census phases, report all 135 contrasts, and add the exact mathematics and tables to both papers. The 85 tests and both manuscript builds pass; use the resulting numerical artifact for the figure reorganization.
3. Recompose the five primary figures from source-bound data; promote diagnosis, factorial, and weighting evidence. Move admission, temperature details, and redundant schematics with their captions and provenance.
4. Rewrite the main in the proposed order, then organize the supplement and workshop. Keep main-visible exceptions and the full identical proof chain.
5. Reconcile the figure manifest, figure numbering, nested includes, snapshot bindings, and placement checks. Current checks hard-code eight ICLR main figures and hosted-after-training ordering; update those editorial expectations deliberately while preserving numerical, cohort, official-style, and page-limit checks.
6. Build and inspect both PDFs and the workshop source bundle. Verify no broken references, no silently changed metric units, no cohort drift, and no new completed-result claim without admitted evidence.

The existing [prompt-hint ablation plan](../../artifacts/modebench_prompt_ablation_20260911/ANALYSIS_PLAN.md) is a valuable complementary control. The current manuscript admits the local panel while omitting the incomplete hosted panel. The plan compares original/neutral wording in both settings and includes Level-3 transfer for Level-2-trained local checkpoints. Reuse admitted local evidence now and the hosted work when ready; do not duplicate its evaluation campaign or make the pending outcome a prerequisite for this rewrite. It does not by itself control every cross-level change in prompt, syntax, and budget.

## Source anchors for the cross-level recommendation

- [Current Level-2 construction discussion](../main.tex:894) and [within-level terminal evidence](../main.tex:1471).
- [Construction/admission manifest](../../var/results/modebench_level2_r5_frozen_repeat1/admission_fairness_report.json).
- [Frozen level comparison and terminal sources](../results/modebench_level_comparison_snapshot.json).
- [Native Level-1 inherited configurations](../../var/artifacts/e72_frontier_source_runs.json).
- [Existing Level-3 confirmation evaluation of the reference reserve](../../var/results/modebench_level3_v1/confirmation_05b_countdown.json).
- [Hosted protocol and its measurement boundaries](../results/frontier_comparison_20260911_protocol.tex).
- [Current figure inventory](../FIGURE_MANIFEST.md).

The [supporting cross-level audit](crosslevel_evidence_and_options_20260911.md) records exact launcher/runtime sources, common-stratum coverage, prior reserve exposure, and archived-checkpoint availability for these options.
