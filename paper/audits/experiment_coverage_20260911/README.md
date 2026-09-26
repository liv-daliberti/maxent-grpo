# Experiment coverage and main-body emphasis audit

Current update (2026-09-12 UTC): both manuscripts now cover the prompt and sampling-budget follow-ups in their main bodies, the Grok/Kimi temperature qualifications, all three completed inference ablations, and the restored semantic factorial. The latter was rebuilt with current strict source admission, the Falcon Countdown seed-59 exclusion, and E85's repaired Pantry semantic arms. See [the integration audit](../inference_followups_20260912/README.md). The dated audit below is preserved as the original finding; completed-versus-pending experiment scope remains explicit.

The main Dr.GRPO/replay, MaxRL/replay, Level-2, and hosted comparison families are represented in both papers. The core findings and their principal limitations are already in the main bodies. Coverage is not literally complete: one supporting factorial is represented only partially, and two important completed follow-ups have no main-body mention or cross-reference.

This is an audit of the current ModeBench paper study, not a claim that every historical pilot in the repository belongs in the paper. All 27 ICLR and 29 workshop source files in the recursive input trees resolve. Disk presence alone does not establish compiled-paper coverage.

## Coverage map

| Experiment or analysis | Current main body | Current supplement | Assessment |
|---|---|---|---|
| Level-1 Dr.GRPO / ReplayDr.GRPO | Figure 4, cross-scale findings, 74 admitted seed pairs | Exact endpoints, source exclusions, trajectories | Covered |
| Level-1 MaxRL / ReplayMaxRL | Figure 5, three scales, all 75 pairs; explicit 3B results | Per-domain contrasts and both-metric trajectories | Covered |
| Level-2 four-arm factorial | Figure 6, four complete domains, partial Pantry disclosed | Exact pairwise and common-four-arm cohorts; training curves | Covered; incomplete experiments are not missing publication results |
| GRPO, UCPO, sparse RLEP-Dr | 0.5B comparison matrix and design description | Additional selected scales, exact counts, effects, trajectories | Covered in the paper's selected scope |
| Fixed Semantic-MaxEnt × replay | Figure 4 includes the 0.5B semantic-only effect | Estimator and registry retained, but no full factorial display | Reporting gap: Falcon results and effects conditional on replay / interactions remain artifact-only |
| Uniform versus frequency replay | Qualitative contribution stated; workshop gives +.318 extra modes | Frozen primary analysis, current extensions, mixed effects, uncertainty | Covered; one numerical anchor would strengthen the ICLR prose |
| Longitudinal conditional concentration | Main Results gives collapse and mitigation findings with stream/eligibility limits | All 135 contrasts and detailed source qualifications | Covered; key result already promoted |
| Raw-breadth precheck and fresh-gradient degeneracy | Introduction/Method point to supporting evidence | Precheck, raw-output diagnostics, action-uniform references | Appropriate supporting placement |
| Fixed-bank score trajectories | Both conclusions retain the declining tail limitation | Figure, scores, all-identity coverage and lack of causal control | Appropriate supporting placement |
| Seven hosted deployments | Hosted figure and descriptive conclusion | Original protocols, normalization, Opus wording, all cells | Covered |
| GPT-5.6 temperature sweep | ICLR main Figure 8; workshop main summary | Complete profile-specific results | Covered |
| Grok/Kimi requested-temperature tests | No result-specific summary | Full comparison and token-limit failure | Worth one balancing sentence in the main hosted discussion |
| Opus selected first-valid retries | No detailed main summary | Preserved as a distinct variable-effort diagnostic | Appropriate supporting placement |
| Matched original/neutral prompt control | No mention or cross-reference in either main body | Complete local results and Python failure audit | Important main-body emphasis gap |
| Fresh 64-response discovery control | No mention or cross-reference in either main body | Complete local panel, two plots, collision and matched-correctness references | Important main-body emphasis gap |
| Hosted prompt/discovery follow-ups | No completed claim | Pending-panel scope disclosed; collection ongoing | Complete results must be reviewed and integrated when available |
| Older bundled discovery, AUC, scheduler/token-entropy telemetry | Removed | Archived under the approved cut plan | Deliberate exclusions; not accidental missing evidence |

## Priority changes

1. Add a compact main-body paragraph on the two completed local follow-ups, with both appendix references. Scope it to Qwen2.5-0.5B and the tested Level-2/3 Python, MathIR, and Pantry subset. The prompt control shows substantial Python dependence on the supplied strategy/example, not an isolated preference intervention. The 64-draw control shows persistent MathIR concentration alongside additional late discovery in neutral Python replay and Pantry. Both outcomes matter; a universal saturation claim would be wrong.
2. Add a sentence to the hosted temperature discussion: Grok shows no clear breadth increase at the tested higher setting, while Kimi incurs extensive token-limited failures. The GPT no-reasoning improvement should not be the only visible temperature outcome. These tests concern their own fixed settings and do not establish general temperature invariance.
3. Restore a compact supporting table for fixed Semantic-MaxEnt, including semantic-only, added-on-replay, and interaction effects where admissible. The retained `fixed_semantic_factorial_effects.json` advertises ten five-seed four-arm blocks, but predates the current Falcon Countdown exclusion. Reconcile it against the current source census before copying its estimates or five-seed labels. The 3B seed-70 artifact is descriptive and is not a completed five-seed factorial. No new main figure is needed.
4. Optionally put the frozen +.318 [.272,.358] uniform-weighting extra-mode effect into the ICLR main paragraph, alongside its inconclusive correctness contrast. The workshop already gives the point estimate.

## Concrete main-text draft

Suggested after the Level-2 result and before the hosted observations:

> Two local follow-ups test dependence on prompt guidance and sampling budget. Removing strategy guidance, including an executable Python example, substantially lowers Python correctness; trained MathIR remains strongly concentrated under both wordings (Appendix~\ref{app:prompt-hint-ablation}). A separate collection of 64 fresh responses per selected problem and wording finds persistent concentration in most trained MathIR conditions, but substantial additional discovery in Pantry and neutral-prompt Python replay (Appendix~\ref{app:sampling-budget-ablation}). These Qwen2.5-0.5B checks cover selected Level-2/3 problems. They show that eight draws can understate available alternatives and that prompt-driven correctness changes must be distinguished from redistribution among correct solutions.

Suggested short addition beside the GPT temperature summary:

> Temperature effects vary across tested deployments: Grok's breadth change is inconclusive, while Kimi's higher-temperature condition produces extensive token-limited failures (Appendix~\ref{app:hosted-temperature}).

These are review drafts, not manuscript edits or validated layout results. Preserve the user's original figure sequence and both page budgets. Space can come from shortening repeated protocol detail already provided in the supplement; this audit does not propose moving the original figures. The workshop can use a shorter version with the same two follow-up references.

## What does not need to move into the main body

Complete seed tables, every trajectory, estimator implementation, raw response taxonomy, retry accounting, and proof details can stay supplementary. The main text already communicates the essential distinction between correctness and conditional concentration, the replay intervention, its cross-scale and Level-2 evidence, weighting, hosted observations, and the fixed-bank limitation.

The approved streamlining review specifically archived the earlier bundled verified-support-discovery experiment because it jointly changed proposals, semantic regularization, and replay. It is distinct from the newly completed 64-response sampling-budget control. Reinstating it is not required for the present paper's primary claims.

The audit changed no manuscript, figure, estimate, cohort, or running experiment. `source_check.json` records the source hashes and complete input trees.
