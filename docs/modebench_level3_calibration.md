# Level 3: Qwen 3B difficulty matched to Qwen 0.5B on Level 1

Status: frozen revision1 fails Graph Coloring confirmation; the other four comparisons are still running. No Level3 dataset is admitted.

## Objective

Construct Countdown, Graph Coloring, Python Factors, MathIR, and PantryPlan
problems for which frozen Qwen2.5-3B-Instruct has approximately the same
initial sampled success rates as frozen Qwen2.5-0.5B-Instruct on Level 1.
Each domain must have exactly 384 training, 128 development, and 128
evaluation rows, matching Level 2. Match each split's existing Level 2
valid-mode-count histogram and retain the existing verifier and canonicalizer.
Preserve Pantry's support/family composition. Keep new identities disjoint
from Level 1, every available Level 2 revision, and other Level 3 splits.

## Calibration protocol

The selected comparison uses the established Level 2 Qwen r5 interface for
both Qwen models: boxed-direct without grammar for Graph Coloring, and the
existing hybrid solver prompt plus target-blind legal syntax for the other
four domains (Countdown uses countdown_legal_v3). This measures Qwen 0.5B on
Level 1 reference problems and Qwen 3B on Level 3 candidate problems under
the same interface. It is not a reproduction of historical E78 training
interfaces or learning curves.

The first diagnostic used boxed-direct freeform answers and 192 output tokens
in all domains. It returned zero success for Qwen 0.5B in both Python and
Pantry, with observed answer-format failures. Before candidate selection or
confirmation, the protocol was amended to the existing Level 2 interface to
provide useful nonzero targets in all five domains. The diagnostic receipts
are retained; they serialize their complete settings under the earlier label
`original_level1`. The amendment, exact replaced job IDs, and replacement
manifests are recorded in
`var/artifacts/modebench_level3_v1/guided_interface_amendment.json`.
The user was asked for a preference; the stated recommended default is used
until a reply changes it.

Use the same temperature 1, top-p 1, 192 generated tokens, native chat
template, and eight samples per problem. All checkpoint paths, prompts,
input rows, decoding settings, and sampling seeds are recorded in receipts.

Measure sampled correctness (pass@1, the fraction of eight samples that
verify), pass@8 (the fraction of problems with any verified sample), and
distinct verified canonical modes@8. Calibrate the first two metrics per
domain. Distinct-mode coverage is a reported diagnostic, not an outcome
to force equal: subsequent diversity differences remain experimental results.

Candidate generation varies task structure and numeric complexity while
preserving exact valid-mode counts. Evaluate development candidates only.
Select a generator configuration or a mixture of configurations from
development evidence, then freeze that choice before generating and
evaluating the confirmation split. Do not select individual confirmation
problems using model outcomes. If confirmation fails, retain its receipt and
mark that revision unconfirmed; any subsequent revision needs a fresh
confirmation split and explicit recorded development history.

Before looking at candidate outcomes, set practical matching tolerances to
8 percentage points for pass@8 and 4 percentage points for pass@1, requiring
both in every domain. Report prompt-level uncertainty alongside observed
differences; passing these numerical tolerances is an approximate empirical
match, not a proof of statistical equivalence. Use four registered sampling draws per problem per model during development
fitting (seeds 6318000--6318003), then four independent draws for the final
check (6319000--6319003). The initial one-draw runs are retained as structural
pilots; their outcomes alone cannot qualify a final recipe. Do not
interpret repeated draws as additional independent problems.

## Development fitting amendment before confirmation

The first graph recipe is retained at
`var/artifacts/modebench_level3_v1/recipes_v1/graph_coloring.json` as a diagnostic.
It minimized errors on 128 selected development rows across 1,771 weight
recipes. Although each pool's hash ordering was fixed before fitting, choosing
weights this way also chose which portions of those ordered pools appeared in
the scored subset. Fixed ordering therefore did not prevent selection overfitting.

That recipe's selected pass@1/pass@8 were 17.55%/41.60%, while the forecast
from all pool rows, stratified by support and using the same exact allocation,
was 20.47%/49.12%. The baseline was 19.09%/38.28%. The selected pass@8 was
7.52 percentage points below its own pool forecast: 2.56 fixed-weight
random-subset standard deviations. This is a descriptive diagnostic, not a
post-selection p-value or a statistical-equivalence test. No confirmation
outcomes were used to identify or correct this issue.

Recipe schema v2 ranks all weight choices solely by full-pool cell forecasts.
For each metric the forecast is the sum of each cell/tier's allocated count
times its mean over every scored pool row in that cell, divided by 128.
The exact controlled allocation preserves support/family counts and global
tier totals. Ranking minimizes the same tolerance-scaled maximum error, then
squared error, with integer weights as the deterministic final tie break.
Selected-row residuals do not enter the objective or its tie breaks.

After choosing that primary optimum, the fitter scores its one deterministically
selected development set. Both the full-pool forecast and the selected set must
satisfy the unchanged 4-point pass@1 and 8-point pass@8 gates. If either fails,
the recipe fails; the fitter does not try another weight choice, row order, or
hash seed to manufacture a passing selected set. New recipes go to `recipes_v2`;
old diagnostic recipes are not overwritten and cannot qualify for finalization.
Fresh, independent four-draw held-out confirmation remains required in all five
domains. This amendment changes neither sampling budgets nor matching tolerances.

## Required evidence before completion

- Five verified datasets, each with 384/128/128 rows and exact reference
  support histograms, plus cross-split and historical identity disjointness.
- Versioned generator configuration, source hashes, split seeds, row hashes,
  and exact local model checkpoint identities.
- Qwen 0.5B Level 1 and Qwen 3B Level 3 baseline receipts under the same
  named interface and sampling budget.
- Development calibration records and an independently sampled, held-out
  confirmation report with all five domains inside both matching tolerances.
- A clear report of residual differences and sampling uncertainty. Training
  runs and learning-curve equivalence are outside this dataset-building task.


## MathIR sign-law development revision and resource record

The MathIR v3 full-pool forecast passed both gates, but its one fixed selected
set had pass@8 0.25390625 against the baseline 0.333984375. The absolute gap
0.080078125 exceeds 0.08. The failed recipe is preserved in
`var/artifacts/modebench_level3_v1/recipes_v4/mathir.json`; it does not qualify
through rounding, an alternate mixture, or another selection seed.

A separate prospective revision conditions the original one-sided binding
law on (1) a>0,b<0, (2) a<0,b>0, or (3) a*b<0; its fourth preset conditions
the existing rational law on positive e and f. Coefficient bounds, original
family identities, six A--F actions, four-step limit, and five canonical modes
remain fixed. Earlier sign-group diagnostics were noisy and inconsistent;
they motivate an experiment and do not establish an improvement. The new
source, exact laws, generation seeds, dependency hashes, all 512 verified
pilot rows, and final exclusion checks are recorded in
`var/data/modebench_level3_calibration_mathir_sign_v1/pools/mathir/generation_plan.json`.
The isolated generator passed 17 tests before model evaluation. Shared
routing remains unchanged until a complete four-draw recipe qualifies.

The original four sign-law jobs requested A5000 GPUs and were still pending
with zero runtime when the scheduler estimated overnight starts. They were
cancelled and replaced with identical tasks on permitted low-priority A6000
resources. Replacement jobs are 31146082--31146085; the full old/new commands
and task hashes are in
`var/artifacts/modebench_level3_v1/mathir_sign_development_jobs.json`.
This changes the requested GPU and scheduler route, not checkpoint, FP16
precision, prompt, grammar, context limit, batch size, seeds, or decoding.
Final confirmation is planned on the same A6000 resource type for both
models; the manifests remain unsubmitted until all five recipes and final
split provenance pass. No confirmation outcomes have been used.

## Resumption audit, 8 September 2026

The user resumed dataset construction. The four passing recipes in
`var/artifacts/modebench_level3_v1/recipes_v4` were authenticated against the
current generators and complete development receipts. MathIR v3 remains a
strict failure; the registered sign-law revision is still sampling. No final
split or confirmation outcome has been produced.

| Domain | Level 1 Qwen 0.5B pass@1 / pass@8 | Selected Level 3 Qwen 3B pass@1 / pass@8 | Development gates |
| --- | --- | --- | --- |
| Countdown | 1.709% / 12.305% | 2.222% / 12.305% | Pass |
| Graph Coloring | 19.092% / 38.281% | 16.504% / 45.508% | Pass |
| Python Factors | 32.178% / 92.578% | 34.131% / 89.844% | Pass |
| MathIR v3 | 5.298% / 33.398% | 7.764% / 25.391% | Fail |
| Pantry | 5.664% / 19.531% | 5.957% / 18.555% | Pass |

Passing means that both the full-pool forecast and the one fixed selected
set meet the original numerical tolerances. These are development results;
independent confirmation remains required. The latest progress and archived
resumption record are in `var/artifacts/modebench_level3_v1/progress.json`.
The read-only `ops/exp_scaling/status_modebench_level3.py` command now includes
revision-specific job ledgers, saved batch counts, and recorded recipe
versions. Use `advance_modebench_level3.py` to authenticate recorded recipe
status against current source and receipt hashes.

The source-only held-out audit is recorded in
`var/artifacts/modebench_level3_v1/heldout_source_audit_20260908T192053Z.json`.
All five Level 1 evaluation references have 128 unique rows, the same support
histograms as the Level 2 evaluation targets, and zero identity or exact-prompt
overlap with baseline development or any of the 76 candidate pools checked.
For Pantry, Level 1 and Level 2 have matching support and family marginals but
different support-by-family joint tables. Level 3 inherits the Level 2 joint
composition as specified above. Family proportions remain 25% each in both
splits; the support distribution changes between development and evaluation.
This is a source-composition qualification to the difficulty comparison,
not a change to the registered comparator, fitting rule, or tolerances.

## Passing five-domain freeze revision

The complete sign-law experiment passes both forecast and selected-set gates:
its selected MathIR pass@1/pass@8 are 7.1777%/27.9297%, versus the baseline
5.2979%/33.3984%. The frozen mixture uses tier weights [0, 0, 19, 1]/20.
The public generator route now selects the sign-law source. Fresh recipes in
`var/artifacts/modebench_level3_v1/recipes_v5` bind the updated source tree.
The other four domains retain exactly the same scientific fields as v4;
only source provenance changes. The failed MathIR v3 recipe remains intact.

The adoption passed 134 recipe, route, support, and confirmation-audit tests.
The canonical prepared confirmation plan is
`var/artifacts/modebench_level3_v1/resume_20260908T191229Z_confirmation/plan.json`.
It uses identical A6000 resources for both models, excludes the node associated
with repeated preemption, and authenticates all five frozen datasets and
receipts before submission. Its worker verifies the sealed inputs again
before invoking the unchanged evaluator.

## Published datasets and confirmation execution

The frozen dataset is `var/data/modebench_level3_matched_v1`: all five domains
have exactly 384 training, 128 development, and 128 evaluation problems
(3,200 total). Its immutable identity report records passing structural checks,
source hashes, recipe hashes, all excluded candidate pools, and split hashes.
Independent final Python and MathIR audits also passed all 1,280 and 3,200
original-verifier witnesses, respectively.

The canonical confirmation plan authenticated 303 pinned files and submitted
ten jobs. All were still pending with zero runtime when the scheduler estimated
hours or overnight waits. Test-only probes found earlier availability for
one-hour requests on the regular `all` queue. Comparable four-draw development
runs took 16--37 minutes; host-memory requests remain 48 GB because measured
MathIR peak usage was about 26 GB. The pending jobs were replaced with the same
sealed worker arguments, A6000 GPUs, six CPUs, original task payloads, and
registered sampling settings; only the queue and requested walltime changed.
The original plan and submissions remain intact. The amendment and current
job IDs are recorded in
`var/artifacts/modebench_level3_v1/confirmation_one_hour_regular_queue_20260908T1942.json`
and `var/artifacts/modebench_level3_v1/confirmation_regular_queue_jobs.json`.
No held-out outcomes were available when this resource amendment was made.

The one-hour regular queue also remained pending in actual scheduling.
Before any confirmation batch existed, a second prospective resource amendment
fixed the same RTX 6000/Turing hardware for both models in all five domains.
Node007 had ten idle 24 GB GPUs and enough host RAM for five 48 GB jobs at a
time. The installed Torch build supports compute capability 7.5; installed
vLLM 0.8.4 automatically uses its available XFormers CUTLASS attention backend
on this hardware. This differs from Ampere's FlashAttention2 backend and is
explicitly part of the confirmation resource record. Checkpoints, FP16,
interface, task files, seeds, batching and all decoding settings remain sealed.
The pre-outcome amendment is
`var/artifacts/modebench_level3_v1/confirmation_rtx6000_before_outcomes_20260908T1952.json`.
A read within the first running allocation confirmed Quadro RTX6000,
compute capability7.5 and24576MiB device memory.

## First held-out result: Graph Coloring failure

Both Graph confirmation receipts completed and authenticated after the temporary
filesystem failure was repaired with an explicit project-local TMPDIR. The
failed runtime attempts produced no sampling batches. Under the prospectively
registered common RTX6000/XFormers hardware, Level1 Qwen0.5B scored
22.4854% pass@1 and64.4531% pass@8; Level3 Qwen3B scored10.4004% and35.5469%.
Both unchanged matching gates fail. The first frozen revision therefore cannot
be admitted, regardless of the remaining four results. All outcomes and frozen
choices remain intact; these results may inform a documented subsequent
revision, which will need fresh confirmation data. The independent Graph audit
is `artifacts/modebench_level3_graph_coloring_confirmation_subset_audit_20260908T200839952956Z.json`.

## Sampling overlap discovered during confirmation (2026-09-08)

The installed vLLM 0.8.4 V0 implementation expands an n=8 request into child seeds `s` through `s+7` (`vllm/sequence.py:1433–1434`). The four registered consecutive draw seeds therefore use only 11 distinct child seeds, and every prompt receives the same seed block. The earlier description of four independent draws is incorrect. In Graph Coloring, 99.9% of adjacent development-draw outputs at overlapping seed positions are identical; held-out baseline and candidate overlap exceeds 99.5%. Evidence is retained in `artifacts/modebench_level3_rng_overlap_root_cause_20260908T2025Z.json`.

The v1 marginal per-draw measurements remain recorded, including the failed Graph comparison. Their averages have strong dependence, and the prompt-only bootstrap omits the shared RNG dependence. All v1 results are diagnostic, and cannot establish the intended independently sampled five-domain match. Remaining sealed jobs continue without source or input changes.

Prospective correction: `ops/modebench_independent_seeds.py` maps a versioned canonical tuple of domain, problem text, and registered draw label to a 60-bit SHA256 block multiplied by eight. Different prompt/draw keys receive disjoint aligned child-seed ranges; a collision aborts sampling. Seeds are independent of answers, metadata, model identity, row order, task sharding, and outcomes. Nine regression checks pass. Integration must record the complete seed schedule in the receipt identity and each draw, and use a new interface/schema so v1 evidence cannot be silently accepted. Both models retain the same prompts, syntax constraints, FP16 weights, n=8, temperature, token budget, and match thresholds. All five development fits require new receipts; confirmation must use fresh disjoint prompts and new draw labels.

## Independent sampling revision v2 (2026-09-08)

The prospective v2 protocol is `var/artifacts/modebench_level3_v2/protocol.json`. It retains the exact models, prompts, legal syntax, FP16, temperature 1, top-p 1, 192 output tokens, eight samples per draw, all support/family constraints, both tolerances, and the forecast-only fitting algorithm. Its four development labels are 6328000–6328003 and confirmation labels are 6329000–6329003; labels feed the per-prompt seed allocator rather than vLLM directly. The evaluator uses the new interface `level2_qwen_r5_independent_v2` and distinct receipt schema, authenticates the complete schedule and recorded child seeds, and requires vLLM 0.8.4 V0.

A fresh Level 1 Countdown control is infeasible under the original law and all-history exclusions: the finite catalogue has 787 identities, of which only 19 remain unused (128 are required). The complete capacity plan and exclusion inventory are `var/artifacts/modebench_level3_v2/level1_fresh_reserve_capacity_failed_20260908T2042Z.json` (SHA256 `43dc015cc8eefad6ef591d2b6daff5dcffae0fc6c451ebbb96b9faf1ab4cf959`). No replacement baseline reserve was published.

The prospective amendment `var/artifacts/modebench_level3_v2/confirmation_control_amendment.json` (SHA256 `cd8168785e9d469d7d715123ff0120170e9091cd6ce7aad3d3c4ceb3aff93dd6`) therefore retains all five original 128-row Level 1 evaluation controls and re-scores them with corrected independent draws. Only Level 3 receives fresh held-out evaluation prompts. These are reused controls, not an untouched two-sided confirmation. All fitting remains on original development inputs and new independent receipts; no v1 confirmation score enters weight ranking. The amendment and its pre-submission attestation predate all v2 job submissions and outcomes. The previous paragraph's fresh-baseline requirement is superseded by this amendment.

All 25 development jobs were submitted as 31146852–31146876. Ledger: `var/artifacts/modebench_level3_v2/development_jobs.json`; implementation seal SHA256 `e03ffa74a476638401ddadb49b950f3d74377457114bf13b630ed8834be34736`. The seal authenticates 358 files, 3,136 prompts, 12,544 distinct request blocks, and 100,352 distinct child seeds. Pantry's original baseline dev remains 64 rows; every other task has 128. Jobs use RTX6000/XFormers/V0, 48 GiB host RAM, 6 CPUs, one hour, and the workspace scratch directory. All depend on the two Graph development runtime checks. A documented resource-only amendment releases those two pending checks into spare node007 capacity while the remaining old diagnostic jobs finish; the v2 dependencies and scientific settings are unchanged.

New entry points (old sealed sources are preserved):

- `ops/evaluate_modebench_level3_independent.py`: independent evaluation and seed-receipt authentication.
- `ops/exp_scaling/fit_modebench_level3_independent.py`: authenticate all five domain development receipts, then use unchanged mixture fitting and one selected-set check.
- `ops/exp_scaling/advance_modebench_level3_independent.py`: read-only progress by default; `--fit-complete` publishes one immutable fit for each completed domain.
- `ops/exp_scaling/finalize_modebench_level3_independent.py`: exact-refit all five recipes; authenticate the amendment and prior/control data; publish fresh 384/128/128 splits after all checks. Use `--protocol-amendment var/artifacts/modebench_level3_v2/confirmation_control_amendment.json`. Fresh train/eval seeds equal the original domain seeds plus 1,000,000 and the original split offsets. Prior v1 development may be reused as development, but prior train/eval and fixed controls remain protected.
- `ops/audit_modebench_level3_independent_match.py`: require actual source rows, immutable recipe/amendment provenance, complete structural checks, and disjoint RNG streams; recompute both unchanged gates.

The independent auditor/finalizer/evaluator/fitter review passes 93 combined tests, with the 358 sealed calibration inputs unchanged. No v2 match has yet been claimed.

The fixed-development hardware check is complete: `artifacts/modebench_level3_graph_fixed_development_runtime_bridge_comparison_20260908T205454727707Z.json`. On the same original development prompts and legacy draw labels, baseline pass@8 changes only 0.3828125→0.384765625 and candidate pass@8 remains exactly 0.455078125. Baseline pass@1 changes +0.01416015625; candidate changes −0.00048828125. This makes the hardware change an inadequate explanation for the large held-out pass@8 shift, while preserving the explicit old-seed and candidate batch-context limitations.

The first three corrected jobs have saved real batches. `var/artifacts/modebench_level3_v2/first_runtime_independent_seed_authentication.json` recomputes their complete prompt/draw schedules, validates every saved batch's identity and seed metadata, and authenticates the unchanged evaluator sources. No incomplete metrics were used for fitting.

All ten legacy v1 receipts and the full original audit are complete: `var/artifacts/modebench_level3_v1/resume_20260908T191229Z_confirmation/confirmation_report.json`. Every receipt/source/model/interface check passes, but four domains miss at least one numerical gate: Graph both; Python and MathIR pass@1; Pantry pass@8. Countdown is within both tolerances. The entire campaign remains correlated-RNG diagnostic evidence, with a separate interpretation note; it is neither pooled with corrected draws nor used for v2 recipe ranking. Corrected development is now running. The ten-cell v2 confirmation plan is prepared under `var/artifacts/modebench_level3_v2/confirmation/`, with no confirmation seal, claim, or submissions until all passing recipes and fresh Level3 data exist.


The corrected initial Graph development fit is complete and fails: baseline pass@1/pass@8 is 0.204345703125/0.546875; the best forecast is 0.1627025075/0.4289536830 and its sole selected set is 0.171630859375/0.44921875. The immutable failed recipe is `var/artifacts/modebench_level3_v2/recipes/graph_coloring.json`; no alternate weights or ordering seeds are tried on these pools. Countdown and the other corrected domain runs continue.

Development-only regrading identifies a concrete Graph candidate issue: about 86.6–86.7% of outputs for the easiest support-4 and support-6 cells have the wrong length. Those cells use two hidden vertices, while three-color answers are common. A new isolated Graph v7 range is being prepared with exactly three hidden vertices and simple visible anchors, preserving color-label symmetry, the original prompt/verifier, exact support cells, quota-prefix stability, and all-history exclusions. The route changes only through an explicit registered candidate revision; the original 358 sealed files and completed baseline receipt remain intact. The finalizer and future confirmation preflight remain unsealed and are being extended to authenticate that additional generator provenance and its RNG exclusions. No Graph v7 GPU jobs or v2 confirmation jobs have been submitted.


Corrected Countdown now passes both forecast and selected development gates: weights `[0.05, 0.20, 0, 0.75]`, baseline pass@1/pass@8 `0.016845703125/0.119140625`, selected `0.0234375/0.115234375`. Both four-operand jobs finished normally; no timeout recovery was needed. The saved recipe reproduces exactly through the finalizer after fixing an in-memory-vs-JSON dictionary-key comparison (integer tier keys serialize as strings). This only canonicalizes exact JSON equality; fitting, ordering, outcomes and tolerances remain unchanged.

Graph v7 is now registered and submitted. Its immutable candidate protocol is `var/artifacts/modebench_level3_v2/graph_v7/candidate_protocol.json` (SHA256 `f33cadbab7b5c8cdedb04b3505fcb6b059fabd3d217efa26c21e5f561cc23255`). Four 128-row pools at `var/data/modebench_level3_calibration_graph_v7/pools/graph_coloring` pass 24 focused generator tests, 3,384 original-grader development witnesses, and 2,560 globally disjoint ephemeral capacity witnesses. No final evaluation dataset was generated by these capacity checks. The first three presets use five vertices and three hidden colors with sparse anchors or simple hidden forests; the fourth uses six vertices and three hidden colors as a harder bracket. All color and vertex labels are sampled symmetrically.

The four jobs are 31147184, 31147186, 31147187 and 31147188, recorded in `var/artifacts/modebench_level3_v2/graph_v7/development_jobs.json`. Their seal SHA256 is `a66a6e172bda3b294a8efba7ed3706f834b72f241cbae063bc28dcb72ba2fb32`: 503 source/input pins, all four pool certificates revalidated, the exact completed corrected baseline reused, and 2,048 additional request blocks disjoint from all original 12,544. The launcher passes 76 guard tests. New generator/adapter/registration files are frozen alongside the original 358 files; finalizer/auditor/confirmation code remains unsealed until all five passing recipes exist.

The explicit revised recipe route reuses the original full-pool fitting objective, grid, selection seed and both gates. Its registration authenticates the materializer and recorded historical source bytes. The final dataset snapshot and confirmation preflight include all revised generator inputs and every actual fitted candidate-pool RNG stream, including zero-weight tiers. The unsealed confirmation-plan extension is documented in `var/artifacts/modebench_level3_v2/confirmation/registered_candidate_extension_amendment_20260908T2135Z.json`; confirmation tasks, draw labels, fixed controls, hardware and tolerances remain unchanged. The integration review passes 90 tests. No corrected confirmation outcomes exist yet.


The corrected original Python range also fails development: baseline pass@1/pass@8 `0.237060546875/0.845703125`; the best forecast is `0.31394490383748197/0.8177737193362193`, and the sole selected set is `0.31396484375/0.81640625`. Weights `[0,0,0.95,0.05]` fail only the pass@1 gate. The unchanged result is preserved at `var/artifacts/modebench_level3_v2/recipes/python_factors.json` (SHA256 `77f681f5efd3ca3b3d7534183111f675c9d11e3ff5a2bc116a5688a4f8374b62`). A new isolated Python v5 candidate range is being designed from corrected development diagnostics, focusing on input parity and factor structure to reduce repeated successes while retaining pass@8. No alternate weights/order seeds are tried on the failed range; thresholds and the evaluator remain fixed. Its future registration namespace is `var/artifacts/modebench_level3_v2/python_v5/`. The finalizer now dispatches explicitly registered Graphv7 or Pythonv5 recipes and reauthenticates their recorded sources; all97 relevant integration tests pass, with all503Graphv7sealed files unchanged. No Pythonv5 GPU jobs or corrected confirmation jobs exist yet.

### Independent development update, 2026-09-08T22:38:09.134234+00:00

Graph v7 completed all four jobs and passes the unchanged full-pool and selected-set development gates. The selected set scores 0.207763671875 per attempt and 0.55078125 pass@8 versus Level1 baseline 0.204345703125 / 0.546875, using weights [0.20, 0, 0.75, 0.05]. An independent audit exactly reproduced the saved recipe and four pools, regraded all 16,384 attempts without discrepancies, and authenticated all 646 current frozen file pins. Countdown also passes. Fresh confirmation is still required.

Python v5 was registered, structurally audited, sealed and submitted as jobs 31147696–31147699. Its immutable seal is `3b9f126eb04ddd42ae9aba78b15fe69273f50c74c52a2c6f6b5f3807c518e931` and contains 646 file pins. The four new pools contribute 2,048 independent request blocks disjoint from all 14,592 prior blocks. The consolidated submission ledger is `var/artifacts/modebench_level3_v2/python_v5/development_jobs.json`. The failed original Python recipe remains unchanged.

MathIR and Pantry have encountered scheduler preemption; automatic resumes retain the original job IDs and authenticated batch caches. MathIR d2/d3 retain 32/35 validated batches after their second preemptions. Saved logs and checkpoint hashes are preserved before restart truncation. A prospective attempt to change six pending jobs to the regular partition was rejected by the cluster submission policy in every case; no job settings changed and no replacement jobs were submitted.

### Full development inventory in prospective confirmation, 2026-09-08T22:51:05.067620+00:00

A readiness review found that the prepared confirmation chain did not retain every execution artifact from the latest Python revision, and the standalone auditor only checked confirmation RNG blocks against the five receipts used by each domain recipe. Before final data or confirmation outcomes exist, the launcher and independent auditor now inherit the complete 646-file seal and independently reconstruct all 33 development sources, including rejected original Graph/Python ranges. They reproduce the same 16,640 disjoint request blocks and require every one of the future 5,120 confirmation blocks to be disjoint from the full development inventory and all other confirmation sources. The final auditor additionally binds the canonical seal to its campaign execution claim and each receipt to its sealed source, model, schedule and output path.

The combined auditor/finalizer/confirmation-launcher suite passes 113 tests, and all 646 pre-existing frozen file pins remain unchanged. The prior plan is preserved; the updated unsealed plan SHA is `50b5346433f049d9cb2f824105a58e33b48fa27f6ca7effd38538bf6191c1df2`. No scientific settings or development selection rules changed. The prospective amendment is `var/artifacts/modebench_level3_v2/confirmation/full_development_inventory_amendment_completed_20260908T2252Z.json`.

### Explicit regular-queue recovery, 2026-09-08T23:25:28.615952+00:00

Repeated low-priority preemptions and long scheduling forecasts prevented the remaining calibration from finishing. The cluster forbids changing a submitted job partition; a scheduler dry run accepted the same request in the regular `all` partition, whose configured preemption mode is OFF. A separate prospective recovery protocol preserves the exact models, RTX6000 hardware, precision, renderer, verifier, sampling settings, task files, output paths and all 163 committed batches (10,432 attempts). The existing 646-file scientific seal and 33-source/16,640-block inventory remain unchanged. No sampling or fitting choices were changed.

The recovery implementation passed 85 focused guards and independent review. Its protocol SHA is `3edb67ca4d597cd78d8b3f87bf7abe58dc700f21bbf1b528d23c5a116e3bfae8`; its execution seal SHA is `65b3990674e8560d084f328a4e189709bfc99cd012e8e6620efd81879c8d2cab`, covering 1,065 immutable files. All original cell locks were held during a cancellation restricted to the eleven still-pending IDs. Every original job was confirmed CANCELLED and every committed cache byte was verified unchanged before any replacement submission. Replacement workers use their genuine scheduler IDs, acquire both original and new cell locks, and call the unchanged evaluator with resume enabled. Old worker claims and submission records remain unchanged.

The immutable recovery ledger is `var/artifacts/modebench_level3_v2/regular_queue_recovery/development_jobs.json` (SHA `bfcfc53886ecc0b44e52a8c5c8657a3e09d399d7b85f29bc5d6ed71c0516035d`). MathIR d2/d3 now use jobs 31149147/148; Pantry baseline/d0–d3 use 31149149–153; Python v5 d0–d3 use 31149154/155/156/158. All eleven submissions succeeded. Start forecasts changed after submission; these jobs were still pending at this update. No immediate start or final difficulty match is claimed. Recovery-aware monitors distinguish original and actual job IDs. Final confirmation will additionally pin the completed recovery execution attestation, while retaining the scientific source inventory unchanged.

### Fixed automatic continuation, 2026-09-09T00:05Z

The remaining eleven calibration jobs are queued behind CPU allocations on the only RTX6000 node. A reviewed fixed continuation now waits for their registered receipts. Its source is `ops/exp_scaling/continue_modebench_level3_v2.py`; the immutable continuation seal is `var/artifacts/modebench_level3_v2/continuation/seal.json` (SHA `304d74c6f88a24f9d84c18a599e6a178fb4e1335fcac57db14c5635529773e43`), with1,101 source/input pins. All236 combined continuation, runtime, fitting-owner, finalizer, recovery-audit and confirmation tests pass; the actual preflight authenticates both frozen models and exactly reproduces the accepted Countdown and Graph recipes. Readiness is recorded in `continuation_readiness.json`.

Review found that the old MathIR watcher could fit other completed domains because it called the broad progress helper with fitting enabled. It was stopped before any new recipe appeared and replaced by `watch_modebench_level3_mathir_v2.py`, which fits only the five registered MathIR receipts under an exclusive lock. The new MathIR owner and existing Python v5 owner remain responsible for those fits; the continuation owns Pantry fitting only. No GPU task, committed sample, candidate law, selection rule or threshold changed.

The detached continuation process is1855778, with immutable start evidence and durable logs under `continuation/`. It exact-refits all five fixed recipes before publishing the fresh full dataset, validates all15 splits, publishes the complete recovery execution attestation, and prepares/submits exactly ten registered confirmation cells. Before the final match audit it authenticates each actual worker claim, scheduler completion and GPU/backend runtime. Missing receipts after completed execution, failed development gates, source changes, or an ambiguous interrupted phase stop progression and preserve the evidence. It does not choose replacement candidates, retries, weights, subsets or seeds.

Only Countdown and Graph have passing corrected development recipes. MathIR, Python and Pantry remain unresolved, and no final Level3 v2 data or corrected confirmation result exists yet. Running the continuation is not evidence of a difficulty match.

### Corrected MathIR development passes, 2026-09-09T00:30Z

Both remaining MathIR jobs completed normally through the sealed regular-queue recovery. The sole MathIR fitting owner published `var/artifacts/modebench_level3_v2/recipes/mathir.json` (SHA `2ee513ecec4967478b7c88b8d5b87009efc50c9b6a1f73583975cf10d711e0cc`) with fixed weights `[0,0.05,0.50,0.45]`. The Level1 baseline is0.04833984375 pass@1 and0.265625 pass@8; full-pool forecast is0.06843185424804688/0.223968505859375; the sole selected128-row set scores0.05517578125/0.193359375. All four unchanged gates pass. The selected pass@8 difference is−0.072265625, within the0.08 tolerance; no alternate selection or weights were tried.

The independent review exactly reproduces the recipe and finalizer validation, authenticates all1,101 frozen inputs and both recovered execution chains, and regrades all20,480 attempts from the five MathIR receipts without discrepancies. Its immutable artifact is `var/artifacts/modebench_level3_v2/mathir_completed_review/independent_completed_development_audit.json` (SHA `85337c4d47c2eb4bef854833fcfc42f8e83e0b6121fd6328fa02a4c3103c58c4`). MathIR joins Countdown and Graph Coloring as development-passing domains; fresh held-out confirmation remains required.

The Pantry baseline also completed and all2,048 attempts independently regrade correctly, with all28 original cache batches retained. Its public receipt briefly lagged the completed scheduler record on the shared filesystem; a later recheck authenticated the same canonical receipt, so no job was rerun and both observations are preserved. Pantry candidate sampling has resumed, while the remaining Pantry and Python candidates await scheduler allocations. The fixed continuation remains running and waits only for those two recipes before final dataset construction and confirmation.

### Corrected Pantry development passes, 2026-09-09T01:03Z

All five Pantry recovery jobs completed0:0. The fixed continuation fitted the registered pools once and published `var/artifacts/modebench_level3_v2/recipes/pantry.json` (SHA `4d7ec52ec2ccb10fd57d0301baa83cb3856fe93cde5e98e4d4a06acbd29837c4`) with weights `[0.55,0.40,0.05,0]`. Baseline pass@1/pass@8 is0.05078125/0.22265625, full-pool forecast0.0583251953125/0.20833333333333334, and selected development0.0625/0.21484375. All four unchanged gates pass; the sole selected set preserves the exact joint support/family cells.

The independent review exactly reproduces the fit and finalizer validation, authenticates all1,101 frozen inputs and five recovery chains, and regrades all18,432 attempts without discrepancies. It binds the driver-owned fit journal and the resolved baseline receipt-visibility evidence. Report: `var/artifacts/modebench_level3_v2/pantry_completed_review/independent_completed_development_audit.json`, SHA `fb6d1b34971d96fa48e3da2cbb523c3760eb01c185e4a5d4a05d0c3bb38a3d05`.

Four domains now pass development; Python remains unresolved. All four Pythonv5 jobs have started. The sole Python fitter and separate independent reviewer wait for their complete registered receipts. Final dataset construction and fresh confirmation remain gated on the fifth passing recipe; no final match is claimed.

### Python v5 fails; prospective v6 research, 2026-09-09T02:03:17.261227+00:00

All four Python v5 jobs completed. The sole fixed fit produced weights `[0.05,0.65,0,0.30]`, forecast pass@1/pass@8 `0.2796001483867695/0.758847261679293`, and selected `0.283935546875/0.76953125`, versus baseline `0.237060546875/0.845703125`. Both forecast gates and selected pass@1 fail. The immutable failed recipe SHA is `6431f9042fdd87c2614d10609a6dcd1cf95ab1f7f6328c49a093a1a3636185b8`. No alternate weights, subsets or seeds are tried.

The fixed continuation terminated correctly with `needs_calibration_revision` at01:20:57UTC. Its1101 inputs and all phase records remain frozen. Four other domains pass development. All eleven recovered jobs finished0:0; independent complete execution grading is underway. New Python candidate laws will be prospectively registered in isolated sources after capacity and distribution checks. No final data or fresh confirmation exists yet.

### Prospective Python v6 minimum-case bands

Four fresh128-row development pools are registered under `python_v6/candidate_protocol.json`, SHA `15885c03ec9cdedb0624fb80aded112d21edaa9a09134f5e52c7315fd81a131f`. The laws use four distinct numbers with smallest proper divisor at most5 and at least two proper divisors. Presets constrain the minimum to60–69 with maximum192 or384, then broaden the minimum to60–79/max192 and60–89/max384. Uniform joint support/minimum-band profile tickets preserve every eligible case set’s probability within a support cell; the original prompt, verifier and support distribution are unchanged. All60 support cells pass capacity, with2560 ephemeral full-split witnesses and6144 original-grader witness calls. An independent reconstruction exactly reproduces all512 published rows. These are hypotheses from completed development diagnostics, not passing accuracy results; calibration has not yet started.

### Python v6 sealed and sampling

All four fresh Python v6 calibration jobs were submitted as31151405–31151408 and started on node007. Their immutable1145-file scientific/execution seal SHA is `7f3a49ae3314ef314accac14be7668c920454b8d9206ca07026e81c772a0f266`; all inherited sources remain unchanged. The independent RNG inventory now contains37 sources and18,688 disjoint request blocks. The235 combined development guards pass. The sole fitter PID948742 and independent completed-run auditor PID950431 wait for their registered full receipts; neither chooses new weights or subsets after a failed selected set.

The future v6 finalizer/auditor/confirmation integration passes all171 guards and independently reproduces the complete37-source inventory. It preserves the original fixed Level1 controls, 5120 fresh confirmation blocks, unchanged matching gates, and requires a passing completed-v6 execution proof before admission. Four other domains still pass development. No new Python accuracy result or final match is claimed.

### Reviewed v6 continuation is running

The isolated replacement continuation started asPID953090 after independent post-seal review. Its seal `continuation_python_v6/seal.json` SHA `70f7cc8451fd253ae2c1cfecf6dc8ed72e198ee0b791035d9436484a00c171a0` freezes1186 files and64 inventories, preserving all earlier evidence plus the reviewed new pipeline and both owner startup records. All50 driver guards pass. The four Python jobs are sampling; the fitter and independent auditor wait for complete registered receipts. The driver requires allfive passing development recipes and full Python execution/grading proof before generating3200 final rows and submitting the ten fresh confirmation cells. It stops on failed gates or ambiguous execution and never selects a replacement subset, seed or weights.

### All five development recipes pass; final data published

Python v6 passes both forecast and selected-set gates with fixedweights `[0.15,0.80,0,0.05]`. Baseline pass@1/pass@8 is `0.237060546875/0.845703125`; forecast is `0.2449742260777417/0.82985730632215`; the sole selected set scores `0.251708984375/0.826171875`. Recipe SHA `18c32165c2b755092245d27b4e49b5c05d673323de7d69fd8dddc5ebe64f3486`. Its independent full audit replays all20,480 original-grader attempts without discrepancy, authenticates all four actual jobs and exactly reproduces the recipe; audit SHA `ad6b204e7b6c5a63ac656422b3508f3f190b8681008c372d3ed031b8e1a9d523`.

The fixed continuation has now published `var/data/modebench_level3_matched_v2`, containing384 train/128 dev/128 eval rows in each of five domains,3200 total. Dataset identity SHA `78f88b29e8b12315a9f33e7c7674453ca89017dbe3d90122cf0c457b7d1b8bcc`. Final structural authentication is running before confirmation submission. Development passage and dataset publication do not establish the fresh held-out match.


### 2026-09-09T03:18:12.387422+00:00: fresh five-domain confirmation submitted

All ten fixed confirmation cells were submitted once as jobs 31151637–31151646 after independent development and final dataset audits passed. Both Graph Coloring jobs completed normally; the remaining cells are sampling or queued. The sole continuation PID 953090 will authenticate completed execution and run the final audit. The 2,232-file confirmation seal is `5c9cae0c3d5509db20b8c22418da759710b2a9540eac74cc81ce8e743adca79c`; independent RNG review verifies 190,464 disjoint child seeds across all registered DEV and CONF sources. Fresh Level3 confirmation remains pending; fixed Level1 controls are reused with fresh independent draws under the prospective amendment.


### 2026-09-09T04:17:50.467757+00:00: V2 fresh confirmation failed; prospective two-domain revision

All ten jobs completed0:0. Final report SHA fd77c76274b81c73bfb78626666cb4078d4b3968eecd5cac3b05f9e33c854b12 has complete valid evidence. Independent original-grader replay of40,960 attempts agrees exactly (proof SHA62c112aa891d47f4e5db359f62d5fb1047235d64caeea55cd2c7b8cf72b4c6d5). Countdown, MathIR and Pantry meet both unchanged tolerances. Graph pass1 delta is -.04736328125, beyond .04; Python deltas are +.041748046875 and +.080078125, beyond .04/.08. The failure is preserved, including full dataset and immutable summary artifacts/modebench_level3_matched_v2_summary.md.

The active goal continues with a prospective isolated V3 revision of Graph/Python, retaining the three passed domains and explicitly reporting adaptive second-round evidence. All five completed Level1 evaluation receipts will be frozen as empirical reference targets to avoid moving measured targets; candidate development outcomes alone will rank fresh mixtures, while old Level3 confirmation is diagnostic only. This changes the next-round estimand to a fixed measured Level1 benchmark and makes no claim of fresh Level1 evidence or statistical equivalence.
