# ModeBench cross-level population and protocol options

Prepared 2026-09-11 from current local sources. Planning/read-only audit only: no manuscript changes, reserve generation, weight restoration, or evaluation/training jobs.

## Recommendation

Keep the main causal experimental unit the within-level replay contrast, separately for Dr.GRPO and MaxRL, at the registered terminal endpoint. Present Level 2 as replication/robustness in a second benchmark construction and protocol. Do not present a difference between native Level-1 and Level-2 terminal results as an isolated causal effect of difficulty. Put development admission and construction matching in the appendix, alongside the population/interface ledger. The hosted study is a separate descriptive model/protocol landscape; it does not identify RLVR as the cause of hosted concentration.

Suggested main language: “Replay is evaluated within each level under a common protocol for the four arms. Level 2 supplies a second construction and evaluation setting; cross-level differences also reflect changes in prompt populations, interfaces, and training distributions.”

Suggested appendix language: “Level-2 evaluation support-count histograms were matched to the designated E117 Level-1 confirmation reserve, whereas the native Level-1 terminal results use the historical test sets. Training-count histograms were matched to native Level-1 training. Consequently, the displayed terminal cross-level comparison is not a paired or isolated difficulty intervention.” Add the interface differences below. “Designated confirmation reserve” is a historical designation: these rows now have model outcomes and should not be called untouched.

## Verified population facts and sources

- `var/results/modebench_level2_r5_frozen_repeat1/admission_fairness_report.json`, `level1_reference` (around lines 885–911), binds native Level-1 training, E117 development, and E117 confirmation for Countdown/Graph/Python/MathIR; Pantry uses native datasets. Its `domains` and histogram checks establish which matching actually happened.
- `paper/results/modebench_level_comparison_snapshot.json` binds terminal native Level-1 and Level-2 evaluation origins and the complete-domain admission policy. Each origin gives a JSONL path and line. Retain these exact admitted draw records, deduplicate per source policy, and never select a latest/best log ad hoc.
- Native Level-1 / Level-2 eval mean support counts: Graph 6.265625 / 6.0859375; Countdown 4.5234375 / 4.3671875; Python 229.4375 / 251.6875; MathIR 5 / 5; Pantry 18.1875 / 18.1875. Same mean does not prove equal distribution; matched histograms do not establish paired semantic identities.
- `paper/results/frontier_hosted_20260911.json`, `dataset_provenance_notes`, already records that E118 native L1 differs from E117 construction references. `paper/results/frontier_hosted_20260911_appendix.tex` explains unpaired prompts and differing interfaces. Training and hosted populations should remain independently identified.

## Verified protocol differences: affect BOTH training rollouts and terminal evaluation

Native Level-1 schedule source: `var/artifacts/e72_frontier_source_runs.json`, first domain blocks begin Graph:23, Countdown:1143, Python:2263, MathIR:3383, Pantry:4503. Its `inherited_eval_config` gives both train/eval generation caps, prompt template, canonical action task, prompt limit, context length. E78 imports these at `ops/exp_scaling/launch_e78_verified_replay_only_05b.py:75`; the inherited variables are exported at `ops/exp_scaling/launch_e72_b3a_replay_ablation.py:238`–265.

| Domain | Native L1 prompt | Native L1 train/eval response caps | L2 prompt / syntax | L2 train/eval response caps |
|---|---|---:|---|---:|
| Graph | qwen_boxed | 192 / 192 | qwen_boxed / none | 192 / 192 |
| Countdown | qwen_boxed | 192 / 192 | qwen_level2_countdown / countdown_legal_v3 | 192 / 192 |
| Python | qwen_boxed | 192 / 192 | qwen_level2_python_factors / domain_legal_v1 | 192 / 192 |
| MathIR | qwen_boxed | 64 / 64 | qwen_level2_mathir / domain_legal_v1 | 192 / 192 |
| Pantry | qwen_pantry_support_mask | 8 / 8 | qwen_level2_pantry / domain_legal_v1 (intended contract) | 192 / 192 (intended) |

Pantry is excluded from the four-complete-domain terminal mean. Original E119 held records inherited `canonical_action_task=pantry_support_mask`, which conflicts with the new Level-2 contract (`src/oat_drgrpo/args.py:555` rejects simultaneous canonical actions). Do not infer an admitted Pantry runtime from the original launcher alone; source-admissible repaired runtime would need separate audit. This does not affect the verified four-domain difference.

Native prompt/context limits: Graph/Countdown/Python 256/512, MathIR 256/384, Pantry640/704. E119 sets prompt/context1024/2048 for every domain. These are caps, not observed lengths; do not claim all prompts were truncated in L1 or that all increased budgets were used.

E119 source: `ops/exp_scaling/launch_e119_level2_qwen05b_factorial.py:38`–50 (prompts and syntax), :136–146 (train/eval data, common prompt/syntax, train192/eval192, prompt1024/context2048, trainingG16 and terminalK8×4,T1,p1). The actual held scheduler records in `var/artifacts/e119_level2_qwen05b_factorial_jobs.json` confirm these settings for all sampled domain controls. `snapshot_root` points to `var/artifacts/source_snapshots/e76_tuned_scale_50d36295558a8958`.

Runtime checks: current `src/oat_drgrpo/actor.py` and `templates.py` are byte-identical to archived E119 versions. Actor :584–599 copies terminal eval max_tokens; :655–662 applies guided syntax to mode-coverage evaluation; :918–924 applies it to training rollout. `templates.py:99` is the short boxed-only L1 system; :110–138 gives L2 solver hints (small-divisor/dispatch Python, explicit algebraic-isolation order MathIR, systematic Countdown). Archived `ops/train.sh:293`–324 maps prompt/syntax/train/eval caps to runtime args. Therefore this is not merely a development-admission difference.

## Tier 1: existing-data support-standardized sensitivity (no GPU)

Use source-admitted per-prompt JSONL terminal records. They retain prompt/reference identity, per-sample rewards/keys, per-prompt P,D,M and draw seeds. Join arms by exact prompt identity within each level and keep the four K8 draw groups; do not pool32 generations and then call that distinct8. Use exact certified task support count H from canonical dataset/reference metadata. **Python logged `answer_mode_count` is the placeholder1 in both levels**; its reference JSON `num_modes` contains the actual product count (e.g32). Reconstruct/validate H from `reference.num_modes` and dataset source, not the generic logger field. Avoid confusing enumerated-library support with unrestricted executable support, especially hosted Countdown.

For each domain choose one outcome-independent target distribution w over common support strata, then report mu_std(level,arm)=sum_h w_h E[Y|level,arm,H=h]. Compute replay contrasts within level first, then the descriptive difference of those contrasts across levels. Keep P and D primary; B=D-P counts additional observed modes and does not adjust away correctness. Preserve joint arm/metric/seed vectors in uncertainty estimation.

Graph and Countdown have all their observed exact H strata represented at both levels; MathIR has H=5 throughout. Python exact-H overlap comprises26 strata and retains117/128 native-L1 prompts and114/128 L2 prompts. These are from one source-admitted seed43 draw per level; verify invariant dataset identity across all arms/seeds before analysis. Nonoverlap count strata are real positivity failures: disclose excluded mass and change of target population. Do not extrapolate to unavailable strata. Coarse prespecified H bins can retain rows but leave within-bin composition mismatch, so label that compromise. Report stratum sizes, weight maximum, effective sample size, and native/unweighted estimates alongside standardized estimates. An overlap distribution such as normalized min(n1h,n2h) is an outcome-independent descriptive target; no benefit-selected trimming or weights.

This sensitivity addresses only marginal support-count composition. It does not remove task structure, prompt guidance, decoder constraints, context/response budgets, or separate training distributions; it cannot make cross-level difficulty causal. Because outcomes have already been inspected, call it retrospective sensitivity, not preregistered confirmation. Uncertainty must distinguish finite fixed-bank performance from superpopulation generalization and training-seed variability.

## Tier 2: reevaluate existing frozen checkpoints

### Minimum population-reference repair

Evaluate all four Level-1-trained arms, all five registered seeds, on the designated E117 confirmation reserve for the four complete domains. Use the original per-domain L1 evaluation law to retain interpretation of native L1 policy behavior, or explicitly adopt a new common law and label it; do not silently switch prompts/constraints. Compare with the already recorded Level-2-trained results on the histogram-matched Level-2 eval population.

Cost at128prompts×4draws×K8:80checkpoints,327,680responses, plus harness-bridge checks. This is evaluation only, no optimizer updates. It fixes the reference-population mismatch at the support-histogram level. It still compares different prompt instances, separately trained policies, and different level protocols; it does not identify a pure difficulty effect. Do not replace native historical results silently; retain them and report this as new held-out sensitivity.

### More informative crossed evaluation

A 2(train level)×2(eval population)×4(method)×5(seed)×4(domain) grid holds each trained policy fixed while it is evaluated on both task populations. Under one specified shared per-domain evaluation interface, this separates eval-population sensitivity from which training package generated the checkpoint.160checkpoints×2evalpopulations×4096responses=1,310,720responses if the full grid is freshly evaluated. Existing diagonal outcomes can be reused only if evaluation law/seed/provenance is exactly compatible; otherwise they are bridge references, not replacement cells.

The within-fixed-policy contrast estimates transfer/evaluation-population change for the specified task populations and interface. Differences between L1-trained and L2-trained policies on a common test bank estimate the full training-package difference, including training prompts/constraints/budgets, not isolated training difficulty. Same seed labels alone do not make task instances paired or remove these differences.

### Feasibility and endpoint identity

All80L1 and80L2 complete-domain arm×seed cells have local `MODEL_ARCHIVE.json` receipts. We verified these by deriving run dirs from the source-admitted terminal origins. All are status retired, with terminal exports named `step_03073`, whereas published terminal evaluations are step3072. This may be bookkeeping, but **must verify the export-to-evaluation step/weight relationship and reproduce a native evaluation bridge before claiming identical endpoints**. Do not automatically choose latest checkpoint.

Example receipts:
- `var/data/xdr_qwen25_0p5b_instruct_grpo_compute_matched_e78_replay_only_graph_control_s43/MODEL_ARCHIVE.json`
- `var/data/xdr_qwen25_0p5b_instruct_grpo_compute_matched_e119_level2_graph_drgrpo_s43/MODEL_ARCHIVE.json`

They include repo `od2961/maxent-grpo-models`, immutable commit, file SHA, archive verification receipt, original terminal export, and a restore command via `ops/archive_completed_models.py`. Example weight size988,097,824bytes each;80weights≈79GB total download if all restored, but sequential restoration can bound resident storage. Remote existence/access was not revalidated and nothing was restored.

`ops/eval_exact_answer_mode_coverage.py:194` accepts explicit checkpoint/data/decode controls but :240 constructs plain vLLM sampling, with no domain guided syntax. Its template options are limited; it is not a ready-made reproduction of the current four-domain training harness. `ops/evaluate_modebench_level2_viability.py:185`–188 hardcodes development splits and applies its admission profiles; it must not be used unmodified as a terminal confirmation evaluator. Reuse actor/verifier code in a frozen read-only inference harness; validate prompts, tokenizer, syntax, cap, dtype, seeds, batch behavior, keys and metric reconstruction against archived native rows. Lock inclusion and failure handling before new outcomes. Retain all intended cells and failures, never outcome-select seeds or checkpoints.

## The designated reserve is no longer untouched

Original contract: `paper/preregistration/e117_stage1a8_untouched_evaluation_reserves_20260825.md:14`–18 acknowledges historical eval reuse; :22–48 defines two128prompt banks and semantic disjointness; :76–81 reserves confirmation for a later protocol with fresh training seeds. The designated confirmation rows have since been exposed for construction matching and **evaluated on frozen models**.

Concrete evidence: `var/results/modebench_level3_v1/confirmation_05b_countdown.json:174` source is E117 confirmation/countdown/eval; :184 information_boundary says confirmation explicitly authorized and prompts loaded; :189 level1; generated2026-09-08T20:34:44Z and complete at :35556. Analogous complete Graph/Python/MathIR results exist. Later `var/results/modebench_level3_v2/confirmation_python_v6/confirmation_05b_countdown.json` records a further evaluation. These are frozen-base outcomes, not proof of training-optimizer contamination, but they rule out claiming this reserve has never been used or inspected.

Distinguish (1) training identity overlap, (2) metadata used in construction, (3) model-output-guided selection/tuning, and (4) reuse for inferential hypotheses. Audit these exposures rather than labeling all holdout data invalid. Reuse can still give useful performance sensitivity on identity-disjoint held-out tasks. To make a new genuinely prospective confirmatory claim, freeze hypotheses, checkpoint eligibility, interfaces, target weights, metric family and analysis first, then generate/seal a fresh identity-disjoint reserve from a frozen recipe. This is a new study; it cannot retroactively satisfy the original E117 fresh-training-seed contract.

## Tier 3: genuinely paired difficulty intervention

Construct paired easy/hard task variants from the same latent instance with a verified mapping of valid modes (and equal support where feasible). Freeze the explicit structural manipulation and all other interface/resource settings: template guidance, answer representation, admissible syntax, token budget, evaluationK,T,p and identity contract. Validate correctness and mode correspondence before seeing model outcomes. “Harder” is an operational hypothesis about that manipulation, not an abstract scalar assured by the level label or larger combinatorial support.

Evaluating the same frozen policies on both variants, with stateless randomized presentation/fresh sampling streams and paired prompt-level inference, estimates the effect of that specific evaluation-task manipulation. It does not estimate the effect of training on harder tasks.

To isolate training difficulty, run a new matched training factorial from identical initialization: training-task manipulation×objective(Dr.GRPO/MaxRL)×replay, fresh paired seed blocks, frozen curricula and budgets, and a common untouched evaluation bank (or both prespecified evaluation versions). Specify whether budgets match prompts, samples, tokens or FLOPs; changed response caps imply equal steps need not imply equal computation. Estimate prespecified replay×training-manipulation interactions with joint uncertainty. Existing E119 can support the effect of its complete construction/training/interface package, not attribution to only its difficulty component.

## Presentation priority

Main: problem/measurement; occupancy identity and why P,D both matter; controlled within-level replay effects; Level2 replication across another setting; hosted descriptive breadth concentration; conditional mechanistic theory with assumption boundary. Keep causal/diagnostic/descriptive evidence roles distinct. For Level2 main figure show replay effects within each level side-by-side rather than a slope claiming hardness caused larger benefit. A single concise caption note can identify different populations/interfaces and direct to the appendix ledger.

Appendix: dataset/version/split identity ledger; native vs reserve histograms and semantic disjointness; prompt/syntax/cap table; development admission; complete seed/arm terminal admission; existing-data standardization if undertaken; fresh reevaluation if undertaken; full proof and implementation correspondence. Native and sensitivity results remain visible together, and no retrospective patch is labeled an original preregistered result.
