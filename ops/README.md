# Experiment workflow

The active workflow is deliberately narrow:

1. `run_experiment.sh` is the public entry point and resolves model, data, and
   arm presets.
2. `train.sh` is its low-level OAT command builder.
3. `submit_countdown_comparative.sh` submits matched seeds and methods.
4. `run_countdown_comparative_eval.sh` evaluates final checkpoints.
5. `analyze_countdown_comparative.py` produces the paper regressions.

The Slurm entry point is `slurm/train_node302.slurm`.

## Storage defaults

The shared launcher decouples evaluation from durable storage. By default it
evaluates every quarter prompt epoch, writes one rolling resumable DeepSpeed
checkpoint per prompt epoch, exports model weights only at the terminal step,
and removes all optimizer checkpoints after successful terminal evaluation.
Failed or requeued jobs retain the latest recovery state, and Slurm requeues
reuse a stable per-job attempt directory. Override these defaults only with
the explicit `OAT_ZERO_EXPORT_*`, `OAT_ZERO_RESUME_*`, or
`OAT_ZERO_PRUNE_RESUME_ON_SUCCESS` variables; the legacy `SAVE_*` variables
exist for frozen source compatibility.

## Evaluation defaults

ModeBench sampled evaluation uses four fixed K=8 draws (seeds 1001--1004) at
every evaluation boundary. The headline pass@8, mean@8, coverage@8, and
distinct@8 values are arithmetic means across those draws; SD, SE, minimum,
maximum, and all four raw values are logged separately. The durable
`eval_mode_coverage_draws.jsonl` sidecar retains each prompt, reference,
generated response, reward, normalized answer, and draw seed. It also retains
a dedicated deterministic K=1 greedy trace for pass@1; pass@1 has no Monte
Carlo error bar. The compute-divergence figures show all five outcomes without
smoothing; four-draw points have raw dots and range whiskers, while legacy
single-evaluation points are labeled as lacking repeated-draw uncertainty.

Supported variants are `grpo`, `xdr`, `xdr_tau_control`, `xdr_sac_dual`,
`maxent`, `maxent_control`, `maxent_dual`, `xdr_adapt`, `grpo_entropy`, and
`seed`. The landed comparison uses the first two. The next two control
signed-surrogate xDr. The three `maxent*` variants optimize the direct
on-policy sequence-MaxEnt objective at fixed, proportional-feedback, or
Haarnoja-dual entropy coefficient. The remaining variants are controls.

New `maxent_dual` runs default to `OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7`.
The controller applies log-alpha Adam to the EMA entropy error while retaining
the instantaneous entropy in telemetry. A decay of zero is the explicit
legacy behavior. The EMA value and decay are checkpointed, and checkpoints
from the old instantaneous rule fail closed instead of resetting the sensor
silently.

Dataset generation and exact coverage evaluation are handled by:

- `make_exact_countdown_mode_data.py`
- `make_exact_answer_mode_data.py`
- `make_python_factor_mode_data.py`
- `verify_python_factor_mode.py`
- `eval_exact_answer_mode_coverage.py`
- `slurm/eval_answer_mode_coverage_node302.slurm`

`make_python_factor_mode_data.py` materializes ModeBench's third environment.
Each answer is a bounded one-line Python lambda. The verifier first rejects
unsafe AST nodes, then calls the function in a killable isolated interpreter;
the executed output vector is both the correctness witness and canonical mode.
The deterministic 384/128 split has exact prompt-local mode counts and two
externally certified distinct modes per row.

The compute-scaling launchers, parser, and run log are in `exp_scaling/`. Its
analysis plan is still marked as a draft, so current curves are exploratory.
`exp_scaling/plot_collapse_telemetry.py` is the single source for the 3B
token-entropy, informative-group, and coverage--entropy figure plus its
machine-readable snapshot summary.

`exp_scaling/launch_e4_tau_control.sh` is the scale-aware E4 entry point for
the matched Dr.GRPO/fixed-xDr/proportional-feedback experiments. The maintained
grid is 0.5B, 3B, and 7B; 7B uses a two-GPU layout with CPU optimizer and
activation offload. E4 is a live exploratory campaign. Its analysis and
post-hoc provenance disclosure live in
`../paper/preregistration/e4_tau_control.md`.

`exp_scaling/launch_haarnoja_dual_extension.sh all` submits only the fourth
Haarnoja-dual arm for all six environment/scale cells. Existing controls are
reused. `exp_scaling/refresh_haarnoja_dual_curves.py` refreshes its six tidy
artifacts before `plot_divergence.py` rebuilds the four-method grid. The frozen
prospective E5 specification is
`../paper/preregistration/e5_haarnoja_dual.md`.

`exp_scaling/launch_on_policy_maxent_extension.sh smoke` runs E11's mandatory
32-step literal `alpha=0.05` standard-MaxEnt gate. Validate it with
`exp_scaling/check_e11_standard_maxent_smoke.py`; only a subsequent gated
three-arm smoke may authorize the 54-run replacement grid. Refresh curves with
`exp_scaling/refresh_on_policy_maxent_curves.py`. The frozen prospective
specification is `../paper/preregistration/e11_standard_sequence_maxent.md`.
The completed literal gate failed its length/EOS guard, so the adaptive smoke
and analytical grid remain blocked.
E6/E7 remain retired candidate objectives, E8 remains a failed sampled-
advantage diagnostic, and E9/E9b/E10 retain their historical traces.

E16 is the completed E15-derived 0.5B finite-action replication; E17 is its
live 3B continuation. Their entry points are
`exp_scaling/launch_e16_canonical_maxent_replication.sh` and
`exp_scaling/launch_e17_canonical_maxent_3b.sh`. E18 and E19 add the shared
matched canonical Dr.GRPO (`alpha=0`) controls at 3B and 0.5B, respectively,
using `launch_e18_canonical_drgrpo_3b_control.sh` and
`launch_e19_canonical_drgrpo_05b_control.sh`. `make e18-config` and
`make e19-config` replay both control configurations without submitting jobs.
Each control is shared by fixed, proportional, and dual MaxEnt within a
task/seed; historical free-text Dr.GRPO is not a matched control because it
changes the policy support. The E18/E19 comparisons are post-hoc and must be
reported as exploratory.

E23 and E24 extend the same frozen canonical treatments to 7B Countdown and
graph coloring, respectively. Their launchers are
`exp_scaling/launch_e23_canonical_maxent_7b_countdown.sh` and
`exp_scaling/launch_e24_canonical_maxent_7b_graph.sh`; both stage all nine
method/seed jobs held, audit the cohort, and only then release it to Slurm.

After a smoke cell terminates, first run
`exp_scaling/audit_e16_canonical_endpoint.py` on its update-32 checkpoint,
then use `exp_scaling/check_e16_canonical_smoke.py validate-cell` with the
campaign identity, three-arm manifest, metrics, stdout, endpoint audit, and
terminal Slurm state. Once all six results exist, the same tool's `approve`
command replays their raw evidence and writes the immutable approval consumed
by `full`. `exp_scaling/verify_e16_smoke_approval.py` replays that evidence
again at Stage-R authorization; a hand-written pass status is insufficient.

`exp_scaling/check_e4_7b_smoke.py` validates only step count, finite training
entropy/controller diagnostics, and checkpoint structure. It deliberately
does not inspect evaluation outcomes or compare smoke arms.

Watch the current E51 cohort with:

```bash
make monitor
```

The dashboard contains only Countdown, graph coloring, and Python-factor E51
matched Dr.GRPO/policy-entropy runs at paired seeds 43/44/45. It refreshes
every 30 seconds, while a non-overlapping background task reparses only the
current E51 curve artifacts and atomically rebuilds
`../paper/figures/e51_current_canonical_05b_live.png` every 60 seconds. The
earlier `e45_e51_current_canonical_05b_live` path is republished as an identical
compatibility alias. `READY` means the frozen configuration exists but no
matching manifest, allocation, or metrics are discoverable yet. The view keeps
the furthest persisted evaluation ("landed") separate from the current
attempt, so requeues do not erase progress. Completion is capped at 50 passes.
Press Ctrl-C to close the dashboard; the jobs continue normally, and the same
command reconstructs the view later. Use
`python3 ops/exp_scaling/monitor_campaign.py --current-canonical-only --once`
for a single snapshot, or `make current-canonical-figure` for a one-shot curve
and figure refresh.

E22's 0.5B free-form conditional-token Haarnoja-dual rows are included in the
same dashboard. Their curves appear as a separately labeled purple dashed
line in the combined compute-divergence figure and both relevant task splits;
the canonical-only manuscript trajectory figure remains unchanged. Run
`make e22-figures` for a one-shot curve and figure refresh.

E25 adds the 3B scale extension of E22-v2's base-preserving free-form
conditional-token Haarnoja dual for Countdown and graph coloring. Its launcher
is `exp_scaling/launch_e25_modebench_freeform_dual_3b.sh`. Post-hoc E28 adds
fresh matched 3B free-form Dr.GRPO controls through
`exp_scaling/launch_e28_modebench_freeform_drgrpo_3b_control.sh`; the contrast
remains exploratory because E25 had begun before E28 was frozen.

## Prospective MATH environment

`math500/import_oat_math.py` imports the byte-exact math artifacts released
with the original OAT Dr.GRPO paper. It pins the initial upstream commit and
SHA-256 values, materializes the level-3--5 training DatasetDict under
`var/data/oat_drgrpo_math_paper/train`, and materializes MATH-500 as the sole
held-out `math` split under `var/data/oat_drgrpo_math_paper/eval`.

```bash
python ops/math500/import_oat_math.py
python ops/math500/import_oat_math.py --audit-only
```

The prospective canonical-MaxEnt design and its no-launch boundary are in
`../paper/preregistration/e20_math_canonical_maxent.md`. MATH-500 has
variable-length symbolic answers, so the existing three-action graph and
Countdown codecs cannot be reused. Dataset readiness does not authorize a
free-text or multiple-choice substitute under the E20 identifier.

E64 provides a separate, smoke-gated transfer experiment without claiming
free-form reasoning modes. It trains on the frozen first 384 MATH12K rows and
holds all 500 MATH-500 rows out for evaluation. The `math_verified_answer`
contract collapses every verifier-positive completion for a prompt to one
canonical `correct` key: verified-mass replay can anchor a discovered correct
solution, while the known-mode balance controller must receive zero eligible
observations.

```bash
# Dry-run both contracts without submitting.
bash ops/exp_scaling/launch_e64_math500_realism_smoke.sh config
bash ops/exp_scaling/launch_e64_math500_realism_matched.sh config

# Inspect the one-seed, 96-update smoke.
python ops/exp_scaling/audit_e64_math500_realism_smoke.py
python ops/exp_scaling/audit_e64_math500_realism_checkpoint.py

# The six matched 12-pass jobs fail closed until that audit passes.
bash ops/exp_scaling/launch_e64_math500_realism_matched.sh full

# Once launched, refresh the six-run audit and full live figure.
python ops/exp_scaling/audit_e64_math500_realism_matched.py
python ops/exp_scaling/plot_e64_math500_realism.py

# Or keep both refreshed every 60 seconds.
bash ops/exp_scaling/watch_e64_math500_realism.sh
```

The frozen protocol is
`../paper/preregistration/e64_math500_realism_transfer_05b.md`.

## E59 executable MathIR extension

E59 is a separate mathematical ModeBench extension of E58's latest
verified-first global-replay algorithm. The model emits only prompt-local
action IDs; the MathIR interpreter executes those exact transformations and
derives the canonical key from the resulting normalized state path. The
384/128 split has five exhaustively enumerated valid modes per prompt.

```bash
# Replay the frozen smoke and matched configurations without submitting.
bash ops/exp_scaling/launch_e59_mathir_global_replay_smoke.sh config
bash ops/exp_scaling/launch_e59_mathir_matched.sh config

# Inspect the running/terminal one-pass smoke.
python ops/exp_scaling/audit_e59_mathir_global_replay_smoke.py

# Submit the six-job Dr.GRPO-versus-E58 cohort only after a passing smoke.
bash ops/exp_scaling/launch_e59_mathir_matched.sh full
```

The protocol is
`../paper/preregistration/e59_mathir_global_verified_replay_05b.md`. The
broader untouched-model family-coverage probe failed and remains recorded as
failed; E59 is therefore explicitly exploratory and smoke-gated.
