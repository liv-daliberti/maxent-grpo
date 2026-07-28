# Exploration compute-scaling campaign — run log

Prereg: `paper/preregistration/exploration_compute_scaling.md`
Goal: turn "diversity at flat accuracy" into "group-level exploration is where RL
compute pays off" via E1 (compute-scaling divergence), E2 (two-stage), E3 (temp control).

## Launched 2026-07-16 (~00:15)

| Exp | Stamp | Model | Arms | Horizon | Node | Jobs | Status |
|-----|-------|-------|------|---------|------|------|--------|
| E1 pilot | `gce1_05b` | 0.5B | grpo, xdr τ0.05 | 16 ep (~3072 st) | node105 a5000 | 29997357–62 | RUNNING |
| E1 3B | `gce1_3b` | 3B | grpo, xdr τ0.05 | planned 16 ep; stopped at 2 ep | node302 a100 | 30000479–84 | CANCELLED intentionally (trend established) |
| E3 3B | `gce3_3b` | 3B | grpo T=1.2 | 16 ep | node302 | — | NOT LAUNCHED (staged; gated on E1) |
| E2 3B | — | 3B | two-stage | — | node302 | — | NOT BUILT (needs E1 epoch-1 ckpt) |

Decision 2026-07-16: leave agora untouched; 3B E1 starts as agora clears (no deadline).

## How to run / analyze

- Launch: `bash ops/exp_scaling/launch_e1_05b_pilot.sh` (or `_e1_3b`, `_e3_tempcontrol_3b`).
- Curve: `var/seed_paper_eval/paper310/bin/python ops/exp_scaling/parse_scaling_curve.py --stamp-prefix <stamp>`
  → writes `var/artifacts/<stamp>_scaling_curve.json` + prints coverage@8 divergence table.
- Metrics source: inline `<run_dir>/debug_*/train_metrics.jsonl` (no checkpoints needed);
  keys `eval/{multi,unique}_answer/sampled_{any_correct,mean,mode_coverage,distinct_correct}_at_8`
  and `.../accuracy` (greedy). E1 primary = multi_answer coverage@8 vs log(step) interaction.
- Watch: `squeue -u od2961`; per-run log `var/artifacts/logs/xdr_train-<jobid>.out`.

## Update 2026-07-16 (midday)

- BUDGET KNOB (the one that matters): steps = MAX_TRAIN / G. num_prompt_epoch is
  ignored when MAX_TRAIN binds. N pool-passes -> MAX_TRAIN = N x pool x G.
  First pilot ran 3 passes (not 16); first 3B submit would have run 1 pass —
  both corrected. Corrected jobs: 3B `gce1_3b` 30000479-84 (16 passes, running
  since 11:19, ~32 h); 0.5B extended `gce1b_05b` 30000485-90 (24 passes, running).
- PILOT RESULT (3 passes, decisive): baseline coverage@8 DECLINES .12->.07 while
  xDr climbs .11->.19; gap widens monotonically. 3B epoch-1 (from the 1-pass runs):
  gap +14.1 cov / +21.1 pass@8 / +0.79 distinct at flat mean@8.
- PAPER: new §Results "Longer Training: Collapse Versus Divergence"
  (sec:compute-scaling) + Figure `paper/figures/compute_divergence` (built by
  `ops/exp_scaling/plot_divergence.py`); abstract + Discussion updated; compiles
  clean (27 pp). Refresh when long runs land:
  `parse_scaling_curve.py --stamp-prefix gce1b_05b; ... gce1_3b; plot_divergence.py; (cd paper && make)`
  (also in a % NOTE above the subsection in main.tex).

- 2026-07-16 ~12:55: launched E1 domain-generality arm `cde1_05b` (0.5B Countdown
  easy3, 24 passes = MAX_TRAIN 147456, jobs 30000793-98, node105). Rationale: fig
  is GC-only; 0.5B Countdown has traction (pass@8 ~.40) so dynamics are testable
  there; 3B Countdown-4 is floor-bound (.073) so dynamics uninformative at 3B.

- 2026-07-16 ~13:35: 0.5B GC extended (gce1b_05b) CANCELLED at ~14/24 passes by
  user decision -- both arms fully converged at the collapsed floor (~.06
  coverage), so the remaining passes carried no information. KEY FINDING kept
  on disk (train_metrics.jsonl + gce1b_05b_scaling_curve.json): xDr DELAYS
  collapse ~5-10x in passes (gap peak +.127 @ ~1.3 passes -> ~0 by ~12-14)
  but does not prevent it under extreme repetition of the tiny 192-pool.
  "Delay, not immunity" at 0.5B; whether 3B (5x pool) closes too is the open
  question the running gce1_3b answers. Cancellation freed node105 for the
  0.5B Countdown queue (cde1_05b).

## Update 2026-07-16 ~14:50 (figure refresh #2)

- Re-parsed all stamps; rebuilt fig:compute-divergence (now 3 rows: 0.5B GC,
  0.5B Countdown NEW, 3B GC; 3B Countdown row auto-appears when cde1_3b lands).
- 0.5B Countdown (cde1_05b relaunch, thru 3/24 passes, 2 xdr seeds): NO baseline
  collapse; xDr early lead (+15.5 pass@8 / +5.0 cov @ 1 pass) dissipates by 2-3
  passes -> "buys speed, not a persistent gap". Sentence added to
  sec:compute-scaling (domain dependence, exploratory).
- 3B GC thru 1.5 passes (2 seeds): gap WIDENS +10.9->+14.5 cov, +16.8->+26.2
  pass@8, +0.59->+0.81 distinct; baseline declining from half-pass peak on all
  three metrics; mean@8/greedy gaps now positive (+4.3/+4.9, watch as more
  passes land). 3B paragraph updated (1.25 -> 1.5-pass snapshot).
- Paper compiles clean, 18 pp (post-restructure), no undefined refs.

## Next steps
1. Treat gce1_3b as an intentionally stopped exploratory trajectory, not a
   completed 16-pass endpoint; the frozen right edge is two seeds at 2 passes.
2. cde1_05b full 24 passes + cde1_3b (queued): refresh again; 3B Countdown row
   auto-appears.
3. Commit prereg (disclose launch-vs-commit timing honestly).
4. Build E2 two-stage (Phase-A ckpt -> Phase-B plain Dr.GRPO); E3 temp control.

## Update 2026-07-16 (3B collapse telemetry snapshot)

- Consolidated the ad hoc diagnostic and phase plots into
  `plot_collapse_telemetry.py`. It writes the paper PDF/PNG and
  `paper/results/collapse_telemetry_3b_summary.json` from the longest coherent
  metrics log for each seed.
- Through the current common telemetry step 1471 (1.44 passes; two seeds per
  arm at the right edge), the 25-step-smoothed token entropy is .087 for
  Dr.GRPO and .165 for xDr; the informative mixed-group fraction is .577 vs
  .885. Dr.GRPO first remains below entropy .20 at step 209, versus step 872
  for xDr (4.2x later).
- In the narrow observed entropy overlap [.179, .218], coverage@8 averages
  .224 for two Dr.GRPO observations and .282 for seven xDr observations
  (+.058). This is explicitly descriptive: observations occur at different
  steps and do not identify a causal reallocation of entropy to modes.
- xDr's effective aggregation count remains mild (30.6/32 at the snapshot),
  so there is no telemetry evidence that its continuing token-entropy decline
  is driven by progressively stronger candidate-weight concentration.
- Paper language is now "collapse resistance, not collapse immunity." The
  0.5B extended run closes its peak +.127 coverage gap by 12--13 passes; the
  running 3B campaign determines whether the larger pool ultimately does the
  same.

## Staged 2026-07-16 (E4 entropy-feedback tau; not launched)

- Added a label-free xDr controller that runs at tau=.05 for 64 globally
  averaged entropy observations, targets 80% of the warmup mean, and lowers
  candidate-aggregation tau toward .005 only below target (EMA .9, gain 20).
- Staged matched 1.5B and 3B graph-coloring recipes with exactly three arms:
  Dr.GRPO, fixed xDr tau=.05, and entropy-feedback xDr; three seeds, G=32,
  16 passes, identical evaluation and checkpoint cadence.
- This intervention was designed from the already observed 0.5B/3B telemetry.
  `paper/preregistration/e4_tau_control.md` records that post-hoc provenance
  and freezes the exploratory outcomes before launch.
- Launchers: `launch_e4_tau_control_1p5b.sh` and
  `launch_e4_tau_control_3b.sh`. Neither has been executed.

## Staged 2026-07-16 (E4 7B scale amendment; not launched)

- Added Qwen2.5-7B as the next same-family scale after 3B; using an unrelated
  model near 5B would confound model family with scale.
- Consolidated E4 behind `launch_e4_tau_control.sh`; the 1.5B, 3B, 7B-smoke,
  and 7B wrappers now resolve through the same three-arm configuration.
- The 7B smoke runs dedicated non-analysis seed 9001 for each arm over 256
  updates, crossing the controller's 64-update warmup. Full 7B submission
  requires explicit `OAT_ZERO_7B_SMOKE_APPROVED=1` after operational checks
  only.
- 7B requests 192 GB host memory, CPU optimizer/activation offload, and retains
  a rolling four-checkpoint window. At this staging point no E4 job had been
  executed.

## Update 2026-07-16 ~15:05 (figure refresh #3, divergence_grid_latest)

- Refreshed var/artifacts/divergence_grid_latest.png (= copy of the rebuilt
  compute_divergence.png; the "latest" viewing convention). Paper rebuilt.
- 3B GC thru 1.75 passes (2 seeds): gap off its 1.5-pass peak but still wide
  (+12.1 cov / +22.5 pass@8 / +0.68 distinct); mean@8/greedy blips at 1.5
  regressed to ~0 at 1.75 (single-draw noise — correctly kept out of prose).
  Text snapshot synced 1.5 -> 1.75 passes ("remains wide, after a 1.5-pass peak").
- 0.5B Countdown thru 4 passes: still converged (gap ~0-+3 pts) — transient-lead
  reading unchanged.

## Stopped 2026-07-16 ~15:46 (E1 3B graph coloring)

- User-directed early stop: the qualitative divergence was already clear, so
  the remaining planned 14 passes no longer justified the compute.
- Cancelled only corrected `gce1_3b` jobs 30000479--84. Jobs 30000479--83 had
  started; 30000484 (xDr seed 45) was still pending. Countdown jobs were left
  untouched.
- Frozen curve: all three matched seeds through 1 pass; seeds 43--44 through
  2 passes. At the final two-seed point, xDr--Dr.GRPO is +13.1 coverage points,
  +23.4 pass@8 points, and +0.72 distinct modes. This is a deliberately stopped
  exploratory trajectory, not a completed long-horizon or confirmatory result.
- Re-parsed `var/artifacts/gce1_3b_scaling_curve.json` from the last written
  metrics. Do not relaunch this stamp without an explicit new decision.

## Update 2026-07-16 ~15:58 (all-environment refresh)

- Re-parsed all four compute-dynamics cells and rebuilt
  `compute_divergence.pdf`: 0.5B/3B x Countdown/graph coloring.
- 0.5B Countdown now has a seven-pass common horizon (three baseline, two xDr
  seeds). Its coverage gap is +1.1 points there; the +7.0 pass@8 gap tracks a
  +7.7 mean@8 gap rather than greater answer-set breadth.
- 3B Countdown has landed only its initialization evaluation: three baseline
  and two xDr jobs, with coverage .051 vs .048. The figure shows these points
  on non-magnified axes and labels them `INITIAL EVAL ONLY`; no post-training
  checkpoint or trajectory has landed yet. The sixth job (xDr seed 45) remains
  queued.

## Launched 2026-07-16 ~16:10 (E4 7B operational smoke)

- CS A6000 availability confirmed on `node206`; added explicit Slurm
  partition/account overrides to the common comparative launcher.
- Submitted the outcome-blind seed-9001 smoke only, not the analytical 7B
  campaign: Dr.GRPO 30001480, fixed xDr 30001481, feedback xDr 30001482.
- Placement is `cs` / `allcs`, `node206`, one `gpu:a6000` and 192 GB host
  memory per job, with a 12-hour limit. Dr.GRPO and fixed xDr started
  immediately; feedback xDr is resource-pending because only two 192 GB jobs
  fit concurrently under the node's scheduler allocation.
- Full seeds 43--45 remain gated. Run `check_e4_7b_smoke.py` only after all
  three arms cross the 64-update warmup and write a loadable step-256
  checkpoint; approval must not inspect outcome differences.

## Blocked 2026-07-16 ~16:20 (E4 7B single-A6000 smoke)

- Dr.GRPO 30001480 and fixed xDr 30001481 both failed before training with
  `RuntimeError: vllm cannot load the model` on one A6000. Neither wrote a
  training-metrics record. The feedback arm 30001482 had not started.
- Cancelled all three jobs to release CS resources. This is an operational
  failure, not a null outcome; no smoke metrics enter any figure or analysis.
- The full 7B campaign remains gated. A retry requires a revised placement or
  model-serving configuration and a fresh operational smoke decision.
- Concurrent refresh: 3B Countdown landed its first post-training evaluation
  at 0.67 passes for five jobs; 0.5B Countdown advanced to an 8.3-pass common
  horizon. `compute_divergence` and its paper prose were refreshed accordingly.

## Retried 2026-07-16 ~16:25 (E4 7B two-A6000 smoke)

- Added `launch_e4_tau_control_7b_smoke_2xa6000.sh` with a fresh
  `gce4_taucontrol_7b_smoke_2xa6000` stamp. Each job requests two A6000s and
  sets total GPUs and GPUs per actor to two, yielding one tensor-parallel vLLM
  actor across the pair while the collocated learner sees both devices.
- Submitted Dr.GRPO 30001489, fixed xDr 30001490, and feedback xDr 30001491 on
  `node206` (`cs` / `allcs`, 192 GB, 12 hours). All three are priority-pending
  behind a higher-priority CS A6000 request; no metrics exist yet.
- The smoke remains seed 9001 and outcome-excluded. Full 7B seeds 43--45 remain
  gated on all three retry arms passing the operational checker.

## Blocked 2026-07-16 ~16:28 (E4 7B two-A6000 smoke v2)

- The first two-A6000 attempt established that the intended placement works:
  vLLM loaded one tensor-parallel actor across GPUs 0 and 1, using about 7.1
  GiB of model memory per worker. It then stopped before training on OAT's
  assertion that global `rollout_batch_size` be divisible by the two-GPU actor
  width. Cancelled jobs 30001489--30001491; they produced no analysis results.
- Updated `launch_e4_tau_control_7b_smoke_2xa6000.sh` to use global rollout
  batch 2 and per-device rollout batch 1, under the fresh
  `gce4_taucontrol_7b_smoke_2xa6000_v2` stamp.
- Submitted Dr.GRPO 30001492, fixed xDr 30001493, and feedback xDr 30001494.
  The first two jobs again loaded vLLM across both A6000s, then DeepSpeed
  rejected global train batch 32 with per-device train batch 32 over two
  learner ranks (zero gradient-accumulation steps). The feedback arm had not
  started. Cancelled all three without analysis results.
- This remains an outcome-blind operational smoke at seed 9001. Full 7B seeds
  43--45 remain gated, and no smoke value is plotted as an experimental result.

## Preflight 2026-07-16 ~16:34 (E4 7B two-A6000 smoke v5)

- Preserved global train batch 32 and set per-device train batch 16, satisfying
  DeepSpeed's two-rank batch identity with one gradient-accumulation step. The
  rollout geometry remains global 2 / per-device 1.
- Jobs 30001515--30001517 (v3) were submitted correctly but cancelled while
  priority-pending: node206 became planned for higher-priority CS work, so a
  12-hour smoke no longer fit the backfill window. A v4 submission to the
  non-preemptible `all` partition was rejected before any job was created.
- Submitted the fresh `gce4_taucontrol_7b_smoke_2xa6000_v5` stamp through the
  `lowprio` partition under `allcs`, allowing the idle A6000 nodes node103,
  node104, and node208. Dr.GRPO 30001581 and fixed xDr 30001582 ran on
  node208; feedback xDr 30001583 ran on node103. Every job requested two
  A6000s, 192 GB host memory, and 12 hours. Low-priority jobs may be requeued
  if an owning partition needs a node.
- All three jobs loaded vLLM tensor-parallel, initialized the two-rank ZeRO-2
  learner with gradient accumulation 1, and entered step-0 evaluation. The
  live scheduler log then revealed `max_steps=32`: one epoch over the 1,024
  prompts at train batch 32 could never reach the step-256 smoke gate or cross
  the controller warmup. Cancelled all three before extended training; their
  operational records remain outcome-excluded.

## Running 2026-07-16 ~16:40 (E4 7B two-A6000 smoke v6)

- Corrected the smoke budget to eight prompt epochs over the 1,024-prompt
  graph-coloring pool. At global train batch 32 this resolves to 256 optimizer
  steps, matching the checker, the step-64 controller warmup, and the step-256
  checkpoint. `max_train=1024` now states the actual pool cap explicitly.
- Submitted Dr.GRPO 30001591, fixed xDr 30001592, and feedback xDr 30001593
  under `gce4_taucontrol_7b_smoke_2xa6000_v6`. The first two are running on
  node103 and the feedback arm is running on node104; every job has two A6000s.
- Live initialization logs for all three resolve one actor and two learner
  ranks over GPUs 0--1, report successful tensor-parallel model loading, and
  print `num_policy_sgd_steps_per_episodes=32; max_steps=256` with no startup
  traceback.
- This is still the seed-9001 operational smoke, not the full 7B experiment.
  Full seeds 43--45 remain gated on all three arms reaching step 256 with
  finite entropy logs and non-empty checkpoints.

## Storage cleanup 2026-07-16 ~16:47

- The shared filesystem had only 336 GB free. The dominant debris was
  DeepSpeed optimizer state from stopped graph-coloring campaigns, not source
  code, datasets, metrics, or model caches.
- Removed 21 optimizer-state shards from the cancelled 3B graph-coloring
  campaign (about 955 GiB allocated) and 65 from the cancelled 0.5B
  graph-coloring campaigns (about 474 GiB allocated). These files are needed
  only to resume training.
- Retained all 86 corresponding model-state checkpoints, every training metric
  log, manifests, evaluation artifacts, plots, and paper inputs. Active
  Countdown and 7B runs were excluded from cleanup.
- Free shared capacity rose from 336 GB (96% used) to 1.4 TB (81% used). The
  active 7B smoke jobs 30001591--30001593 remained running throughout.

## Figure refresh 2026-07-16 ~16:50

- Re-parsed all four landed compute-scaling stamps. The 0.5B Countdown block
  now has a ten-pass common horizon (individual runs through 10.7 passes), and
  the five active 3B Countdown jobs now include step-512 / 1.33-pass
  evaluations in addition to initialization and step 256.
- Updated `compute_divergence` and its manuscript prose. At the newest common
  3B Countdown point, fixed xDr remains essentially matched to Dr.GRPO; this
  short trajectory is displayed as progress, not a domain conclusion.
- The 7B graph-coloring row remains workflow status only: all three two-A6000
  smoke arms are running, and no smoke outcome enters the figure.

## Figure amendment 2026-07-16 ~16:52

- Added greedy `pass@1` as the first metric column in both Countdown and graph
  coloring, alongside sampled `pass@8`, `coverage@8`, and `distinct@8`.
- Job 30001482 is not pending: it was the never-started single-A6000 feedback
  smoke and was cancelled at 16:20. Its corrected two-A6000 successor is job
  30001593, running under the outcome-excluded v6 smoke stamp.

## Figure refresh 2026-07-16 ~17:02

- Re-parsed all landed curves after adding `pass@1`. The 0.5B Countdown common
  horizon advanced to 12.3 passes (individual runs to 13), while all five
  active 3B Countdown jobs reached the two-pass evaluation.
- The two-pass 3B Countdown snapshot shows an early sampled-breadth separation
  for fixed xDr, while greedy `pass@1` and `mean@8` remain nearly matched.
  Manuscript language labels this as a short exploratory trajectory rather
  than a domain conclusion.
- The outcome-excluded 7B smoke remains status-only and contributes no curve.

## Adaptive-arm figure visibility 2026-07-16 ~17:52

- Added `xdr_tau_control` as an explicit orange-dotted series in the
  compute-divergence plotting contract and legend. The 7B status row now names
  Dr.GRPO, fixed xDr, and feedback xDr separately instead of hiding them behind
  “3 arms.”
- No orange outcome curve is drawn yet. Job 30001593 is the outcome-excluded
  operational smoke and remains before the controller's 64-observation warmup;
  the 0.5B and 3B analytical curve files contain only Dr.GRPO and fixed xDr.

## Launched 2026-07-16 ~18:03 (full analytical 7B E4)

- Diagnosed the v6 smoke's premature end: OAT clamps `max_queries` by
  `max_train`, so `max_train=1024` exhausted the query budget after 17
  controller observations. It could not cross the 64-observation warmup.
- Cancelled outcome-excluded smoke jobs 30001591--30001593. No smoke outcome
  is admitted to analysis.
- Promoted `launch_e4_tau_control_7b.sh` to the two-A6000 analytical recipe and
  submitted all three arms at seeds 43--45 under
  `gce4_taucontrol_7b_full_2xa6000_v1`: jobs 30001867--30001875. Each requests
  two A6000s, 192 GB, and seven days in `lowprio/allcs` over nodes 103, 104,
  and 208.
- The analytical budget is 16 prompt-pool passes with checkpoint evaluation
  every 256 collection steps (half a pass). Inline metrics and evaluations are
  retained; ZeRO optimizer checkpoints are disabled and model retention is
  one rolling snapshot so nine 7B jobs do not exhaust shared storage.

## Figure refresh 2026-07-16 ~18:25

- Parsed the first admissible 7B records: initialization evaluations for seeds
  43 and 44 in Dr.GRPO, fixed xDr, and entropy-feedback xDr. These six records
  are displayed as initialization only; their sampled differences are not
  treatment effects. Six jobs are running and the three seed-45 jobs are
  queued.
- Refreshed every older scaling curve. The 0.5B Countdown common horizon is
  now 15.3 passes (individual runs to 16.3). The 3B Countdown jobs now share a
  3.33-pass horizon, where fixed xDr minus Dr.GRPO is +2.1 coverage points,
  +7.8 pass@8 points, +0.11 distinct modes, and +2.7 pass@1 points.

## Launched 2026-07-16 ~18:30 (entropy-feedback scale/domain extension)

- Corrected the figure/workflow mismatch: the orange entropy-feedback arm had
  only been launched at 7B graph coloring. Added a feedback-only extension for
  0.5B and 3B in both Countdown and graph coloring, reusing the landed Dr.GRPO
  and fixed-xDr controls under their exact E1 recipes.
- Submitted seeds 43--45 for Countdown 0.5B (30002161--30002163), Countdown 3B
  (30002164--30002166), graph coloring 0.5B (30002167--30002169), and graph
  coloring 3B (30002170--30002172). The 0.5B jobs target idle A5000 node204
  through `lowprio/allcs`; the matched 3B jobs remain queued for A100 node302.
- Added `OAT_ZERO_ONLY_ARMS` to the shared comparative launcher so incremental
  treatments do not duplicate expensive landed controls. Optimizer archives
  are disabled and one rolling model snapshot is retained for these extensions.
- Moved the six 0.5B jobs to idle A5000 node204 under `lowprio/allcs`; all six
  started with the resolved full query budgets and entropy-feedback controller
  enabled. Their initialization evaluations landed. The 3B jobs remain queued
  for the matched A100 node302.

## Figure refresh 2026-07-16 ~18:52

- The 0.5B feedback extension now has post-training curves: all three Countdown
  seeds through step 512 / 1.33 passes, and all three graph-coloring seeds
  through step 384 / two passes (one seed through step 448). All six controllers
  are active and have reached the configured minimum tau of 0.005.
- At the latest common feedback point, Countdown feedback minus fixed xDr is
  +4.4 coverage points and +0.17 distinct modes but -6.0 pass@1 points; graph
  coloring feedback is close to fixed xDr (-0.6 coverage points and essentially
  equal distinct modes). These are early exploratory tradeoffs, not endpoints.
- Refreshed the fixed-arm curves as well: 0.5B Countdown now has a 17-pass
  common horizon (individual runs to 18), and 3B Countdown has a four-pass
  common horizon. The 3B feedback jobs remain queued; 7B remains initialization
  only.

## Figure refresh 2026-07-16 ~18:58

- Re-parsed every scaling source. All three 0.5B Countdown feedback seeds now
  reach step 640 / 1.67 passes; all three graph-coloring feedback seeds reach
  step 576 / three passes, with one seed through step 640 / 3.33 passes.
- At the latest common feedback point, Countdown feedback minus fixed xDr is
  +5.8 coverage points, +0.24 distinct modes, +3.1 pass@8 points, and +1.8
  pass@1 points. Graph coloring is +0.6 coverage points and +0.07 distinct
  modes, with pass@8 5.6 points lower. These remain short exploratory curves.
- The fixed-arm 0.5B Countdown common horizon advanced to 17.3 passes
  (individual runs to 18.3). The 3B feedback jobs remain queued, and the 7B
  graph-coloring campaign still has initialization only.

## Figure refresh 2026-07-16 ~19:02

- A final live re-parse captured the completed 0.5B Countdown feedback runs:
  all three seeds reached step 768 / two passes. Feedback minus fixed xDr at
  that point is +8.6 coverage points, +0.35 distinct modes, and +6.0 pass@8
  points; pass@1 and mean@8 are 1.8 and 3.8 points lower.
- The three graph-coloring feedback seeds reached step 704 / 3.67 common
  passes, with one seed through step 768 / four passes. At the common point,
  feedback is +1.3 coverage points and +0.10 distinct modes relative to fixed
  xDr, with pass@8 4.0 points lower.
- The fixed-arm 0.5B Countdown common horizon advanced to 17.7 passes
  (individual runs to 18.3). No 3B feedback or post-training 7B result has
  landed.

## Figure refresh 2026-07-16 ~20:12

- Re-parsed all nine curve sources. The 0.5B fixed-arm Countdown comparison
  now has a 21.67-pass common horizon (individual runs to 22.67), and the 3B
  Countdown comparison has a six-pass common horizon.
- The 0.5B graph-coloring feedback arm advanced to step 1792 / 9.33 common
  passes, with one seed at step 1856 / 9.67. At the common point, feedback
  minus fixed xDr is +9.6 coverage points, +0.60 distinct modes, +22.7 pass@8
  points, +5.9 pass@1 points, and +0.6 mean@8 points. The adaptive arm remains
  broad after fixed xDr has returned near the baseline regime, but no plateau
  or endpoint has landed.
- The 0.5B Countdown feedback common horizon remains two passes, with one seed
  at 2.33 after requeue. Its current common-point gaps versus fixed xDr are
  +4.3 coverage points, +0.15 distinct modes, and +2.3 pass@8 points, with
  pass@1 and mean@8 lower by 3.4 and 5.3 points.
- Initialization is now present for all three seeds in every 7B arm. No
  post-training 7B evaluation or 3B feedback evaluation has landed.

## Figure refresh 2026-07-16 ~21:05

- All five fixed-arm 0.5B Countdown jobs reached step 8960 / 23.33 passes. The
  fixed-xDr gaps are effectively zero: +0.1 pass@1, +0.1 pass@8, +0.2 mean@8,
  and -0.2 coverage points relative to Dr.GRPO.
- The five 3B Countdown jobs reached step 2816 / 7.33 common passes, with one
  baseline seed at eight passes. Fixed xDr is +1.0 coverage, +5.7 pass@8,
  +6.3 pass@1, and +6.5 mean@8 points at the common horizon.
- The 0.5B graph-coloring feedback arm reached step 3328 / 17.33 common passes,
  with one seed at step 3456 / 18 passes. At the 14-pass horizon shared by all
  fixed-arm seeds, feedback minus fixed xDr is +4.6 coverage points, +0.30
  distinct modes, and +12.0 pass@8 points. Feedback coverage has nevertheless
  fallen to 9.8% by 17.33 passes: adaptive tempering substantially delays the
  collapse but has not stopped it.
- Countdown feedback remains at two common passes after repeated requeues. The
  3B feedback arms remain queued, and all 7B arms remain initialization-only.

## Figure refresh 2026-07-16 ~22:05

- Countdown feedback moved beyond the prior frontier for seeds 44 and 45,
  which now reach step 896 / 2.33 passes. Seed 43 remains at step 768 / two
  passes after seven scheduler restarts, so the three-seed common horizon is
  still two passes. At that horizon, feedback minus fixed xDr is +6.4 coverage
  points, +0.25 distinct modes, and +4.2 pass@8 points; pass@1 and mean@8 are
  lower by 6.3 and 5.7 points.
- All three 0.5B graph-coloring feedback seeds completed step 4608 / 24 passes.
  The scheduled endpoint evaluation has 8.8% coverage and 0.57 distinct modes:
  substantially delayed, but continuing, collapse.
- Fixed-xDr seed 43 produced the first post-training 7B evaluation at step 256
  / half a pass. No control or feedback seed has a matched post-training point,
  so this is displayed as progress rather than a treatment comparison.
- The parser now ignores a duplicate terminal evaluation when it follows the
  scheduled evaluation at the same step, preserving the figure's one-draw-per-
  checkpoint contract.

## Night queue hardening 2026-07-16 ~22:16

- Audited the complete matrix of Dr.GRPO, fixed xDr, and entropy-feedback xDr
  across Countdown and graph coloring at 0.5B, 3B, and 7B. Landed controls are
  reused rather than duplicated.
- Moved the repeatedly preempted 0.5B Countdown feedback jobs 30002161--63 from
  `lowprio` to non-preemptible `cs/allcs` on idle A5000 nodes 202--203. Their
  existing run stamps and furthest coherent attempts are preserved.
- Replaced the six node302-blocked 3B feedback jobs with two-A6000 CS jobs:
  Countdown 30002806--08 and graph coloring 30002809--11. The Countdown trio
  started immediately on node207; the original pending jobs 30002164--66 and
  30002170--72 were cancelled only after their replacements were accepted.
- Moved pending 7B graph-coloring jobs 30001867, 30001869, and 30001871 onto
  non-preemptible CS placement; all three started on nodes205/207. The six
  already-running graph jobs were left undisturbed.
- Submitted the previously missing full 7B Countdown matrix on CS A6000 nodes:
  Dr.GRPO, fixed xDr, and entropy-feedback xDr at seeds 43--45, jobs
  30002812--20. The nine jobs are queued behind the older 3B feedback work and
  use the shared easy3 pool, 16 passes, and two A6000s per job.

## Night breadth scheduling 2026-07-16 ~22:21

- A strict FIFO estimate left every 7B Countdown job several days behind extra
  graph-coloring seeds. Rebalanced the available CS A6000 memory so one complete
  7B Countdown method triplet runs tonight: seed-43 Dr.GRPO 30002812, fixed xDr
  30002813, and entropy-feedback xDr 30002814.
- The just-started 7B graph jobs 30001867/69/71 and 3B graph jobs 30002810/11
  were returned to the queue after negligible startup-only runtime; they were
  released from hold and remain eligible. Six older 7B graph jobs and 3B graph
  feedback seed 43 (30002809) continue running.
- All three 0.5B Countdown feedback jobs 30002161--63 started on node202 in the
  non-preemptible CS partition. All three 3B Countdown feedback jobs
  30002806--08 continue healthy two-GPU initialization/training on node207.
- The 7B Countdown Dr.GRPO and fixed-xDr seed-43 initialization evaluations
  landed immediately after launch; the feedback initialization was still in
  startup at the last queue audit. These are initialization checks, not a
  treatment comparison.
- Initialization evaluations also landed for all three 3B Countdown feedback
  seeds and graph-coloring feedback seed 43. No post-training 3B feedback
  evaluation had landed at handoff.

## Figure refresh 2026-07-16 ~22:30

- Re-parsed all twelve curve sources after moving the live extensions onto CS.
  Two 0.5B Countdown feedback seeds now reach step 1152 / three passes; seed 43
  remains at step 768 / two passes, so the three-seed common horizon and its
  previously reported fixed-xDr contrasts are unchanged.
- The 7B graph-coloring feedback seed 45 reached step 256 / half a pass
  (pass@1 0.414, pass@8 0.859, coverage 0.278). Fixed-xDr seed 43 is the only
  other post-training 7B graph-coloring checkpoint. Because the two points are
  from different seeds and Dr.GRPO has no post-training point, the figure shows
  both as thin progress traces and makes no method comparison.
- Initialization evaluations have landed for every arm of the running seed-43
  7B Countdown triplet, all three 3B Countdown feedback seeds, and 3B
  graph-coloring feedback seed 43. None has a post-training evaluation yet.
- Queue audit at refresh: the seed-43 7B Countdown triplet, all three 0.5B and
  3B Countdown feedback jobs, one 3B graph-coloring feedback job, and six 7B
  graph-coloring jobs are running. The remaining requested seeds are queued.

## Countdown control recovery 2026-07-16 ~22:49

- The ten scheduler-visible 0.5B/3B Countdown control jobs were not healthy:
  every actor had terminated with CPython's `none_dealloc` fatal error while
  the learner remained blocked on the dead Courier RPC. Slurm therefore kept
  the allocations in `RUNNING` state after their logs stopped advancing.
- Preserved every landed metric and checkpoint, submitted replacement jobs
  30002923--30002934, then cancelled the twelve old running/pending jobs
  30001013--18 and 30001019--24 after every replacement was accepted.
- The five immediately schedulable 0.5B jobs resumed from step 8064 / 21
  passes, restored optimizer and prompt traversal state, and advanced beyond
  the checkpoint. Fixed-xDr seed 45 (30002928) is queued behind them.
- The five immediately schedulable 3B jobs resumed from step 2304 / six
  passes and all restored model, optimizer, and prompt traversal state.
  Fixed-xDr seed 45 (30002934) is queued behind them.
- Added opt-in highest-checkpoint discovery and a progress watchdog to the
  canonical launcher. These replacement jobs check `train_metrics.jsonl`
  progress after a one-hour startup grace; a 45-minute stall terminates the
  process tree and requeues the same job, up to four restarts. Subsequent
  restarts rediscover the numerically highest checkpoint rather than returning
  to the original recovery point. The recovery path also skips rewriting an
  already-loaded checkpoint at the initial resume evaluation, avoiding a
  second multi-gigabyte I/O burst on future restarts.
- Updated the curve parser to treat above-zero attempts as checkpoint resumes:
  it preserves the coherent pre-checkpoint prefix, switches to the resumed
  branch only after that branch overtakes the crashed frontier, and continues
  to keep unrelated step-zero reruns separate.

## Five-pass standardization 2026-07-17 ~11:23

- Standardized every maintained scaling launcher and restart path to a maximum
  of five complete prompt-pool passes in Countdown and graph coloring at every
  model scale. `ops/train.sh` is the final guard: inherited requests above five
  are explicitly logged and capped before the Python trainer starts.
- Corrected a misleading budget convention in the launchers. `MAX_TRAIN` caps
  unique dataset rows loaded per pass; it is not total rollout samples. The
  launchers now set it to the actual pool size (192, 384, or 1,024) and use
  `NUM_PROMPT_EPOCH=5` for the repeated-traversal budget.
- Cropped every compute-divergence panel to the shared zero-to-five-pass axis.
  Historical overrun metrics remain in the raw curve JSONs for provenance but
  are excluded by the plot loader and from the paper's five-pass comparisons.
- This ceiling was chosen after observing longer exploratory trajectories and
  is therefore documented as a post-outcome amendment, not a confirmatory
  preregistration change.

## Stale-allocation recovery 2026-07-17 ~11:33

- The scheduler's `RUNNING` label overstated experimental progress. Countdown
  3B feedback jobs 30002806--08 and graph-coloring 3B feedback seed 43 job
  30002809 had fatal actor failures followed by one-hour NCCL timeouts, but
  their batch allocations remained alive. The nine running 7B allocations
  stopped writing between 06:33 and 08:38 and showed 0% GPU utilization at the
  audit; they were blocked inside learn or parameter synchronization.
- Requeued all 13 stale allocations in place: 30002806--09, 30002812--14, and
  30001868/70/72--75. Existing metrics remain under their original run stamps;
  restarted attempts execute the new five-pass cap.
- Enabled recovery by default in `ops/run_experiment.sh`: highest-checkpoint
  discovery, a 45-minute no-metrics watchdog after a one-hour startup grace,
  same-job Slurm requeue, and up to six restarts. Checkpoint-free 7B runs start
  a fresh attempt under the same stamp, which the parser keeps separate until
  it overtakes the prior trajectory.
- Moved the three never-started graph-coloring 7B jobs 30001867/69/71 from a
  July-24 CS estimate back to the A6000 `lowprio` pool used by their six matched
  peers. This completes the runnable nine-job graph-coloring matrix without
  changing model or method settings.

## mltheory and idle-node recovery 2026-07-17 ~11:41

- The campaign launchers had hard-pinned the missing 3B entropy-feedback arms
  to CS A6000 nodes even though `mltheory/node302` was idle. Replaced that
  placement in situ: countdown jobs 30002806--08 and graph-coloring seed-43 job
  30002809 now occupy all eight node302 A100s and are producing training steps.
- Original graph-coloring feedback seed-45 job 30002811 failed during startup.
  Submitted replacement 30005532 under the same analytical run stamp on two
  `mltheory/node105` A5000s, with optimizer and activation offload to respect
  the cards' 24 GiB memory. Direct inspection showed ten idle GPUs and 489 GiB
  host memory available. The replacement initialized the feedback controller
  successfully.
- Fixed an early-exit hole in the progress watchdog: a child that stopped
  before creating `SAVE_PATH/train_metrics.jsonl` could trigger `set -e` in the
  probe and bypass automatic requeue. A missing metrics directory is now an
  expected empty state, and the wrapper reaches the child-status/requeue path.
- Moved the pending Countdown 7B seed-43 triplet onto otherwise idle 48 GiB
  hardware: Dr.GRPO 30002812 and fixed xDr 30002813 on node403 L40s, and
  feedback xDr 30002814 on node805 A6000s. All three began immediately.
- Original Countdown 7B seed-44 Dr.GRPO 30002815 and fixed-xDr 30002816 failed
  in the same startup window. Their same-stamp replacements 30005533 and
  30005534 target two A40s each on node101 and remain scheduler-eligible.

## Pass-axis, query-budget, and storage repair 2026-07-17 ~12:24

- Corrected the compute-divergence x axis to use
  `prompt_consumed / (num_samples * prompt_pool_size)`. Dividing global steps
  by prompt rows understated progress by two on placements with
  `rollout_batch_size=2`. All curve JSONs and the figure were rebuilt with the
  rollout-normalized pass coordinate.
- Confirmed that two-prompt optimizer steps also made fixed `eval_steps`
  schedules half as frequent in prompt-pass units. Comparative submissions now
  export an evaluation interval in prompts; `run_experiment.sh` converts it to
  optimizer steps using the resolved rollout batch size. Existing live jobs
  retain their submitted cadence; future jobs/restarts use pass-normalized
  cadence when the prompt interval is present.
- OAT's generic argument validation silently clamped an explicit
  `max_queries=100000000` to `max_train`. Once `max_train` was corrected to the
  prompt-pool row count, that would end five-pass jobs after roughly one pool
  of rollout queries. The xDr entry point now preserves an explicit positive
  query budget after upstream validation. Corrected jobs report five prompt
  epochs and `max_queries=100000000` at runtime.
- Replacement graph-3B feedback seed 45 job 30005538 is training on two
  node105 A5000s after the first wrapper-level failure. Countdown-7B seed-44
  replacements 30005533/30005534 are training on node101 A40s. An L40 attempt
  for Countdown-7B Dr.GRPO seed 43 was stopped after repeat ZeRO-2 optimizer
  failures; replacement 30005624 is queued for two node302 A100s.
- Deleted 49 inactive `checkpoints`/`saved_models` trees while preserving all
  metrics, evaluations, manifests, plots, and dataset definitions. This freed
  2,089.5 GiB: `var/data` fell from 2.1 TiB to 29 GiB and filesystem usage from
  78% to 56%. Cleanup job 30005623 depends on the full current campaign and
  will repeat the active-stamp-protected cleanup after it terminates.

## Live-motion audit and 7B baseline recovery 2026-07-17 ~13:50

- Reparsed every Countdown/graph-coloring curve and rebuilt
  `compute_divergence` from the rollout-normalized five-pass axis.
- A timed metrics audit separated genuine optimizer motion from Slurm's
  allocation state. All six 3B feedback runs were advancing; several 7B runs
  were advancing or in fresh startup, but Countdown-7B Dr.GRPO seeds 43 and 44
  were still allocated after fatal ZeRO optimizer OOM/NCCL failures.
- Cancelled the dead L40/A40 allocations 30002812 and 30005533. Seed 43 already
  had two-A100 replacement 30005624; submitted matching two-A100 seed-44
  replacement 30005908 with prompt-normalized evaluation cadence. Extended
  cleanup job 30005623's dependency set to protect the new replacement.

## Graph-coloring 3B full-five-pass controls 2026-07-17 ~14:56

- Submitted a clean five-pass extension of the incomplete graph-coloring 3B
  Dr.GRPO and fixed-xDr controls under stamp `gce1_3b_full5_qeval_v1`.
  Seeds 43--45 are jobs 30006113--30006118 on node302, one A100 per job.
- The 1,024-prompt pool evaluates every 256 prompts (one quarter epoch), with
  runtime enforcement against hardware-dependent or looser step cadences.
  Optimizer checkpoints are disabled; inline evaluations and the latest model
  snapshot are retained. All six jobs were accepted and initially pending on
  the node302 reservation.

## E5 Haarnoja-style entropy-dual extension 2026-07-17 ~14:35

- Froze the prospective exploratory design in
  `paper/preregistration/e5_haarnoja_dual.md` before launching any E5 job. The
  fourth arm learns `log(alpha_xdr)` with the signed SAC entropy-dual objective
  and scalar Adam, then maps `alpha_xdr = 0.05 / tau`. It is a controller
  transfer, not full SAC or a token-entropy reward.
- Submitted seeds 43--45 for both environments at 0.5B, 3B, and 7B: 18 new
  treatment jobs, with existing Dr.GRPO, fixed-xDr, and proportional-feedback
  controls reused. Every E5 run is capped at five prompt-pool passes and saves
  inline metrics rather than large model checkpoints.
- 0.5B jobs 30006002--30006007 launched on node202 A5000s and all advanced
  beyond the 64-observation controller warmup. Logs contain signed dual
  gradients and nonconstant alpha/tau values, directly confirming the new
  controller is active rather than silently behaving as fixed xDr.
- The initial 3B A6000 submissions 30006008--30006013 were cancelled before
  startup when Slurm predicted a multi-day CS priority wait. Clean replacement
  manifests use two node105 A5000s with optimizer/activation offload:
  Countdown 30006060--30006062 and graph coloring 30006063--30006065. Four
  began immediately; two remain scheduler-eligible behind local capacity.
- 7B jobs 30006014--30006019 request two node302 A100-80GB GPUs each with the
  established two-prompt layout and offload. They are queued behind the active
  7B work rather than being routed to the 48GB devices that caused prior
  optimizer failures.
- Extended cleanup job 30005623 to wait for every active E5 job. Refreshed all
  six E5 curve artifacts and rebuilt `compute_divergence` as a four-method
  grid; missing E5 outcomes remain explicitly marked queued.

## E6 true Candidate-MaxEnt projection extension 2026-07-17 ~15:20

- Froze the prospective design in
  `paper/preregistration/e6_candidate_maxent_projection.md` before running the
  smoke or full grid. The requested `t=0.05` is the projection target
  temperature: score temperature is 1, reference tilt is disabled, and the
  target is therefore exactly the fixed-xDr candidate distribution while the
  learner objective changes from signed clipped PPO to the Appendix
  length-normalized candidate-distribution projection.
- Added three rows: fixed projection (`xdr_maxent`), proportional feedback on
  the projection temperature (`xdr_maxent_tau_control`), and the Haarnoja-style
  entropy dual on that temperature (`xdr_maxent_sac_dual`). The smoke jobs
  30006204--30006206 completed successfully through terminal step 4 with
  finite projection gradients and controller telemetry; they are operational
  validation only and are excluded from analysis.
- Submitted the full prospective grid: three methods x three seeds x two
  environments x three model scales = 54 runs. Countdown jobs are
  30006209--30006217 (0.5B), 30006227--30006235 (3B), and
  30006245--30006253 (7B); graph-coloring jobs are 30006218--30006226,
  30006236--30006244, and 30006254--30006262, respectively. Every run is
  capped at five prompt-pool passes and writes inline evaluation metrics.
- The 0.5B jobs began immediately on CS A5000 nodes and live logs confirmed
  optimizer motion for all three objectives, including nontrivial
  proportional and signed-dual state. The 3B A6000 jobs are CS-priority
  pending; 7B jobs request two A100-80GB GPUs on `mltheory/node302` and are
  pending node availability.
- Extended cleanup sentinel 30005623 to wait for all 54 E6 jobs. Added the six
  E6 curve artifacts and seven-row plotting/monitoring surface; absent outcomes
  are displayed as queued rather than fabricated or silently omitted.

## E6 guardrail failure and E7 objective repair 2026-07-17 ~15:34

- The first E6 0.5B evaluations were not a plotting artifact. All six inspected
  method/environment trajectories converged to mean response length 2, zero
  rollout reward, and all-zero groups. Fixed graph seed 43 reached reward
  0.8125 at step 21 but had reward 0 at step 250; its reward EMA was effectively
  zero. This triggered E6's pre-specified pass@1/accuracy guardrail.
- Cancelled jobs 30006209--30006262. Fifteen 0.5B jobs had trained for roughly
  13 minutes; the remaining 0.5B jobs and every 3B/7B job had not started.
  Metrics remain under the E6 stamps as a negative pilot.
- Diagnosed a structural candidate-length bias. For on-policy uniform targets,
  `E[grad log pi(Y)] = 0`, while E6's per-candidate division gives
  `E[grad log pi(Y)/T(Y)] = grad E[1/T(Y)]`; descent therefore rewards short
  samples even without a reward contrast, creating the observed EOS attractor.
- Froze `paper/preregistration/e7_candidate_maxent_fixed_scale.md` before any
  repaired run. E7 uses a shared `1/T_max` scale, which only rescales the
  forward-KL projection, and excludes reward-constant groups from finite-sample
  self-distillation. Target weights and all three temperature laws are
  otherwise unchanged.
- Repointed launch, curve, plot, and monitor stamps from E6 to E7. A mandatory
  128-step three-arm graph-0.5B smoke must pass finite telemetry, response
  length, trailing reward, and controller-warmup checks before any repaired
  3B/7B job is submitted.
- Submitted that gate as jobs 30006354--30006356 under stamp
  `gce7_maxent_fixedscale_smoke_v1` after 139 maintained tests passed. The
  analytical 54-run E7 grid remains unsubmitted pending the automated gate.

## E7 smoke pass and full fixed-scale grid 2026-07-17 ~15:50

- Jobs 30006354--30006356 completed all 128 updates and Slurm recorded exit 0.
  The frozen gate passed: terminal mean lengths were 3.88, 15.75, and 4.00;
  maximum trailing-32-step rollout rewards were 0.8125, 0.9375, and 0.875 for
  fixed, proportional, and dual projection. Losses/gradients were finite and
  both controllers crossed warmup; the dual learned a nonconstant target
  temperature. Unlike E6, no arm converged to two-token zero-reward output.
- After the gate passed, submitted the complete E7 grid. Countdown job ranges
  are 30006358--30006366 (0.5B), 30006376--30006384 (3B), and
  30006394--30006402 (7B). Graph-coloring ranges are 30006367--30006375,
  30006385--30006393, and 30006403--30006411. Each range contains the three
  methods x seeds 43--45, for 54 unique jobs total.
- All 0.5B/3B jobs target the established CS A5000/A6000 placements; all 7B
  jobs request two A100-80GB GPUs on `mltheory/node302`. Every run retains the
  five-pass cap and prompt-normalized evaluation cadence. Cleanup sentinel
  30005623 now waits for all 54 jobs.

## Stall recovery and immutable job entry points 2026-07-17 ~15:22

- A two-sample metric/log audit found one genuine stall: graph-coloring 3B
  proportional-feedback seed 43, job 30002809, stopped after step 1089 when a
  vLLM worker aborted in `cumem.unmap_and_release` with Python's
  `none_dealloc` fatal error. The learner remained blocked on the dead actor
  RPC while Slurm still reported `RUNNING`. Requeued the same job under the
  same run stamp; restart count is now two and the replacement is pending
  node302 availability. All other allocated campaign jobs wrote fresh metrics.
- Added immediate watchdog recognition of fatal Python/vLLM worker signatures,
  scoped to log bytes from the current attempt, while retaining the 45-minute
  no-metrics fallback. Requeued legacy jobs infer their Slurm output path, so
  job 30002809 also receives the faster detector when it restarts.
- The three graph-coloring 0.5B Haarnoja jobs reached their terminal five-pass
  evaluations but Slurm recorded exit 2. They had started before a live edit of
  `ops/train.sh`; Bash later read the changed file during shutdown and reported
  an unmatched quote. New allocations snapshot `run_experiment.sh` and
  `train.sh` into an attempt-specific local directory before execution, so
  working-tree edits cannot splice a running shell program again.
- Fixed the live monitor to merge duplicate historical directories and to
  carry the original graph-coloring 3B control frontier into the queued
  full-five-pass extension. Focused campaign/cadence/parser/progress tests pass.

## Deterministic vLLM sleep failure and checkpointed recovery 2026-07-17 ~16:00

- Graph-coloring 3B proportional-feedback seed 44, job 30002810, repeated the
  seed-43 failure exactly: after step 1089, the next vLLM sleep call aborted in
  `cumem.py` with `none_dealloc`, its worker exited `-6`, and the learner hung
  on the dead actor RPC while Slurm remained `RUNNING`.
- Requeue alone cannot cross this frontier because the original jobs disabled
  optimizer checkpoints. Replaced the stalled seed-43/44 copies with jobs
  30006412/30006413 under the same run stamps. They retain one rolling
  checkpoint at step 1024, automatically resume it after watchdog requeue, and
  preserve quarter-prompt-epoch evaluation. The old checkpoint-free pending
  jobs 30002809/30002810 were cancelled after replacements were accepted.
- Changed the maintained two-A6000 3B feedback launcher to enable one rolling
  recovery checkpoint. The submission helper now supports explicit append-only
  same-stamp recovery manifests, and the 3B launcher permits a seed subset so
  recovery does not duplicate healthy seeds. Cleanup job 30005623 waits for
  both replacements.

## Live motion and complete figure refresh audit 2026-07-17 ~17:01

- Sampled every allocated campaign run twice across a four-minute window. All
  26 jobs that remained allocated at the second sample advanced their current
  optimizer step; one additional Countdown-0.5B MaxEnt run completed. No job
  crossed the strict 12-minute no-metrics threshold, so no healthy allocation
  was cancelled or requeued.
- Reconfirmed the two historical failure classes. Graph-coloring 3B feedback
  seeds 43/44 aborted their vLLM workers in `cumem.unmap_and_release` with
  `Fatal Python error: none_dealloc` at step 1089. Their replacements
  30006412/30006413 retain a rolling step-1024 checkpoint, automatic resume,
  quarter-epoch evaluation, and Slurm requeue. The six 0.5B Haarnoja jobs had
  already landed terminal five-pass evaluations before a live-edited shell
  script produced their exit-2 status; immutable per-attempt script snapshots
  prevent that shutdown-only failure in new allocations.
- Added `refresh_campaign_curves.py` as the single refresh point for all 26
  figure inputs. The old `make figures` path refreshed only E5/E7, leaving live
  controls and E4 trajectories stale. Every parse now explicitly selects the
  valid five-pass horizon, including restarted attempts. Rebuilt the combined,
  Countdown, and graph-coloring PDF/PNG figures from the 17:00 metrics state.
- Focused cadence, parser, campaign-monitor, and progress-metric tests pass
  (23 tests), along with Ruff and shell syntax checks.

## E7 cancellation audit and Python-source immutability 2026-07-17 ~17:28

- A new scheduler audit found that 38 E7 allocations were canceled together by
  user action at 17:11:47; these were not independent crashes or watchdog stall
  detections. Two running Countdown-0.5B proportional jobs were healthy and
  writing metrics when terminated. The unstarted 3B/7B E7 grid was canceled in
  the same event.
- E7 Countdown-0.5B fixed seed 45 job 30006364 had instead been requeued. Its
  original process reached a landed 4.75-pass evaluation, but restarted
  allocations imported a changed checkout whose argument schema no longer
  recognized the candidate-projection flags. Three restart attempts exited at
  argument parsing. Canceled the resulting pending requeue loop; the 4.75-pass
  metrics remain preserved.
- Closed the underlying reproducibility gap. Comparative submissions now copy
  the complete Python `src/` tree into an immutable campaign snapshot before
  `sbatch`; the snapshot path is exported to every arm. Ad hoc Slurm jobs create
  a persistent per-job source snapshot on first allocation and reuse it across
  restarts. `PYTHONPATH` explicitly prioritizes the snapshot. The existing
  immutable shell snapshots remain in place.
- All 21 remaining allocated campaign jobs advanced and had metric ages below
  95 seconds. No active job was stale. Rebuilt all 26 curve artifacts and both
  environment figures; figure annotations now report the E7 grid as canceled,
  not queued. Shell syntax, Ruff, and 23 focused tests pass.

## E8 direct on-policy MaxEnt replacement 2026-07-17 ~17:42

- Retired E7 as an objective, not merely as an implementation failure. With
  candidates sampled from the behavior policy, its empirical Gibbs target has
  population base measure `pi_old`; it therefore implements a KL-regularized
  improvement step rather than direct policy entropy. The E7 launcher is now
  a fail-closed compatibility tombstone, and its code/traces remain only for
  provenance.
- Froze `paper/preregistration/e8_on_policy_maxent.md` before any E8 training
  outcome. E8 directly adds a leave-one-out sequence-surprisal advantage to
  fresh on-policy Dr.GRPO groups and uses one PPO epoch per rollout. Its fixed,
  proportional, and Haarnoja-dual arms all act on the same positive entropy
  coefficient; both controllers observe the exact normalized sequence-entropy
  statistic in the objective.
- Replaced the maintained public variants, launch surface, monitor, curve
  refresh, figure labels, repository overview, and manuscript appendix with
  the E8 treatment. The full replacement remains blocked on a separately
  stamped 128-update three-arm smoke; no E7 outcome is relabeled or reused.

## E8 smoke pass 2026-07-17 ~17:53

- Jobs 30006794--30006796 completed cleanly on node203 in seven minutes. The
  frozen checker passed fixed, proportional, and Haarnoja-dual direct MaxEnt at
  step 128: all required objective telemetry was finite, each arm retained a
  nonzero reward in its trailing window, and terminal response lengths were
  16.12, 4.12, and 4.00 tokens rather than the E6 two-token collapse.
- Both adaptive arms crossed their 64-observation warmup. At the low-entropy
  endpoint, proportional feedback raised alpha from 0.05 to 0.09515 and the
  dual raised it to 0.05574. This verifies that the replacement actuators move
  in the entropy-preserving direction while acting on direct policy entropy.

## E8 full direct-objective grid submitted 2026-07-17 ~17:54

- Released all 54 analytical jobs only after the automated smoke gate passed:
  Countdown-0.5B 30006799--30006807, graph-coloring-0.5B
  30006808--30006816, Countdown-3B 30006817--30006825,
  graph-coloring-3B 30006826--30006834, Countdown-7B
  30006835--30006843, and graph-coloring-7B 30006844--30006852.
- Each cell has fixed, proportional, and Haarnoja-dual direct MaxEnt at seeds
  43--45, one PPO epoch, prompt-normalized quarter-epoch evaluation, and a
  five-pass ceiling. The 3B runs request two A6000s on CS; the 7B runs request
  two A100s on node302 in `mltheory`. At submission all 54 were accepted and
  pending scheduler resources/priority.
- Extended cleanup sentinel 30005623 across all E8 jobs. It will preserve
  metrics while removing inactive model/checkpoint payloads after the complete
  campaign dependency set terminates.

## E8 first-motion and split-panel refresh audit 2026-07-17 ~17:59

- All 18 E8 0.5B analytical runs started across nodes 202--204 and advanced in
  two live samples. Countdown reached its first 0.25-pass evaluations for all
  nine runs; graph coloring reached 0.25--0.50 landed passes. Every one of the
  39 total allocated campaign jobs wrote metrics within 48 seconds during the
  strict audit. No allocation was stalled or failed, so none was canceled or
  requeued.
- Rebuilt all campaign curves and the new matched split-panel figures. The
  left panel contains only Dr.GRPO plus xDr aggregation-rescaling methods; the
  right repeats Dr.GRPO against direct on-policy MaxEnt. Corresponding metric
  axes share limits. The queued E8 3B/7B rows remain explicitly labeled rather
  than receiving smoke data or retired E7 outcomes.
- Corrected the live dashboard heading from E7 to E8. Ruff/compile checks and
  55 focused objective, smoke-gate, cadence, parser, and monitor tests pass.

## E8 actuator failure audit and retirement 2026-07-17 ~18:18

- The early behavioral curves were not consistent with a working entropy
  actuator. Direct telemetry isolated the failure: as policies concentrated,
  the prompt-group dispersion of E8's centered sampled surprisal collapsed.
  In graph-coloring dual runs, alpha rose from about 0.05 to its 0.5 ceiling
  while carrier dispersion fell from about 0.095 to 0.0011; Countdown fell
  from about 0.029 to 0.0034. The on-policy score identity is valid in
  expectation, but multiplying a vanished finite-group carrier by a larger
  coefficient cannot restore unsampled support.
- Cancelled all analytical E8 jobs 30006799--30006852. The 18 allocated 0.5B
  jobs stopped after approximately 25 minutes and retain their run directories,
  manifests, and partial metric/curve artifacts. The 36 queued 3B/7B jobs were
  cancelled before allocation. Completed smoke jobs 30006794--30006796 remain
  unchanged. No E8 metric is eligible for the replacement grid.

## E9 direct-gradient MaxEnt repair frozen 2026-07-17 ~18:31

- Froze `paper/preregistration/e9_direct_entropy_gradient.md` before any E9
  training outcome. E9 preserves the appendix objective
  `E[R] + alpha H(pi)/T_max`; it does not multiply the registered entropy
  strength by `T_max`.
- Removed entropy from the group-relative completion advantage. The actor now
  differentiates full categorical entropy at each sampled prefix and adds the
  causal future-entropy score term required for the complete autoregressive
  sequence-entropy gradient. This local full-vocabulary term remains active
  even when all sampled completions are identical and is outside PPO clipping.
- Applied Dr.GRPO's outer `1/T_max` scale and its `(G-1)/G` self-including
  reward-baseline attenuation to the entropy term, preserving alpha's stated
  objective units. Both controllers now observe categorical sequence entropy
  in `sequence_nats_per_tmax`, and checkpoint state records that unit tag.
- Repointed the maintained launcher, live monitor, curve refresh, figures,
  docs, and manuscript from E8 artifacts to fresh `*e9_direct_maxent*` stamps.
  The new seed-9005, 128-update three-arm smoke is mandatory before any of the
  54 analytical E9 jobs may be submitted. Exact enumeration tests verify that
  the causal surrogate matches a two-step policy's true sequence-entropy
  gradient; focused Ruff/shell checks and 35 tests pass before smoke launch.

## E9 smoke: operational pass, actuator guard failure 2026-07-17 ~18:53

- Initial CS jobs 30006996--30006998 were cancelled while pending because
  nodes 202--204 were planned until the next day. Two alternate placement
  attempts were also cancelled pending, and one `pvl-lowprio` attempt was
  invalid for the available account. These attempts produced no training row.
  The compact placement record is
  `var/artifacts/gce9_direct_maxent_smoke_v1_attempts.tsv`.
- Jobs 30007047--30007049 started on 24 GB RTX 3090s but OOMed on the first
  direct-entropy backward with a 16-row microbatch. They were cancelled before
  the watchdog could requeue. The repaired jobs kept effective train batch 16
  and used four-row microbatches with gradient accumulation; that setting is
  now shared by every E9 scale.
- Final smoke jobs 30007051--30007053 completed all 128 updates on node024.
  The original mechanical checker passed: all direct-gradient telemetry was
  finite, `(G-1)/G=0.9375`, gradients and tail rewards were nonzero, both
  controllers crossed warmup with the correct sign, and terminal lengths were
  4.06 fixed / 4.00 proportional / 4.12 dual rather than two-token EOS.
- The scientifically relevant actuator check failed. Final normalized
  categorical sequence entropy was 0.00565 fixed, 0.00658 proportional, and
  0.00511 dual. Over the last 16 updates, proportional retained 0.01273 versus
  target 0.03244 (39.2%); dual retained 0.00630 versus target 0.03668 (17.2%).
  Alpha moved in the correct direction (0.0751 proportional, 0.0583 dual), but
  neither adaptive arm held even half its own target.
- Added a clearly post-smoke, fail-closed actuator amendment to the E9 record
  and checker. `launch_on_policy_maxent_extension.sh all` now exits before
  submission on this observed run. No analytical E9 job was submitted; the
  six E9 curve artifacts remain empty and the figure labels them smoke-gated.

## E9b actuator calibration and adaptive smoke 2026-07-17 ~19:43

- Froze `paper/preregistration/e9b_maxent_actuator_calibration.md` before E9b
  outcomes. The direct causal estimator and objective were unchanged. E9b
  first measured a frozen-policy target, then tested fixed coefficient doses
  before allowing repaired controller jobs.
- Calibration job 30007143 supplied the prospectively selected first 64
  zero-learning-rate entropy observations, giving target 0.05191685. OAT
  continued beyond the requested 64 rows; the allocation was cancelled after
  84 rows and its watchdog requeue was cancelled. Overshoot rows are ignored.
- Fixed-dose jobs 30007144--30007147 completed. Alpha 0.05/0.10/0.20 retained
  11.5%/21.4%/25.2% of target. Alpha 0.50 retained 1223.9%, demonstrating
  actuator authority but producing a pathological 145-token terminal mean,
  12/16 no-EOS rollouts, and zero final evaluation accuracy.
- Replaced the proportional controller's unit-sensitive absolute-deficit
  exponent with a dimensionless relative deficit spanning the configured
  log-alpha range. Added explicit frozen entropy targets so both controllers
  act from their first observation. E9b dual used alpha LR 0.03 instead of
  0.003. Controller checkpoint rules and configured targets are tagged and
  incompatible states are rejected.
- Adaptive jobs 30007197--30007198 completed all 128 updates. Proportional
  peaked at alpha 0.3266 but retained 29.2% of target. Dual initially weakened
  alpha on high-entropy batches, reached alpha 0.5 only near the endpoint, and
  retained 13.3%. Both kept tail reward and four-token terminal responses.
- The frozen E9b adaptive gate therefore failed. No analytical direct-MaxEnt
  job was submitted. The failure is now localized to control across a sharp
  low-entropy/long-output transition, rather than estimator support or
  controller sensor units. The maintained test suite passes 166 tests before
  the post-smoke documentation update.

## E10 prefix-ratio estimator repair and smoke 2026-07-17 ~22:10

- Froze `paper/preregistration/e10_prefix_ratio_maxent.md` before outcomes.
  E10 replaces E9's rollout-tangent causal surrogate with the exact
  finite-horizon identity under behavior rollouts: new-policy categorical
  entropy at each state is weighted by the exclusive new/old prefix ratio.
  Positive prefix-occupancy increases use PPO's upper clipping branch.
- Exact enumeration under genuinely distinct old and new two-step
  autoregressive policies verifies both the unclipped entropy value and its
  gradient. Exclusive indexing, masking, clipping, restarted-metric
  discovery, and the fail-closed gate are covered by tests. The full
  pre-submission check passed 169 tests.
- Jobs 30007613--30007616 completed all 128 updates on node023 with exit code
  zero. Fixed alpha 0.05, proportional control, and Haarnoja dual retained
  24.7%, 17.7%, and 11.7% of the frozen target over their final 16 updates.
  The adaptive arms therefore fail the preregistered 50% retention gate even
  though final response lengths, evaluation accuracy, gradients, ratios, and
  tail rewards remained finite and nondegenerate.
- The corrected gradient created transient entropy/length excursions rather
  than stable target tracking. Fixed, proportional, and dual peak mean
  lengths were 40.25, 38.94, and 39.00 before all returned to roughly
  four-token terminal batches. Proportional and dual ended at alpha 0.321 and
  0.317 with only 0.00618 and 0.00412 terminal normalized entropy.
- The isolated alpha-0.50 guard ended at normalized entropy 0.752, mean length
  168.75/192, 14/16 no-EOS rollouts, and zero evaluation accuracy. Its
  terminal exclusive prefix ratio was one, demonstrating why ratio clipping
  cannot constrain the local categorical entropy incentive when learner and
  behavior policies are aligned.
- E10 therefore repairs the finite-step estimator but fails the scientific
  control gate. All 54 analytical direct-MaxEnt cells remain held and no
  analytical E10 job was submitted.

## E11 standard sequence-MaxEnt objective frozen 2026-07-17 ~22:35

- Froze `paper/preregistration/e11_standard_sequence_maxent.md` before any E11
  outcome. E11 retains E10's exact exclusive-prefix importance estimator but
  changes the active objective from `E[R] + alpha H/T_max` to standard
  `E[R] + alpha H`.
- The learner now computes raw categorical sequence entropy and applies only
  Dr.GRPO's one shared outer `1/T_max` update normalization. A unit-tested loss
  helper makes the absence of a second entropy division explicit. Raw
  `maxent_sequence_entropy` drives the controllers; the old normalized value
  remains separately available as `maxent_sequence_entropy_per_tmax`.
- Controller checkpoints now carry `sequence_nats_v1` and reject E9/E10's
  `sequence_nats_per_tmax` state. The frozen raw target is exactly 192 times
  E9b's normalized calibration target. Historical E10 launch code is a
  tombstone so it cannot silently execute under E11 units.
- Because literal standard `alpha=0.05` is 192 times stronger than E10's
  coefficient with the same numeral, E11 begins with a 32-update fixed-arm
  guard. Its failure blocks the adaptive smoke; both stages must pass before
  any of the 54 analytical cells can be reviewed for release.

## E11 literal standard-alpha gate failed 2026-07-17 ~22:37

- Job 30007745 ran the frozen 0.5B graph-coloring, seed-9005,
  `alpha=0.05` gate on node023. OAT ignored the requested 32-row ceiling and
  continued, so the allocation was cancelled after step 38. The checker is
  pinned to the prospectively specified step-32 row; overshoot is retained but
  excluded. The watchdog-created deferred requeue was cancelled before it
  allocated.
- Raw entropy and the comparison diagnostic had the required exact scale:
  62.4276 raw nats and 0.325144 nats per `T_max` at step 32. This verifies that
  the second entropy division was removed in live training.
- The gate failed with mean response length 87 and 7/16 no-EOS rollouts.
  Batch reward was 0.4375 and evaluation accuracy 0.05208, so the failure is
  strong long-output entropy pressure rather than numerical failure or
  complete loss of task signal.
- No proportional, Haarnoja-dual, or analytical E11 job was submitted. All 54
  standard-MaxEnt analytical cells remain held.

## E12 standard-MaxEnt coefficient calibration 2026-07-17 ~23:05

- Froze `paper/preregistration/e12_standard_maxent_dose_calibration.md` before
  outcomes. E12 retained E11's exact prefix-ratio estimator and standard raw-
  sequence objective, and varied only fixed `alpha` over 0.0005, 0.0010,
  0.0015, and 0.0020. This was a single-seed engineering calibration, not a
  comparative result.
- A scheduler-inaccessible first attempt reserved a header-only v1 manifest
  but created no job. Fresh v2 jobs 30007803--30007806 used byte-identical
  source snapshots, ran together on node023, completed the exact step-128
  endpoint, and exited zero. The pre-submission repository check passed 179
  tests; the hardened E12 checker additionally requires contiguous steps
  97--128, the assigned alpha on every row, and no controller telemetry.
- Alpha 0.0005, 0.0010, and 0.0015 were behaviorally safe but retained only
  1.794, 2.113, and 1.715 raw entropy nats over the final 16 updates, below the
  frozen 4.984-nat threshold. Their terminal evaluation accuracies were
  0.198, 0.203, and 0.172.
- Alpha 0.0020 retained 48.464 raw nats but failed behaviorally: final-16 mean
  length 60.76, maximum batch mean length 121.94, and maximum 10/16 no-EOS
  rollouts. Terminal accuracy was 0.172.
- The frozen selector therefore returns no coefficient. This supports a sharp
  engineering transition between entropy collapse and length/EOS runaway in
  the tested bracket; it does not establish a population threshold. No
  adaptive or analytical job was submitted. The next MaxEnt intervention
  must impose an explicit expected-length constraint.

## E13 expected-length-constrained MaxEnt 2026-07-17 ~23:36

- Froze `paper/preregistration/e13_length_constrained_maxent.md` before
  outcomes. E13 keeps standard sequence-MaxEnt at fixed `alpha=0.002` and
  imposes `E[L] <= 16` with a separate projected multiplier. The exact
  exclusive-prefix length estimator uses the conservative PPO `max` cost
  branch, opposite entropy's positive-reward `min` branch. The multiplier
  observes detached unclipped expected new-policy length and is updated for
  the next actor step from a target-initialized EMA.
- Exact variable-length enumeration verifies estimator value and gradient
  under distinct old/new policies. Tests cover prefix indexing, masking,
  cost clipping, the one shared outer `1/T_max`, zero-price equivalence,
  controller direction/projection/state, distributed observation, argument
  confounds, legacy-shell compatibility, and the fail-closed E13 checker.
- Jobs 30007892 (`eta=0.00005`) and 30007893 (`eta=0.00020`) used
  byte-identical source snapshots, completed 128 updates on node023, exited
  zero, and reproduced every frozen controller transition. Neither job
  requeued or required intervention.
- Both passed. Slow/fast final-16 entropy was 9.675/9.594 nats, actor length
  12.97/11.14, and expected length 12.98/11.13. Maximum final-16 no-EOS was
  2/16 in both. Terminal pass@1 was 0.182/0.177, pass@8 0.573/0.510, and
  coverage@8 0.144/0.138. The 5% tie rule selects `eta=0.00005`.
- The preferred slow arm transiently peaked at mean length 86.25 and 7/16
  no-EOS before its multiplier caught up; fast peaked at 40.25 and 3/16.
  This is a successful single-seed endpoint actuator gate, not per-update
  safety or replicated effectiveness. It authorizes a separately frozen
  three-seed replication. No replication or analytical-grid job was
  submitted automatically.
- Final repository verification passed 238 tests plus Python compilation,
  Ruff, shell syntax, and whitespace checks. The rebuilt paper has no
  undefined citations/references or fatal TeX errors.

## Non-MaxEnt five-pass backfill 2026-07-18 ~22:08

- Audited all 72 eligible Dr.GRPO, fixed-xDr, proportional-feedback, and
  Haarnoja-dual seed-runs against the five-pass landed horizon. There were 34
  deficient runs: 15 already had a live replacement, leaving 19 missing
  submissions. The 54 held standard-MaxEnt cells were explicitly excluded.
- Submitted same-stamp replacements for all 19 gaps: Countdown-3B controls
  30012401--30012406, graph-coloring-0.5B fixed-xDr seed 45 job 30012421,
  graph-coloring-3B controls 30012408--30012410, graph-coloring-3B Haarnoja
  jobs 30012423--30012425, and graph-coloring-7B controls/feedback
  30012414--30012419. Append-only manifests preserve the full attempt history.
- The graph-coloring-3B Haarnoja failures were the known vLLM
  `none_dealloc` crash just beyond two passes. Their replacements run on two
  CS A6000s and retain one rolling step-1024 optimizer checkpoint so automatic
  requeue can cross that frontier. The three missing graph-coloring-3B control
  replacements use the same recovery checkpoint policy. Initial pending jobs
  30012411--30012413 were cancelled before allocation when a configuration
  audit found their direct invocation had omitted inline coverage; final jobs
  30012423--30012425 explicitly restore coverage-at-8 every quarter pass.
- The exhausted graph-coloring-7B jobs were failing at the optimizer step with
  a 14.19-GiB allocation request on 48-GiB A6000s. Their six replacements use
  the successful 7B recovery layout: global batch 32, microbatch 4 with
  accumulation, optimizer and activation offload, two A6000s, and the
  non-preemptible CS partition.
- The first graph-coloring-0.5B replacement, 30012407, exposed the analogous
  24-GiB A5000 limit immediately: its 16-row backward OOMed at step 2. It was
  cancelled. Intermediate job 30012420 verified global batch 16 with a
  four-row microbatch and accumulation, but its startup dump exposed a missing
  inline-coverage flag in the direct seed-filtered submission, so it was
  cancelled before training. Final replacement 30012421 retains microbatch 4
  and restores coverage-at-8 evaluation every quarter pass.
- Extended cleanup sentinel 30005623 across all 19 new jobs. At the final
  audit, the dashboard reported 9 running and 25 pending eligible runs, zero
  terminal failures, and every sub-five cell had a live `R` or `P` status.

## Graph-coloring 7B allocator recovery 2026-07-19 ~15:11

- Resolved six displayed terminal failures—E4 Dr.GRPO seed 45, E4 feedback
  seeds 43/44, and all three E5 Haarnoja-dual seeds—to the same deterministic
  runtime fault. Every process completed learner step 1023 and died while
  entering step 1024 in vLLM 0.8.4's CuMem `unmap_and_release` sleep path with
  `Fatal Python error: none_dealloc: deallocating None`. The failure boundary
  and stack were identical across methods and seeds.
- The six earlier backfill jobs 30012414--30012419 had no optimizer checkpoint
  and were already in ineffective startup/requeue loops. They were cancelled
  after the repaired replacements had been submitted held and audited.
- Shared storage had only 501 GiB free, so rolling 7B optimizer checkpoints
  for twelve jobs were not safe. Recovery instead removes the defective path:
  vLLM remains resident (`OAT_ZERO_VLLM_SLEEP=0`) with KV ratio 0.10, while the
  learner uses the validated two-A6000, microbatch-4, optimizer/activation-
  offload layout. The objective, data, seeds, G=32, global batch, learning
  rate, five-pass ceiling, and quarter-pass evaluation cadence are unchanged.
- Submitted and released same-stamp E4 jobs 30016313--30016321 and E5 jobs
  30016322--30016324 on the healthy node103/node104/node208 A6000 pool under
  `lowprio/mltheory`. All twelve explicit Slurm exports passed the held-job
  audit before release. At 15:11 EDT they were pending for priority; the live
  monitor showed `submitted` for all twelve Graph-coloring 7B cells and no
  remaining `FAIL` label.

## Active-safe storage cleanup 2026-07-20 ~12:02

- Audited shared storage and the live Countdown-3B allocations before
  deleting anything. Shared storage had 2.7 TiB free; node103/node104 had
  2.0/1.6 TiB free in `/tmp`, essentially empty `/dev/shm`, and about 235 GiB
  of available RAM each. Each job-local `/tmp/od2961` tree was only 52 KiB.
- Removed 73 inactive `saved_models`/`checkpoints` payload trees from the old
  E1 and E4--E8 campaigns, freeing 315.2 GiB. Run directories, inline
  metrics, evaluations, manifests, figures, and dataset definitions were
  retained. The cleanup discovered active stamps from Slurm and therefore
  could not touch any live run root.
- Explicitly retained all E16--E21 canonical and free-form model state,
  including the four 46-GiB rolling optimizer checkpoints that belong to
  live 3B runs. Shared filesystem usage fell from 63% to 60%, with 2.9 TiB
  free after deletion.
- Countdown-3B matched Dr.GRPO jobs 30020609--30020611 and Standard MaxEnt
  fixed jobs 30020671--30020673 remained `RUNNING` throughout cleanup. All
  six crossed optimizer initialization and emitted advancing finite training
  metrics; their observed RSS was about 84--86 GiB against a 96-GiB job
  limit.

## Countdown E17 dual and E4 7B fixed recovery 2026-07-20 ~12:44

- Recovered the three Countdown-3B Standard MaxEnt Haarnoja-dual seeds under
  the frozen E17 protocol. Original seeds 43/44 jobs 30015737/30015740 were
  zero-second wrapper failures; seed 45 job 30015743 was a genuinely wedged
  writer whose metrics had stopped after step 651 following a shared-storage
  error. The stale writer was cancelled, while its complete step-576 rolling
  optimizer checkpoint was retained for automatic resume.
- Submitted append-only replacement jobs 30020758--30020760. At the recovery
  audit, seed 43 was advancing at step 155 with finite dual/entropy telemetry
  and all NaN/Inf sentinels zero; seeds 44/45 were pending on the constrained
  A6000 pool. Pending placement was widened from nodes 207--208 to identical
  A6000 node 205 as well, without changing the protocol. Seeds 43/44 start
  clean and seed 45 will resume step 576 when it receives an allocation.
- Recovered Countdown-7B fixed-xDr seeds 43/44 from the frozen E4 source. The
  earlier jobs 30008062/30008063 had rejected the newer, inert
  `--maxent-objective` selector. `ops/train.sh` now capability-gates that
  selector: old source may omit it only for non-MaxEnt runs, while a MaxEnt
  request against unsupported source still fails closed.
- Startup-only jobs 30020761--30020764 exposed and isolated two configuration
  hazards before training: expandable allocator segments conflict with the
  vLLM CuMem sleep pool, while a resident 7B actor cannot use a 0.10 memory
  ratio because the actor weights alone exceed that budget. Their watchdog
  requeues were disabled and the jobs were cancelled. Resident-vLLM attempts
  30020765/30020766 then proved the evaluator and learner initialized, but
  seed 44 OOMed on the first 14.19-GiB backward allocation; both were cancelled
  rather than accepting an input-length-sensitive or cross-seed protocol.
- Final jobs 30020817/30020818 exactly restore the successful E4 seed-45
  memory protocol: two A6000s, vLLM sleep enabled, ratio 0.25, global batch
  32, microbatch 16, optimizer/activation offload, and no expandable allocator.
  Both completed step-0 Countdown evaluation and a full backward/sleep-wake/
  parameter-sync cycle. At the audit they were advancing at optimizer steps
  2 and 1 with no OOM, traceback, or numerical failure. Evaluation remains
  scheduled every 96 prompts, exactly one quarter of the 384-prompt epoch.
- Focused recovery contracts passed (5 tests), as did shell syntax and Python
  lint checks. Append-only manifests retain every discarded attempt for audit.

## Graph-coloring 3B idle-GPU canaries 2026-07-20 ~14:38

- Audited the attached campaign table against live Slurm allocations and host
  memory. Node302 had eight idle mltheory A100s, while node204 had six idle
  A5000s and enough real memory for one additional 96-GiB allocation. Existing
  A6000 capacity was already memory-bound or reserved for submitted recoveries.
- The latest graph-coloring-3B feedback seed-45 and Haarnoja-dual retries had
  not failed in training: both were rejected at startup because frozen E4/E5
  source predates the inert `--maxent-objective` selector. Submitted held,
  audited replacements with the frozen capability-gated runtime and preserved
  the five-pass objective, global batch 32, and quarter-pass evaluation.
- Feedback seed 45 job 30021274 started immediately on two node302 A100s. It
  cleared evaluation and optimizer startup and was advancing at step 29 with
  finite telemetry and no traceback or memory error at the audit.
- Haarnoja seed 43 canary 30021275 started on two node204 A5000s but OOMed on
  the first backward pass at microbatch 16. Its watchdog requeue was disabled
  and it was cancelled. Replacement 30021389 keeps global batch 32 while
  reducing the per-device microbatch to four; it is pending on the same A5000
  hardware in `lowprio/mltheory`, with Slurm's current estimate at
  2026-07-21 10:40 EDT. Seeds 44/45 were intentionally not duplicated before
  this canary proves the repaired memory layout.
- Focused recovery and frozen-source compatibility tests passed (3 tests), as
  did shell syntax, Python lint, and whitespace checks.

## Completed canonical checkpoint cleanup 2026-07-20 ~15:02

- Revalidated every deletion candidate against the 28 live or pending Slurm
  run stamps immediately before removal; no candidate belonged to an active
  job.
- Removed only 33 completed-run `saved_models`/`checkpoints` directories from
  Countdown and graph-coloring E16/E19 0.5B canonical campaigns and the fully
  completed graph-coloring E17 3B Standard MaxEnt campaign.
- Reclaimed 79,786,065,132 bytes (74.3 GiB). All 759 retained training-metric
  and evaluation-result files remained present, so dashboards, trajectory
  figures, and final-result analysis remain reproducible from landed outputs.
- Shared storage reported 2.7 TiB free and 63% utilization after cleanup.

## Countdown E17 node302 A100 backfill 2026-07-20 ~15:14

- Node302 was not empty: graph-coloring-3B feedback job 30021274 occupied two
  A100s. The Countdown E17 launcher was still pinned to the A6000 pool
  (nodes 103/104/208), which is why its remaining jobs did not consider the
  six otherwise-idle A100s. A live check through job 30021274 showed about
  396 GiB host memory available before the new startups.
- Retargeted the existing pending Countdown-3B Standard MaxEnt fixed jobs
  30020671/30020672 (seeds 43/44) and proportional job 30015742 (seed 45) to
  one node302 A100 and 96 GiB RAM each. They started and loaded their exact
  rolling optimizer checkpoints at steps 288, 384, and 672. Healthy fixed
  seed 45 job 30020673 remained untouched on node103.
- Added a narrow held-and-audited proportional recovery path for seeds 43/44,
  whose earlier landed model snapshots had no optimizer state. Fresh
  same-seed jobs 30021732/30021733 target one node302 A100 and 96 GiB each.
  Seed 43 started from initialization and advanced through step 5 with finite
  canonical overlap telemetry; seed 44 remains scheduler-safe pending on host
  memory rather than oversubscribing the node.
- Node302 now reserves 480 GiB and six GPUs across five running allocations:
  the existing two-GPU feedback job plus four one-GPU Countdown jobs. The
  fifth Countdown retry waits on resources. All E17 jobs retain evaluation
  and rolling checkpoints every 96 prompts, exactly one quarter of the
  384-prompt epoch. Startup scans found no traceback, OOM, NCCL failure, or
  numerical sentinel, and the focused recovery contract passed (3 tests).

## E22 free-form dual placement recovery 2026-07-20 ~20:30

- The six frozen E22 0.5B free-form conditional-token Haarnoja-dual jobs
  30024472--30024477 were pending indefinitely because their requested A5000
  node, node105, was unavailable. No duplicate jobs or run stamps were
  created.
- Retargeted the existing pending jobs to `lowprio` node025 with one RTX 3090
  and 64 GiB of host memory per job. The RTX 3090 has the same 24-GiB memory
  class and Ampere architecture as the frozen A5000 target; model, source,
  data, seeds, objective, controller, budgets, and evaluation/checkpoint
  cadence are unchanged. This is an operator-authorized placement exception,
  not a scientific-protocol change.
- All three graph-coloring jobs (30024472--30024474; seeds 43--45) and all
  three Countdown jobs (30024475--30024477; seeds 43--45) began running on
  node025. The node allocated six of ten GPUs, 96 CPUs, and 384 GiB of RAM.
  Initial logs attest the frozen identity, fresh initialization, correct
  task-specific pools, and CUDA detection; the first startup scan found no
  traceback, OOM, NCCL, or storage failure.

## E23 canonical Countdown 7B launch 2026-07-20 ~23:00

- The dashboard's held Countdown-7B Standard MaxEnt rows were not released
  under the failed free-form E11 namespace. Froze a separate E23 canonical
  protocol using the immutable E16/E17 source, Countdown codec, treatments,
  seeds, five-pass budget, and quarter-pass evaluation/checkpoint cadence.
- Initial v1 jobs 30031135--30031143 exposed a pre-training argument mismatch:
  canonical learner sampling requires rollout batch one. V2 jobs
  30031150--30031158 passed that check but exposed OAT's complementary rule
  that a two-GPU actor requires a rollout batch divisible by two. Neither
  namespace produced an evaluation or optimizer update; both cohorts were
  held and cancelled, with manifests and logs retained.
- E23 amendment A2 records the compatible topology: one node302 A100, one
  learner/actor GPU identity, rollout batch one, global batch 16, microbatch
  four, ZeRO-2, vLLM sleep, and optimizer/activation offload. Model, source,
  data, methods, seeds, coefficients, entropy target, and analytical budget
  are unchanged.
- Submitted, held-audited, and released the complete v3 cohort: fixed,
  proportional, and Haarnoja-dual seeds 43--45 are jobs
  30031175--30031183. Fixed seed 43 started immediately, passed both prior
  runtime constraints, loaded the 7B learner and actor, verified the canonical
  108-action support, and entered ZeRO optimizer initialization. The other
  eight jobs were released and remained pending for node302 resources at this
  audit.

## E24 canonical graph-coloring 7B launch 2026-07-21 ~00:54

- Replaced the dashboard's held E11 free-form namespace with a distinct E24
  canonical graph-coloring protocol. E24 transfers the frozen E16/E17
  27-action codec, three Standard MaxEnt treatments, seeds 43--45, group size
  16, five-pass budget, and quarter-pass evaluation/checkpoint cadence to the
  pinned Qwen2.5-7B-Instruct snapshot.
- Reused E23's validated one-A100 7B topology on node302: one collocated
  learner/actor GPU identity, rollout batch one, global batch 16, microbatch
  four, ZeRO-2, optimizer/activation offload, and vLLM sleep at ratio 0.25.
- Submitted all nine jobs held, audited exactly one allocation per method/seed
  cell plus every scientific identity and resource invariant, then released
  jobs 30031621--30031629. At the live audit all nine were `PENDING` for node
  availability, none had `JobHeldUser`, and the campaign dashboard showed
  `P / P / P` for fixed, proportional, and Haarnoja-dual graph-coloring 7B.

## Countdown-3B matched Dr.GRPO seed-43 fresh recovery 2026-07-21 ~01:11

- Diagnosed job 30020609's persistent `R init` state as an ineffective
  restart loop. Its step-480 DeepSpeed directory contained the model-state
  file but not `bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt`; every restart
  raised `FileNotFoundError` before evaluation or training. Slurm recorded 18
  restarts while the allocation continued reserving one node104 A6000.
- Added a fail-closed E18 seed-43 recovery path that explicitly disables
  automatic checkpoint discovery. Replacement job 30031645 was submitted
  held and audited against the frozen E18 run stamp, seed, task, model, A6000
  resource request, and fresh-start marker before the prior writer was
  touched.
- Disabled requeue on 30020609 and cancelled it, then released 30031645. The
  replacement started on node103 from the pinned pretrained model, completed
  the step-0 canonical Countdown evaluation, entered training, and emitted a
  finite step-1 record with no checkpoint-load error, traceback, or OOM.

## Active-safe model-state cleanup 2026-07-21 ~10:00

- Recomputed the cleanup plan directly from the live Slurm queue and protected
  every active run root, including E18 replacement 30031645, all E23/E24 7B
  jobs, and the active E21/E22 cohorts. Datasets, metrics, evaluations,
  manifests, and each inactive analytical run's highest final export remained
  outside the deletion set.
- Removed 80 inactive payloads: 39 raw optimizer-checkpoint trees, 25
  superseded-attempt model exports, and 16 redundant preterminal exports.
  The completed report records 663 files, 1,098,203,072,301 logical bytes, and
  1,373,831,094,272 physically allocated bytes removed.
- Shared storage changed from 5.2 TiB used / 1.9 TiB free (75%) to 4.3 TiB
  used / 2.8 TiB free (61%). A post-cleanup queue audit confirmed all live and
  pending jobs remained present; every reported target was absent.

## E25 3B free-form conditional-token dual launch 2026-07-21 ~10:07

- Froze a treatment-only 3B extension of E22-v2 for graph coloring and
  Countdown easy3. It preserves unrestricted `qwen_boxed` generation,
  `conditional_token_mean`, group size 16, the base-preserving alpha interval
  `[0.000075, 0.00015]`, and E22-v2's fixed task-specific entropy targets.
  No new matched free-form Dr.GRPO control was launched or claimed.
- Selected one node302 A100-80GB per job with 96 GiB host memory, ZeRO-2,
  optimizer/activation offload, vLLM sleep, global batch 16, and backward
  microbatch one. The model is the pinned Qwen2.5-3B-Instruct snapshot and the
  tokenizer was verified byte-identical to E22-v2's 0.5B tokenizer.
- Submitted all six jobs held and audited task, model, method, seed, objective,
  controller target/bounds, five-pass budget, cadence, and resource request
  before release. Graph-coloring seeds 43--45 are 30033851--30033853;
  Countdown seeds 43--45 are 30033854--30033856. All six were pending for
  node302 availability and appeared as `P / P / P` in their new 3B dashboard
  rows at the launch audit.
## E23/E24 four-A100 canonical 7B recovery 2026-07-21 ~10:29

- Retired the ineffective one-A100/96-GiB E23 jobs 30031175--30031183 and E24
  jobs 30031621--30031629 after their repeated ZeRO CPU-optimizer cgroup OOM
  loops produced no optimizer update or eligible outcome. Five E24 wrappers
  caught the first cancellation and requeued with a delayed begin time; their
  Slurm requeue flag was disabled and they were cancelled again, after which
  none of the superseded jobs remained live.
- At the user's direction, amended each individual 7B run to use four node302
  A100s and 192 GiB host RAM. The derived frozen source
  `a02e2d242f797d65d788cfbcf8e278e675031d2f10de31edfca5cf88e07b4b43`
  replicates the current prompt/full G=16 group across learner ranks for
  advantage construction, then partitions the group into four disjoint
  microbatches of four for one synchronized global-batch-16 update.
- Replaced the opaque inherited topology with an explicit frozen submitter
  surface (`72abacc11178931c3d3923c691fa303d06e45aec0dfef686e87ec607ef4bd808`)
  so GPU count, actor topology, rollout layout, batch sizes, offload, vLLM,
  resume, and watchdog settings are visible in each Slurm SubmitLine before
  release. Held draft jobs 30033880--30033888 were cancelled without running.
- Submitted, audited, and released Countdown jobs 30033890--30033898 under
  `cde23_canonical_maxent_7b_v4_4xa100` and graph-coloring jobs
  30033899--30033907 under `gce24_canonical_maxent_7b_v2_4xa100`.
- All 18 replacements request `gres/gpu:a100:4` and 192 GiB. They are queued,
  not held. Five valid E25 3B jobs 30033851--30033855 currently occupy five
  node302 A100s/480 GiB and were deliberately left running; the 7B jobs will
  become runnable as those allocations finish.
- Updated the live monitor, curve refresh, and figure inputs to the replacement
  namespaces. The focused launcher/reporting suite passed (36 tests), 77
  canonical source tests passed directly against the derived snapshot, both
  configuration-only launch gates passed, and the live dashboard showed
  `P / P / P` for every E23 and E24 treatment row.
## E21 MATH-500 0.5B retirement 2026-07-21 ~10:33

- At the user's direction, stopped the unfinished
  `mte21_math_conditional_token_05b_v4` MATH-500 cohort because the one-pass
  free-form jobs were too slow and its original Haarnoja-dual formulation has
  been superseded by the base-preserving conditional-token dual used in
  E22/E25.
- Dr.GRPO seed 45 (30020403) and original Haarnoja-dual seed 45 (30020406)
  were already terminal. Disabled Slurm requeue and cancelled the ten remaining
  active jobs: 30020395--30020402, 30020404, and 30020405. Existing metrics,
  evaluations, and checkpoints were retained as incomplete exploratory
  artifacts rather than deleted or presented as completed outcomes.
- The exact retirement mapping is recorded in
  `var/artifacts/e21_math_05b_retirement_20260721.tsv`. Countdown,
  graph-coloring, E22/E25 base-preserving dual jobs, and all queued four-A100
  7B replacements were left untouched.

## E26 MATH-500 0.5B high-entropy dual launch 2026-07-21 ~10:42

- Launched only the corrected free-form conditional-token MaxEnt
  base-preserving Haarnoja-dual treatment for seeds 43--45. No replacement
  Dr.GRPO, fixed-MaxEnt, or proportional-MaxEnt jobs were submitted; the
  retired E21 runs remain visible as incomplete historical comparisons.
- Calibrated a fixed conditional-token entropy target of
  `0.3653633594512939` nats from the pooled E21 warmup entropy. This is 25%
  above the corresponding legacy pooled 80%-of-warmup target. Preserved the
  base/minimum alpha at `0.000075` and doubled the controller's alpha ceiling
  from `0.00015` to `0.00030`, with log-alpha Adam LR `0.005` and a 64-update
  warmup.
- Submitted held, audited the immutable objective, target, bounds, budget,
  dataset, model, seeds, and resources, then released jobs 30033993--30033995
  under `mte26_math_freeform_conditional_dual_high_entropy_05b_v1`. Each uses
  one node105 A5000 and 64 GiB; all three entered RUNNING and reached the
  step-zero MATH evaluation with clean startup logs.
- Added a separate E26 dashboard row so this exploratory higher-target cohort
  is not conflated with the superseded E21 dual. Launcher/reporting contract
  tests passed (24 tests). The frozen protocol and calibration are recorded in
  `paper/preregistration/e26_math_freeform_dual_high_entropy.md` and
  `paper/results/e26_math_freeform_dual_high_entropy_calibration.json`.

## E26 MATH-500 0.5B aggressive-entropy replacement 2026-07-21 ~10:49

- At the user's direction, replaced the conservative E26-v1 draft before any
  optimizer metric was written. Disabled requeue and cancelled jobs
  30033993--30033995 after they had performed only the step-zero evaluation;
  they contribute no training outcome.
- Raised the fixed conditional-token entropy target from the pooled warmup
  mean to `0.4567` nats (125% of that mean), expanded the base-preserving alpha
  interval to `[0.000075, 0.00060]`, and doubled the log-alpha Adam learning
  rate to `0.010`. With an explicitly configured target, this controller adapts
  from the first entropy observation despite retaining the 64-step warmup field.
- Validated configuration-only output and 24 launcher/reporting contract tests,
  then submitted held and audited seed-43--45 jobs 30033998--30034000 under
  `mte26_math_freeform_conditional_dual_high_entropy_05b_v2`. All three
  immutable records contain the requested target, alpha bounds, LR, objective,
  seeds, and one-A5000/64-GiB resources; all three entered RUNNING on node105.

## E25 3B aggressive-entropy replacement 2026-07-21 ~10:54

- At the user's direction, disabled requeue and cancelled the six weaker E25-v1
  graph-coloring/Countdown jobs 30033851--30033856. The three graph jobs had
  reached roughly update 40, Countdown seeds 43/44 roughly updates 39/66, and
  Countdown seed 45 was pending. Their partial artifacts are retained but are
  not merged with the replacement trajectories.
- Applied the E26-v2 exploration policy on each task's own entropy scale:
  graph coloring target `1.622718550885717` and Countdown target
  `1.347109432487438`, each 125% of its E22 pooled warmup reference. Both use
  base-preserving alpha bounds `[0.000075, 0.00060]`, log-alpha Adam LR `0.010`,
  and fixed-target adaptation from the first entropy observation.
- Configuration and reporting tests passed (31 tests). Submitted held, audited,
  and released graph jobs 30034001--30034003 and Countdown jobs
  30034004--30034006 under their `*_v2` prefixes. All six immutable records
  contain the requested objective, task-specific target, alpha bounds, LR,
  model, seed, budget, and one-A100/96-GiB request. They are queued on node302,
  not held, currently awaiting node availability: four-GPU 7B jobs 30033890
  and 30033891 claimed all eight node302 GPUs during the replacement handoff.

## Active visualization refresh for E25-v2/E26-v2 2026-07-21 ~12:15

- Redirected curve parsing, combined/split divergence figures, and live monitor
  inputs from the retired E25-v1 3B prefixes to the clean E25-v2 prefixes. The
  3B free-form panels now explicitly identify the 125%-target treatment and
  show `PENDING FIRST EVALUATION`; no partial v1 points are pooled into them.
- Reworked the MATH visualization to read the active E26-v2 aggressive dual
  while retaining only E21 Dr.GRPO as a clearly labeled historical reference.
  Cancelled E21 fixed, proportional, and old-dual curves are excluded. The
  active E26 curve already contains live training telemetry through roughly
  0.04 pass, with step-zero MATH evaluation points.
- Regenerated the combined grid, canonical-only, free-form-only,
  environment-split, and MATH PDF/PNG/preview artifacts. The MATH live output
  is now `paper/figures/e26_freeform_math_maxent_live.{pdf,png}`; the stable
  compute preview remains `var/artifacts/divergence_math_maxent_latest.png`.
  Figure-input and E25/E26 reporting contracts passed (34 focused tests), and
  a final figure-input check passed all 6 tests after the label update.

## E27 ModeBench 0.5B aggressive-entropy replacement 2026-07-21 ~12:27

- Launched new treatment-only 0.5B cohorts for graph coloring and Countdown
  using the same corrected conditional-token, base-preserving Haarnoja dual as
  E25-v2/E26-v2. The completed E22-v2 Dr.GRPO runs remain the historical
  matched controls; completed E22-v2 dual outcomes are no longer used as the
  active dashboard row or treatment figure curves.
- Set each fixed target to 125% of its task-specific pooled E22 warmup entropy:
  graph coloring `1.622718550885717` nats and Countdown
  `1.347109432487438` nats. Both cohorts use alpha bounds
  `[0.000075, 0.00060]`, log-alpha Adam LR `0.010`, seeds 43--45, and five
  passes over the prompt pool.
- Submitted all six jobs held, audited the immutable task, model, objective,
  target, alpha bounds, controller LR, seed, budget, and one-A5000/64-GiB
  request, then released them. Graph-coloring jobs 30034356--30034358 entered
  RUNNING on node105; startup telemetry confirmed the requested target and
  finite controller updates. Countdown jobs 30034359--30034361 are queued,
  not held, pending node105 availability.
- Redirected the monitor, curve refresh, and free-form figure inputs to the E27
  treatment prefixes while retaining only E22-v2 Dr.GRPO as the historical
  0.5B reference. The dashboard and figures explicitly label the active 0.5B
  treatment as `E27` with a `125% entropy target`; missing treatment data is
  shown as pending rather than silently filled from the old E22-v2 dual.
- Configuration-only launch validation passed, the focused launcher/monitor/
  figure-input suite passed all 33 tests, and the combined/split figures and
  latest previews were regenerated from the new routing.

## Retired E21 MATH optimizer-state cleanup 2026-07-21 ~12:55

- Removed 11 raw optimizer-checkpoint trees belonging exclusively to the
  explicitly retired `mte21_math_conditional_token_05b_v4` jobs. The removed
  trees occupied approximately 190 GiB; they are irreversible resume state
  for the superseded E21 runs and are not used by E26/E27.
- Retained all 12 E21 exported-model directories, metrics, evaluations, logs,
  run directories, and visualization inputs. Verified that the three active
  E26-v2 MATH roots and three active E27 graph-coloring roots remained intact;
  E27 Countdown had not allocated and therefore had no run root to protect.
- Shared storage moved from 4.3 TiB used / 2.8 TiB free (61%) to 4.2 TiB used /
  2.9 TiB free (59%). Slurm remained unavailable during the post-cleanup
  audit, so the separately authorized E27 Countdown placement amendment to
  nodes202/203 was not applied or assumed successful.

## E27 Countdown 0.5B placement recovery 2026-07-21 ~13:00

- Amended pending jobs 30034359--30034361 in place from
  `mltheory/node105` to `cs/allcs` with eligible nodes202/203, preserving job
  IDs, run stamps, source, data, seeds, objective, controller, and budget. The
  wall-time request was reduced from 24 hours to the previously validated
  four-hour backfill window.
- All three jobs allocated simultaneously on node202 before a follow-up CPU
  shape reduction could be applied. Their effective allocations therefore
  retain 16 CPUs, 64 GiB, and one A5000 each; the rejected post-allocation
  update did not alter the running jobs.
- Startup telemetry reached optimizer steps 16--17 across all seeds, logged
  the requested Countdown entropy target `1.3471094369888306`, and showed
  finite adaptive-alpha updates. The initial audit found no traceback, OOM,
  fatal error, or controller mismatch.

## E23/E24 four-A100 empty-shard recovery 2026-07-21 ~16:45

- Diagnosed the repeated pre-update 7B failures in source hash
  `a02e2d242f797d65d788cfbcf8e278e675031d2f10de31edfca5cf88e07b4b43`.
  The replicated-canonical path correctly constructed one disjoint
  four-candidate shard per learner rank, but its backward loop still iterated
  over the full replicated group length of 16. Each rank therefore issued one
  valid microbatch followed by empty microbatches; Qwen's zero-sized forward
  failed, and NCCL watchdog aborts followed. No affected job completed an
  optimizer update.
- Created immutable fixed source hash
  `044f6df047788dc8b67bbe224281a403c6d5eab04de89881f5a17cfe5c147cf9`.
  Its only behavioral diff makes the loop terminate at the rank-local shard
  length. A regression contract verifies four nonempty four-candidate shards
  exactly cover the global group of 16; both configuration gates and 39
  focused source/launcher/reporting tests passed.
- Submitted and held-audited Countdown replacements 30035422--30035430 under
  `cde23_canonical_maxent_7b_v5_4xa100_fix` and graph-coloring replacements
  30035431--30035439 under `gce24_canonical_maxent_7b_v3_4xa100_fix`. Disabled
  requeue and cancelled the 18 broken jobs 30033890--30033907.
- Released fixed seed-43 canaries 30035422 and 30035431; the other 16 jobs
  remain user-held until both canaries pass a real optimizer update. The
  canaries are scheduler-pending because five healthy E25-v2 3B treatments
  currently reserve five of node302's eight A100s and 480 GiB. They require
  four A100s and 192 GiB on that one node and will allocate once at least two
  of those 3B allocations finish.

## E28 post-hoc matched 3B free-form Dr.GRPO controls 2026-07-21 ~17:00

- At the user's direction, froze a fresh matched-control extension for both
  E25-v2 ModeBench environments. E28 changes only the entropy treatment:
  ordinary unrestricted free-form Dr.GRPO uses `alpha=0`, `xdr_tau=inf`, and
  no entropy or aggregation controller while preserving E25-v2's model,
  source and execution snapshots, datasets, group size 16, learning rate,
  five-pass budget, evaluation cadence, offload topology, and seeds 43--45.
- The comparison is explicitly post-hoc because E25-v2 had begun before E28
  was requested. Historical E1 3B Dr.GRPO remains contextual only and is not
  relabeled as matched evidence.
- Configuration-only validation passed for graph coloring and Countdown, and
  33 focused launcher, monitor, and figure-routing tests passed. Submitted all
  six jobs held, audited their resolved scheduler records, and released the
  complete cohort atomically: graph coloring 30035541--30035543 under
  `gce28_freeform_drgrpo_3b_v1`; Countdown 30035544--30035546 under
  `cde28_freeform_drgrpo_3b_v1`.
- All six controls are scheduler-pending on node302 after release. The monitor,
  curve refresh, and free-form figures now pair E28 Dr.GRPO with E25-v2 rather
  than leaving the 3B panels treatment-only.
- A resolved SubmitLine diff for paired graph and Countdown seed-43 jobs found
  no non-treatment runtime mismatch. Differences were limited to run/protocol
  identity, `variant=grpo`, and removal/zeroing of the MaxEnt dual and its
  length-controller-only arguments.

## E23/E24 four-A100 runtime validation and release 2026-07-21 ~17:12

- Temporarily requeue-held active E25-v2 jobs 30034001--30034005 and held its
  sixth pending job 30034006 to make all eight node302 A100s available. The
  five interrupted jobs were only 8 minutes into their first allocation and
  remained intact as held Slurm records rather than being cancelled.
- Both corrected seed-43 canaries then allocated together with the requested
  topology: Countdown job 30035422 and graph-coloring job 30035431 each used
  four A100s and 192 GiB on the single node302 host. Four-rank learner and
  vLLM initialization completed without OOM or fatal NCCL errors.
- Countdown completed optimizer step 1 on all ranks at 17:11:44 EDT and was
  observed through step 3. Graph coloring completed optimizer step 1 on all
  ranks at 17:12:13 EDT and entered step 2. Neither reproduced the retired
  empty-microbatch traceback; both continued normally after the exact old
  failure boundary.
- Released replacement jobs 30035423--30035430 and 30035432--30035439 into
  the normal scheduler queue, then released E25-v2 jobs 30034001--30034006.
  The final audit showed the two canaries RUNNING and every other corrected
  7B and restored 3B job PENDING for resources, with no user-held jobs in
  either set. Scientific settings remained unchanged.
- Refreshed the campaign curves and all combined/split canonical, free-form,
  Countdown, and graph-coloring figures. The live monitor now follows the
  replacement prefixes and reports persisted seed-43 7B progress rather than
  the retired jobs; 32 focused monitor, figure-input, and source-contract
  tests passed after the runtime release.

## E28 Countdown matched-control recovery 2026-07-22 ~11:33

- Diagnosed terminal jobs 30035544--30035546 as a shared infrastructure
  interruption rather than an objective failure. All three allocations ended
  at the same instant with Slurm reason `ReqNodeNotAvail`; their logs stop
  mid-update without a traceback, OOM, or runtime error.
- Preserved the same E28 run stamps and complete optimizer state. Seeds 43 and
  44 resume from step 384 / 1.00 pass, and seed 45 resumes from step 864 / 2.25
  passes. Restored the protocol-pinned Qwen2.5-3B-Instruct revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1` after its shared-cache snapshot
  had been removed; no training artifact or checkpoint was replaced.
- Added an audited same-stamp Countdown recovery path to the E28 launcher.
  It submits replacements held, verifies task, model, objective, resources,
  run stamps, and checkpoint settings, and supports releasing an already
  staged trio without duplicate submission. The launcher syntax check and all
  three focused E28 contract tests passed.
- Submitted and released recovery jobs 30046076--30046078 on three idle
  node302 A100s. All three entered `RUNNING`; startup logs selected the exact
  expected step-384, step-384, and step-864 checkpoints, with watchdog requeue
  enabled for up to eight allocation restarts.

## 0.5B/3B free-form resume-contamination repair 2026-07-22 ~12:25

- Supersedes the E28 Countdown recovery immediately above. The restored
  learner state was not synchronized to actors before the resumed boundary
  evaluation or first rollout in the old frozen source. Jobs 30046076--30046078
  were stopped before accepting their output; artifacts were preserved for
  audit. Seeds 43 and 44 were already contaminated from step 96, so step 384
  was not a valid rollback point.
- Audited every E22-v2/E27 0.5B and E25-v2/E28 3B free-form seed. All eighteen
  0.5B task/arm/seed trajectories are single uninterrupted attempts from step
  zero through terminal evaluation and never entered the faulty resume path;
  they are certified clean and are not needlessly rerun.
- Frozen the remediation addendum
  `paper/preregistration/modebench_freeform_resume_repair_20260722.md`. Exact
  clean optimizer states remain only for E28 Countdown seeds 43, 44, and 45
  at steps 96, 96, and 864. The other nine 3B branches must replay from the
  pinned pretrained initialization because their qualifying optimizer states
  were pruned; model-only exports were explicitly rejected as substitutes.
- Added resume-before-eval actor synchronization and a one-time external
  bootstrap mechanism. On watchdog requeue, a repaired run follows its own
  rolling checkpoint; it uses the predecessor bootstrap again only if no new
  clean checkpoint exists. Two rolling optimizer checkpoints are retained and
  automatic success pruning is disabled pending repair validation.
- Rejected and cancelled held staging jobs 30046264--30046275 because critical
  recovery settings were inherited through `--export=ALL` and therefore absent
  from the auditable Slurm `SubmitLine`. They performed no training. Updated
  the shared submitter to pin those fields explicitly.
- Source hash
  `33fbde7130221b188ad251696345e37fd227f7ef3ce21c19cabcab1f80637359`
  and 11 focused repair/parser/sync tests passed. Submitted, held-audited, and
  released repair-v2 jobs 30046276--30046287 under four fresh prefixes. Nine
  jobs start from initialization; E28 Countdown jobs 30046285--30046287 use
  exact clean one-time step-96, step-96, and step-864 bootstraps. At release,
  five fresh-start jobs were running on node302 and seven released jobs were
  pending for resources/node availability. All five allocated jobs passed
  initialization and entered optimizer step 1 without a traceback, OOM, or
  NCCL failure. Storage had 4.5 TiB free.
- Updated the central curve refresh to clip the four abandoned 3B free-form
  prefixes seed-by-seed at the audited clean boundary, retaining the valid
  predecessor boundary evaluation and discarding all downstream contaminated
  points. Regenerated the combined, canonical, free-form, task-split, and 0.5B
  paper figures without smoothing. JSON validation passed, the four observed
  per-seed maxima exactly match the remediation table, and 19 focused figure,
  repair, parser, and three-seed-mean tests passed.

## E30 fixed-draw 0.5B free-form diagnostic 2026-07-22 ~13:45

- Made ModeBench evaluation reproducible by default: four K=8 draws with
  fixed seeds 1001--1004. The compatibility headline is their arithmetic
  mean; every raw draw, prompt outcome, normalized answer, reward, and seed is
  retained, with SD, SE, minimum, and maximum reported without smoothing.
- Completed the prospectively frozen terminal diagnostic for all three clean
  E22-v2 Dr.GRPO and E27 conditional-token MaxEnt seeds on Countdown and graph
  coloring. Correct vLLM V0 jobs 30046468 and 30046469 exited successfully.
  Partial output from a superseded V1 launch was quarantined before metric
  inspection rather than mixed into the analysis.
- Evaluation noise is too small to explain the visible bumpiness. MaxEnt
  pass@8 MC SD was 0.0050 on Countdown and 0.0040 on graph coloring, versus
  raw adjacent-checkpoint MAAD of 0.0703 and 0.0571 (about 14x larger).
- Terminal MaxEnt-minus-Dr.GRPO effects were +0.0944 pass@8 / +0.0286
  coverage@8 on Countdown and +0.2326 / +0.0612 on graph coloring. Greedy
  effects were unresolved, and graph mean@8 was unchanged. Generated the raw
  12-point-per-cell diagnostic figure and machine-readable findings.

## Compute-divergence evaluation contract and five-metric figures 2026-07-22 ~14:30

- Extended every combined, method-specific, environment-specific, and 0.5B
  compute-divergence figure to show deterministic pass@1 plus mean@8, pass@8,
  coverage@8, and distinct@8. Heavy curves remain all-three-seed means and
  thin curves remain unsmoothed seed trajectories.
- Merged the real E30 four-draw terminal results into all twelve eligible
  E22-v2/E27 0.5B free-form curve endpoints. Each curve row retains seeds
  1001--1004, mean/SD/SE/min/max, and links to all raw attempt and per-prompt
  traces. The matching deterministic greedy terminal traces were merged too.
- Audited 1,574 plotted multi-answer rows across twenty curve artifacts. All
  1,574 contain the five requested metrics; twelve have genuine four-draw
  uncertainty. The remaining 1,562 historical points are visibly identified
  as legacy single evaluations because their intermediate model exports were
  not retained; no uncertainty was fabricated.
- Future inline sidecars now retain a dedicated deterministic K=1 pass@1 trace
  plus raw prompts, references, generated response text, rewards, normalized
  answers, and draw identities for every K=8 draw. Wrote the
  machine-readable coverage audit to
  `var/artifacts/compute_divergence_eval_coverage.json`. Seventy-seven focused
  tests passed across the plotting and training environments; lint and visual
  inspection passed.

## E31 responsive entropy-EMA dual default 2026-07-22 ~15:15

- Changed the default sensor for new Haarnoja `maxent_dual` runs from a raw
  one-prompt-group entropy observation to an EMA with decay 0.7. This is a
  deliberately responsive smoother: about two updates of half-life and six
  observations of effective averaging, rather than the roughly 22-update 90%
  settling time of decay 0.9.
- The raw observation remains logged. Added separate EMA entropy, EMA decay,
  and EMA-based error telemetry; only the error passed to log-alpha Adam is
  smoothed. Alpha learning rate, target, projection bounds, and objective are
  unchanged so the intervention is isolated.
- Persisted EMA state and decay in controller checkpoints and versioned the
  rule as `log_alpha_adam_entropy_ema_v2`. Historical instantaneous-feedback
  checkpoints fail closed instead of silently resetting or changing their
  dynamics. Frozen E16--E29 runs and active repairs retain their original
  sources and are not retroactively relabeled.
- Pinned `OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7` through the public runtime,
  low-level trainer, and auditable comparative Slurm submit line. The
  prospective E31 method note was frozen before new compute.

## E32 clean matched 0.5B free-form rerun 2026-07-22 ~14:52

- Froze a new prospective matched design for Countdown and graph coloring:
  Dr.GRPO versus conditional-token EMA-Haarnoja MaxEnt, seeds 43/44/45, ten
  prompt-pool passes, the prior 125% targets, EMA decay 0.7, and quarter-pass
  evaluations. Every evaluation retains deterministic pass@1 and four fixed
  K=8 draws with all raw traces and mean/SD/SE/min/max.
- Restored the exact pinned Qwen2.5-0.5B-Instruct revision after cache cleanup;
  no trained checkpoint or floating model revision was substituted. Froze
  source hash `69a22e21276aa04bea617e4539afdd57a2056ef97b5b08fd925c710c4f561c0a`
  and execution-surface hash
  `b2cc8619554d115213440043a605362a3a9ec749ae40a89b95f11587507b1f65`.
- A first held v1 submission was rejected because its audit incorrectly
  required the active treatment target on inactive Dr.GRPO controls. All
  twelve held jobs 30047228--30047239 were cancelled before allocation. The
  audit was corrected to verify arm-specific invariants and to cancel the
  complete cohort on any failure.
- Submitted, held-audited, and released E32-v2 jobs 30047240--30047251. The six
  graph-coloring jobs allocated on node105 and all produced step-0 repeated
  evaluations plus optimizer metrics without traceback, OOM, or runtime
  failure. The six Countdown jobs initially remained pending because those
  graph jobs' 16-core requests exhausted all 96 schedulable node105 CPUs while
  four A5000s were idle. Live MaxRSS was only 12.7--13.0 GiB and average CPU
  use was about one core, so the pending jobs were amended in place to four
  cores, 32 GiB, and the same-model A5000 low-priority pool on nodes 202--204.
  All six then allocated immediately; no scientific setting or run identity
  changed. The scheduler amendment is recorded in
  `var/artifacts/e32_freeform_05b_ema_10ep_v2_placement_amendment.json`.
  Runtime logs attest terminal-only export, per-pass resumable
  checkpoints (keep two), no success pruning, and the frozen protocol identity.
- Simplified `make monitor` to the four newest 0.5B rows only and a dedicated
  one-minute E32 curve/figure refresh. The first implementation republished
  only `freeform_05b_latest`, leaving the established combined, free-form, and
  task-split files stale even while metrics advanced. Corrected the refresher
  to atomically republish every E32-consuming divergence figure each minute.
  Published `freeform_05b_latest` and rerouted all compute-divergence 0.5B
  free-form panels to E32. Heavy lines require all three seeds; thin seed
  traces remain unsmoothed and every raw draw remains retained. The focused
  53-test regression suite, follow-up monitor/storage
  tests, lint, shell syntax, figure refresh, and visual inspection passed.
- After launch, replaced the plotted four raw-draw dots and min/max whiskers
  with light 95% Student-t bands for Monte Carlo evaluation uncertainty. Each
  fixed draw is first averaged across the three training seeds; the interval
  uses the four draw-level means and df=3. Thin seed trajectories remain
  unsmoothed, pass@1 remains deterministic with no band, and every raw draw
  remains in the artifacts. The visualization-only decision is frozen in
  `paper/preregistration/e32_visualization_amendment_20260722.md`.

## E51 projection-free policy-entropy canonical restart 2026-07-24 ~13:46

- Replaced E50 after its live uncapped Haarnoja log-alpha controller grew far
  above the reference dose in graph coloring. E50 remains immutable,
  post-selection historical evidence and none of its checkpoints, optimizer
  state, canonical bank, controller state, or partial trajectories enters E51.
- Froze a fresh three-domain paired design at 0.5B: Dr.GRPO versus
  `online_canonical_policy_entropy`, seeds 43--45, group size 16, and 50
  complete passes for graph coloring, Countdown easy3, and executable Python
  factors. The controller observes the learner's masked-mean full-vocabulary
  token entropy, holds alpha at 0.10 for 64 observations, then applies
  `alpha_next = 0.10 * entropy_ema / warmup_mean` with EMA decay 0.9.
  It has no Haarnoja loss, Adam or other accumulating optimizer state, target
  bank entropy, lower projection, or upper projection.
- Focused validation passed: 119 controller/argument/bank/parser/E51 tests,
  14 historical E44/E46/E48/E50 compatibility tests, Ruff, shell syntax,
  whitespace checks, and configuration-only audits for all three domains.
  Frozen identity source hash is
  `63f79255eb66096e37b7ade57295983b8deaf1d305f3a335adef0efd02993667`;
  execution-surface hash is
  `299f42d22e6dbeb2051f2af1e009d7cce43867c53fdf15be4da6d88fbbfda82d`.
- Cancelled the exact 18 E50 jobs from their three manifests. Seven running
  jobs initially reappeared pending via the old watchdog; cancelled those
  requeues too and verified that no E50 job remains in the active queue.
  Artifacts were retained.
- Submitted all E51 jobs held, audited arm-specific objective isolation and
  the complete frozen environment, then released the cohort atomically:
  graph coloring 30074925--30074930, Countdown 30074931--30074936, and Python
  factors 30074937--30074942. At the first post-release check graph jobs
  30074925--30074927 were running on node302 and the other jobs were released
  and scheduler-pending.
- Runtime validation crossed the first adaptive update. Treatment job 30074926
  reached observation 65 with warmup reference 0.79596, entropy EMA 0.66423,
  normalized score 0.83451, and next alpha 0.08345. This exactly matches the
  projection-free registered formula. Its startup log explicitly attests the
  new controller, and no startup traceback or runtime error was present. The
  current-canonical monitor, parser, refresh route, and live figure now follow
  E51 rather than the cancelled E50 cohort.

## E51-only monitor and GPU-capacity audit 2026-07-24 ~14:04

- Removed the cancelled E45 MathIR rows from the current-canonical monitor and
  refresh route. The live view now contains exactly the 18 E51 runs in six
  method/domain rows, reports no historical cancellations in its completion
  denominator, and links `paper/figures/e51_current_canonical_05b_live.png`.
  Seventy focused monitor/plot tests and Ruff passed; the live scheduler-backed
  snapshot showed three graph jobs running, fifteen E51 jobs pending, and no
  terminal failures.
- Read-only Slurm inspection found no 64-GiB slot on node302 despite one
  unallocated A100: seven of eight GPUs and 480 GiB are allocated. MLTheory
  has two unallocated A5000s on node105 and two RTX 2080s on node915; both
  accept an immediate one-node 64-GiB placement through `lowprio` in
  `srun --test-only`.
- The allcs-accessible 3090 nodes node020/node022/node023/node024/node026 have
  7/5/3/2/4 unallocated GPUs respectively (21 total at inspection time).
  A one-node allcs `lowprio` 3090 placement tested as immediately runnable.
  Node101 also has two unallocated A40s and accepted the same immediate
  test-only route. At this audit boundary, the E51 jobs were unchanged and no
  scientific placement had yet been altered.

## E51 Countdown/Python RTX 3090 placement 2026-07-24 ~14:13

- Following the user's explicit direction to use the available GPUs, froze
  `paper/preregistration/e51_rtx3090_placement_amendment_20260724.md`.
  Confirmed Countdown jobs 30074931--30074936 and Python jobs
  30074937--30074942 were all pending with zero runtime, zero restarts, and no
  allocation, then user-held all twelve before any scheduler mutation.
- Amended the held jobs in place to account `allcs`, partition `lowprio`, one
  RTX 3090 from node020/node022/node023/node024/node026, eight CPUs, 64 GiB,
  and the original seven-day limit. Every resolved record retained the
  original run stamp, frozen source and execution snapshots, arm, seed, model,
  data, objective, checkpoint settings, and protocol identity. All twelve
  held audits passed before atomic release.
- All jobs allocated: Countdown 30074931--30074936 on node020; Python
  30074937 on node026, 30074938--30074939 on node024,
  30074940--30074941 on node026, and 30074942 on node022. Startup logs were
  clean. Treatment telemetry reached optimizer step 6 on Countdown and step 2
  on Python with finite policy entropy and alpha 0.10 during the registered
  64-observation warmup. Graph coloring remained on the original A100
  placement.

## E51 live-figure compatibility repair 2026-07-24 ~14:23

- Diagnosed a stale viewer path rather than a failed refresh loop. The active
  E51 curve JSONs and `e51_current_canonical_05b_live.png` were updating each
  minute, but the earlier `e45_e51_current_canonical_05b_live.png` filename
  stopped changing after the E51-only monitor rename.
- The renderer now atomically republishes the earlier E45/E51 filename as a
  byte-identical compatibility alias on every E51 refresh. Updated Makefile
  help and operations documentation to describe the six-row E51-only monitor.
  Fifty-eight focused monitor/figure tests and Ruff passed. A live refresh
  produced identical SHA-256 hashes and timestamps for the canonical PNG and
  compatibility alias.

## E51 inverse policy-entropy corrected relaunch 2026-07-24 ~14:46

- Audited the live E51-v1 telemetry after policy entropy visibly collapsed.
  The logged EMA and warmup reference reproduced the controller arithmetic to
  within `6e-8`; this was not a parser, plot, or EMA defect. Graph treatment
  had fallen from a `0.79596` warmup reference to an EMA near `0.099`, while
  the direct multiplier simultaneously reduced alpha near `0.012`. Countdown
  showed the same direction at lower severity; Python remained near reference.
  Matched controls also lost entropy, but the direct response weakened
  canonical pressure precisely when the requested entropy-preserving response
  required it to strengthen.
- Superseded and cancelled E51-v1 jobs `30074925--30074942`; their artifacts
  remain diagnostic-only and are not pooled. Froze the user-directed E51-v2
  correction with the memoryless inverse rule
  `alpha_next = 0.10 * warmup_mean / entropy_ema`. The rule has no optimizer,
  lower projection, upper projection, or numerical epsilon. A nonpositive
  post-warmup EMA fails closed because the unbounded inverse is undefined.
  Controller checkpoint identity is versioned
  `unprojected_warmup_inverse_policy_entropy_alpha_v2`, so v1 state cannot be
  resumed accidentally.
- All focused controller, E51 contract, monitor, and plot checks passed
  (78 tests), as did shell syntax and all three configuration-only launch
  audits. The frozen source hash is
  `ecf5396f4be9b50bf20a85bbd8eb8fb5e8232d9863c3e7434d729b8be964c4fc`;
  execution-surface hash remains
  `299f42d22e6dbeb2051f2af1e009d7cce43867c53fdf15be4da6d88fbbfda82d`.
- Submitted, held-audited, and atomically released the fresh jobs: graph
  `30075025--30075030`, Countdown `30075031--30075036`, and Python factors
  `30075037--30075042`. All twelve Countdown/Python jobs allocated immediately
  on the frozen `allcs/lowprio` RTX 3090 pool. Three graph jobs allocated on
  node302 A100; the other three were scheduler-pending because seven of its
  eight A100s were allocated and the remaining slot was planned/reserved.
- Republished the E51-only monitor and live figure against the v2 prefixes.
  The first scheduler-backed snapshot contained 15 running, three pending,
  zero terminal failures, and no superseded v1 rows. Initial v2 treatment
  telemetry was finite and held alpha at `0.10` during the registered
  64-observation warmup. Graph treatment `30075026` then crossed the first
  adaptive update with warmup reference `0.74498`, entropy EMA `0.60606`,
  inverse multiplier `1.22922`, and next alpha `0.12292`. Thus the live
  response raises alpha when policy entropy falls, matching the corrected
  unbounded inverse rule end to end.

## E51 terminal Python audit and 50-pass figure repair 2026-07-26 ~13:45

- Audited all six Python-factor E51-v2 jobs after the cohort settled. Slurm
  accounting reports `COMPLETED` with exit code `0:0` for jobs
  `30075037--30075042`; each arm/seed contains all 19,200 optimizer steps and
  201 evaluations from pass 0 through pass 50. The 2--4 recorded restarts per
  job were preemptions recovered by the registered resume path, not missing
  endpoints or terminal failures.
- The visible anomaly is real but scientific rather than operational. Python
  policy entropy collapsed in both arms. Treatment seeds 43/44 ended with EMA
  entropy `0.000391/0.000537` and unbounded alpha `260.64/176.64`; seed 45
  ended at EMA `0.1484`, alpha `0.6887`, after an earlier maximum alpha
  `54.03`. Controls also ended near zero token entropy, so the controller did
  not originate the collapse.
- The canonical actuator could not oppose it. Treatment seeds 43 and 44
  discovered exactly one verified outcome per tracked prompt and emitted zero
  canonical entropy advantage on every post-warmup update. Seed 45 finished
  with mean support `1.0115` and had a nonzero entropy advantage on only
  `0.258%` of post-warmup updates. Thus the inverse sensor correctly increased
  alpha, but multiplying an absent canonical-bank advantage produced no
  entropy-preserving gradient. All six terminal quality vectors were identical
  (`pass@1 = pass@8 = mean@8 = coverage@8 = distinct@8 = 0.171875`).
- Diagnosed a separate visualization bug: the E51 renderer declared a 50-pass
  budget, but the shared loader discarded points after its historical global
  10-pass limit. Added an explicit per-renderer `max_training_epochs` argument,
  passed each E51 row's frozen `max_passes=50`, and retained the 10-pass default
  for historical consumers. The 19-test plotting suite, Ruff, and whitespace
  checks passed. Republished and visually verified both E51 figure filenames
  with axes and data through pass 50.

## E52 direct inverse-entropy sentinel 2026-07-26 ~14:35

- The cross-domain E51 audit showed the same structural mismatch everywhere:
  policy entropy was the sensor, but the adaptive coefficient multiplied a
  canonical-bank advantage that is zero before a second verified mode exists.
  Python stayed in that dead zone; Countdown and graph coloring received
  delayed impulses whose scale was unrelated to the token-entropy sensor.
- Froze E52 to separate the jobs. A projection-free inverse controller now
  multiplies direct conditional content-token entropy at every visited prefix,
  with EOS removed and sampled state visitation detached. It holds
  `lambda=0.000075` for 64 observations and then applies
  `lambda_next=0.000075*warmup_mean/entropy_ema`, with no lower or upper
  coefficient projection. The hybrid independently retains a fixed
  validator-bound canonical coefficient `0.10` and novelty coefficient
  `0.50`; neither evaluation labels nor bank support enter the direct
  controller.
- The first held-audited v1 submission (`30124287--30124295`) exposed an old
  argument-validation guard that prohibited every direct-MaxEnt/canonical-bank
  composition. All nine jobs were cancelled before reuse or pooling. A frozen
  execution amendment permits only inverse `conditional_token_mean` plus a
  fixed canonical coefficient; fixed/sequence direct MaxEnt and either
  canonical adaptive controller remain rejected.
- The fresh v2 jobs are graph `30124298--30124300`, Countdown
  `30124301--30124303`, and Python factors `30124304--30124306`. All nine are
  running across node302 A100 and the allcs RTX 3090 pool. Their source hash is
  `2858707e3a83ebdb3491a2a68890bfa788743565cf3205d7baac23c8aba93cce`.
  The first live machine audit found zero arithmetic, sensor, finiteness,
  projection, fixed-bank, gradient, length, or EOS violations. Graph direct
  entropy had crossed warmup with reference `1.26682`, EMA `1.06909`,
  multiplier `1.18496`, and next alpha `8.88717e-5`.
- Replaced the current-canonical monitor with the nine E52 sentinel cells and
  republished `paper/figures/e52_current_canonical_05b_live.png`. Every panel
  uses the explicit full 0--50-pass x-axis. Added a prospective machine audit
  that checks exact inverse arithmetic, objective/sensor identity, unbounded
  projection telemetry, negative direct loss, fixed canonical alpha, finite
  policy gradients, EOS/length guardrails, trailing entropy retention, and
  the frozen last-eight-boundary behavioral gate.

## E52 early transfer checkpoint and terminal-audit hardening 2026-07-26 ~15:00

- All nine E52-v2 jobs and the independent audit/plot sidecar remained live
  with zero controller-arithmetic, finiteness, projection, fixed-bank,
  gradient, length, EOS, or validator violations. No run was stopped or
  retuned from an early evaluation boundary.
- At the matched pass-0.5 boundary, Countdown hybrid reached distinct@8
  `0.798828` and pass@8 `0.558594`, versus matched Dr.GRPO `0.595703` and
  `0.455078`. Python hybrid reached distinct@8 `0.220703` and pass@8
  `0.183594`, versus direct-only `0.191406` and `0.179688` and matched
  Dr.GRPO `0.011719` and `0.011719`. The hybrid's rolling verified discovery
  count was `32`, versus `18` direct-only and `5` control at the then-current,
  slightly unmatched live positions. Graph hybrid remained
  strong through pass 1.0 at distinct@8 `1.614583` and pass@8 `0.833333`.
  These are mechanism/transfer checks only; they do not satisfy the frozen
  50-pass late-collapse gate.
- Hardened `audit_e52_sentinel.py` so prompt consumption alone can no longer
  mark a run complete. A run now also needs finite distinct@8, pass@8, and
  mean@8 telemetry at its exact domain-specific pass-50 step. The change
  affects only the sidecar verifier, not frozen training processes. The
  focused E52/controller/argument suite passed all 117 tests and Ruff passed.

## E52 scale-free stability-gate amendment 2026-07-26 ~15:05

- Replayed the proposed completion semantics against the terminal E51 curves.
  This exposed two cases not excluded by simple treatment-versus-control
  distinct@8 dominance: Python's single-mode plateau had
  `distinct@8 == pass@8`, while graph treatments could finish above a weak
  control after retaining only `2--33%` of their own best rolling-eight
  distinct@8 mean.
- Before any E52 terminal window, froze a target-free amendment based on
  `X_t = distinct@8 - pass@8`. The final hybrid must have positive
  multiplicity excess and beat control's excess at least six of the last
  eight boundaries, and its last-eight distinct@8 mean must retain at least
  half of its own best rolling-eight mean. These checks use no gold mode
  count, reference multiplicity, maximum support, or domain-specific target.
- Corrected the live auditor so eight early boundaries are always reported as
  a provisional preview. A behavioral pass/fail now requires both control and
  hybrid to be complete with exact pass-50 telemetry, preventing an early
  Graph window from falsely terminating the monitor. All 121 focused
  E52/controller/argument checks and Ruff passed.
- Slurm still showed all nine training jobs and the independent sidecar
  running. The refreshed live audit remained `in_progress` with zero
  violations; no training state or coefficient was changed.

## E52 conditional Stage A authorization 2026-07-26 ~15:25

- Built a fail-closed Stage A path for fresh seeds `43,44,45` in all three
  domains and all three arms. It cannot submit until the engineering sentinel
  has exact pass-50 telemetry for every run, all controller/runtime checks
  pass, and each domain passes the preregistered target-free final-window
  multiplicity, control-dominance, quality, and self-retention gates.
- The approval artifact binds the sentinel identity, source and operations
  trees, protocols, sentinel launcher, and auditor by SHA-256. The Stage A
  verifier recomputes every binding immediately before submission. The
  launcher then submits exactly the frozen 27-run Cartesian cohort and an
  independent audit sidecar as one held-and-audited release; it does not
  resume or pool sentinel checkpoints.
- The Stage A auditor requires every individual seed and the three-seed mean
  trajectory to pass the same target-free gates. No gold number of modes,
  domain-specific support target, answer-count threshold, or evaluation
  statistic enters training or selects the entropy coefficient.
- Revalidated all three configurations without submitting jobs, checked shell
  syntax, and passed the focused authorization/audit suite. A missing
  pass-50 approval currently fails closed and creates no Stage A identity,
  manifest, or launch lock. The live sentinel watcher is armed to perform the
  one-shot launch only after a terminal approval appears.

## E52 early actuator checkpoint 2026-07-26 ~15:30

- All nine training jobs and the watcher remained `RUNNING`; exact log scans
  found no traceback, CUDA/OOM/NCCL/runtime failure, or non-finite model or
  controller quantity. The framework's printed `nan` elapsed-time fields are
  initialization placeholders and do not enter optimization.
- Graph's current provisional final-eight-shaped window passed every
  prospective behavioral check: hybrid distinct@8 beat control at `8/8`
  boundaries, multiplicity excess beat control at `8/8`, hybrid had positive
  multiplicity at `8/8`, and it retained `0.894` of its own best rolling-eight
  distinct@8 mean. This remains non-authorizing because the run is early.
- Python supplied the most diagnostic separation: the direct entropy arm
  recovered correctness but was still effectively single-mode, whereas the
  hybrid reached distinct@8 `0.34375` at pass@8 `0.171875`. Countdown hybrid
  reached distinct@8 `1.058594` at pass@8 `0.619141` by pass `1.0`.
  Consequently direct entropy is operating before bank multiplicity exists,
  while the fixed bank begins preserving discovered verified alternatives.
- The live figure was visually checked with a frozen `0--50` pass x-axis in
  every panel. Countdown and Python still had fewer than eight paired
  evaluation boundaries, so their behavioral gates correctly remained
  `pending`; no early outcome triggered a stop, retune, or launch.
## E59 executable MathIR global verified replay — 2026-07-26

- Added a frozen `mathir_action_menu_v1` domain with 384 train and 128 disjoint evaluation problems across four linear-equation families.
- Each prompt exposes six shuffled primitive actions. The submitted action IDs are expanded, executed by the deterministic MathIR interpreter, and canonicalized from the resulting state trajectory; prose labels cannot define or alter a mode.
- Exhaustive enumeration gives exactly five valid canonical solution modes per problem, while the learner receives no gold route catalogue or support feedback.
- E59 uses the latest E58 `verified_first_global_replay_canonical` objective unchanged: zero augmentation before discovery, one global verified replay group, verified-mass replay, known-mode balance, and no direct token entropy.
- Disclosed base probes were mixed: reward-bearing executable support was reachable, and one prompt naturally showed two verified modes at 64 samples, but the stronger multi-family support screen did not pass. The launched run is therefore a bounded engineering smoke, not a positive empirical result.
- Contract, grader, MathIR, and E58 regression tests passed (65 tests); both smoke and matched launch configurations passed.
- Smoke job `30125201` launched on one A100 on `node302`; the six-run matched Dr.GRPO/treatment cohort remains gated on a clean terminal smoke audit.
- Smoke job `30125201` completed all 384 updates and passed the fail-closed terminal audit with zero violations. First verified discovery was step 20; replay activated on all 365 post-discovery updates; terminal controller observations were 365 verified-mass, 82 open-set, and 18 multi-mode balance.
- Smoke held-out initialization/terminal metrics were pass@1 `0.0547 -> 0.1016`, any-correct@8 `0.2188 -> 0.2871`, distinct-correct@8 `0.2363 -> 0.2871`, and canonical coverage@8 `0.0473 -> 0.0574`. These authorize the matched test but are not a matched claim.
- Submitted and released the frozen six-job E59 matched cohort on one A100 per run: Dr.GRPO/treatment seeds 43, 44, and 45 are jobs `30125245` through `30125250`.
- Added the fresh E59 MathIR rows to the current ModeBench tracker and combined E58+E59 live figure. Lightweight job `30125255` refreshes curves, tracker output, and `paper/figures/e59_mathir_global_verified_replay_05b_live.{png,pdf}` every 60 seconds while the cohort is live.

## E64 held-out MATH-500 realism transfer — 2026-07-27

- Registered a separate external-validity/generalization track rather than
  calling ordinary MATH a fifth multi-mode ModeBench domain. Training uses the
  frozen first 384 MATH12K rows; all 500 MATH-500 rows are held out, with zero
  normalized problem overlap and byte/row-order hashes frozen prospectively.
- Added `math_verified_answer`: every `math_verify`-positive completion for a
  prompt maps to one `math_verified_answer:correct` key and every reward-zero
  completion maps to no key. This permits verified-mass replay while making
  known-mode balance structurally ineligible; answer formatting and free-form
  prose are not reported as reasoning modes.
- Froze E64 as three matched Dr.GRPO seeds and three literal-E58 seeds for 12
  passes, with full MATH-500 evaluation at passes 0, 2, 4, 6, 8, 10, and 12.
  The matched launcher fails closed until a one-seed, 96-update treatment
  smoke passes its source/data/job/checkpoint and singleton-actuator audit.
- The focused implementation/configuration surface passed 138 tests. Smoke
  job `30126939` started immediately on an RTX 3090 on `node020`. Untouched
  greedy MATH-500 accuracy was `0.3360` over all 500 rows.
- The first verifier-positive discovery occurred at update 2. From that
  update onward the live audit observed one scheduled singleton replay group,
  verified-mass raw score-gradient sum `-1`, and zero balance eligibility,
  loss, score gradient, and controller observations. The smoke remained live
  and violation-free when this entry was written.
- Smoke job `30126939` completed all 96 updates with zero violations. Replay
  activated on all 95 post-discovery updates; the terminal bank contained 66
  prompt-local exemplars with maximum support exactly one; known-mode balance
  retained zero observations. The verified-mass controller crossed warmup and
  changed its unprojected coefficient from `0.10`, ending at `0.08928`.
- The untouched/terminal full-MATH-500 greedy diagnostic was
  `0.3360 -> 0.3400` accuracy with mean response length
  `562.48 -> 551.74` tokens. This treatment-only smoke change is not a matched
  empirical claim.
- The supplemental terminal gate scanned all 291 saved model tensors
  (630,167,424 parameters) and found no nonfinite values.
- The RTX 3090 pool was saturated or draining at advancement time. All six
  matched 12-pass jobs were therefore held, audited, and released together on
  the idle 10-GPU A5000 `node105`: baseline/treatment seeds 43--45 are jobs
  `30126986--30126991`. Watcher job `30126994` refreshes the six-run audit and
  `paper/figures/e64_math500_realism_05b_12ep_live.{png,pdf}` every 60 seconds
  and exits immediately on a fail-closed violation.

## E65R1 five-domain terminal confirmation — 2026-07-27

- Froze a single paper contract over E61-R1, E64, and the prospective repair:
  ten fixed ModeBench checkpoints, seven fixed MATH-500 checkpoints, pass-12
  and trapezoidal-AUC primary summaries, all three seed points/ranges/paired
  deltas, and explicit cross-domain, repair, and realism interpretation gates.
- Added the E65 singleton escape: literal E58 plus at most one independently
  verified support-only alternate, eligible only after the unbounded
  open-set controller completes warmup, detects entropy below its own
  warmup reference, and the current verified bank has support exactly one.
  The actuator becomes ineligible at support two; no proposal row enters PPO.
  Twenty-five focused controller/counterfactual tests passed.
- The original 12 E65 jobs `30127067--30127078` exposed an execution-contract
  error at runtime validation: the submission wrapper had not pinned
  replicated free-form sampling and local one-GPU weight sync. All had zero
  optimizer progress. They were canceled with artifacts retained, and the
  zero-step amendment was recorded before changing the execution contract.
- E65R1 pins both flags by arm and audits both in every held SubmitLine.
  Its corrected jobs are Graph `30127478,30127479,30127481`, Countdown
  `30127483--30127485`, Python `30127486--30127488`, and MathIR
  `30127489--30127491`. Graph seed 43 allocated on A5000 node105 and crossed
  argument validation into actor/learner initialization without an exception.
- Read-only capacity tests showed that moving the non-MLTheory jobs to idle
  general RTX 2080 nodes would start later than their registered RTX 3090
  route, so Countdown/Python were unchanged. The five remaining zero-runtime
  MLTheory jobs were frozen in a placement amendment, held, changed to one RTX
  2080 on idle node915/node917, fully re-audited for identity and training
  flags, and released together. The allocation exposed a deterministic
  compatibility failure before any optimizer metric: the frozen bfloat16 vLLM
  path requires compute capability >=8.0, while RTX 2080 Ti is 7.5. A second
  frozen amendment forbade changing dtype; all five zero-step jobs were
  requeued-held, restored to their original A5000 request, re-audited, and
  released. Both hardware attempts remain in provenance.
- A final compatible-capacity audit found two immediately usable A6000 slots
  on node103 and three A100 slots on node302; both Ampere families support the
  frozen bfloat16 path, and the E61-R1 Graph/MathIR cohort already uses
  A6000s. Before mutation, the same five jobs remained pending with no
  optimizer metrics. A third frozen placement amendment routed Graph
  seeds 44/45 to node103 A6000 and MathIR seeds 43--45 to node302 A100.
  All five held job records passed identity/configuration audits and allocated
  immediately after release.
- The combined live audit now expects 42 terminal runs across E61-R1, E64, and
  E65R1. It reports zero integrity violations and preserves the frozen E64
  auditor while distinguishing its caught per-example `math_verify` timeout
  traceback from an uncaught training failure. A detached monitor refreshes
  all audits, scaling curves, and the five-row paper figure every 60 seconds.
- The first four prospective singleton escapes occurred without changing the
  protocol: two distinct eligible Countdown prompt banks in seed 43 and two in
  seed 44. Every admission independently had known support `1`, completed
  self-entropy warmup, entropy EMA below its own reference, inverse multiplier
  `>1`, active gate `1`, coefficient projection `0`, and zero conditioned or
  transform proposal rows sent to PPO. Each proposal group admitted exactly
  one alternate; the mechanism audit remained violation-free.
- The fail-closed E65R1 auditor now records each intervention conjunction in
  machine-readable form and rejects any admission missing a registered gate
  predicate. The focused controller/reporting/counterfactual suite passes 31
  tests.
- The live result artifact now separates three evidential roles: E58 Python
  telemetry diagnoses the dead-actuator problem, the one-seed E62R10 pilot
  establishes support-only actuator feasibility but is explicitly labeled
  overactive engineering evidence, and only the prospective three-seed E65R1
  terminal comparison can pass the repair claim. The paper figure and report
  continue to use only fixed, three-seed-complete performance checkpoints.
- Added a supplemental response-level E64 verifier-sensitivity audit without
  mutating the frozen E64 auditor or its primary scores. All six base-model
  step-0 runs produced identical greedy and fixed-seed sampled responses and
  identical 500-prompt reward vectors. Across 4,500 unique saved
  prompt/reference/response tuples, 39 caught timeout diagnostics and one
  caught grader traceback produced zero identical-response reward conflicts.
  The supplemental audit now runs continuously and must cover both raw traces
  at all seven registered MATH checkpoints before the combined terminal audit
  can pass.
- Added a continuously regenerated flat checkpoint table at
  `paper/results/e65_five_domain_confirmation_fixed_checkpoints_live.csv`.
  Every row is a registered, three-seed-complete domain/arm/metric/checkpoint
  combination and includes all three seed values, the mean/range, optimizer
  step, and terminal flag. This is the numerical companion to the paper
  figure rather than a digitized or peak-selected reconstruction.

## E66 same-plumbing actuator-off causal control — 2026-07-27

- The historical E61-R1 E58 arm and prospective E65R1 repair do not provide a
  clean actuator ablation: E65R1 pins replicated free-form request seeding and
  local single-GPU actor-weight synchronization, while E61-R1 used the older
  collector path. The historical E58 comparison therefore remains useful but
  cannot by itself identify the singleton actuator's effect.
- Before any three-seed-complete E65R1 post-training checkpoint landed, froze
  E66 as literal E58 with E65R1's exact source, execution surface, rollout
  plumbing, seed set, data, optimizer, checkpoint schedule, and 12-pass
  horizon. Only counterfactual proposals and the singleton entropy gate are
  disabled. Coefficients remain unbounded and unprojected.
- Submitted the prospective 12-run control: Graph jobs
  `30128394--30128396`, Countdown `30128397--30128399`, Python
  `30128400--30128402`, and MathIR `30128403--30128405`. Eight allocated
  immediately on the frozen A6000, RTX 3090, and A100 placements; four remain
  cleanly pending for those placements rather than silently changing
  accelerator family.
- The fail-closed causal gate now compares E65R1 directly with E66: Python
  terminal pass@8 must improve in both three-seed mean and worst seed; Graph,
  Countdown, and MathIR must remain within a `0.05` mean and `0.15` paired-seed
  loss margin; both audits must pass and E65R1 must contain at least one fully
  valid entropy-gated intervention. E66 versus historical E58 is separately
  reported as execution-plumbing sensitivity, not causal evidence.
- The combined campaign now expects 54 terminal runs. Its paper surface keeps
  ten fixed checkpoints per ModeBench arm and seven per MATH-500 arm, includes
  E66 as a fourth ModeBench arm, emits AUC only after every frozen checkpoint
  lands, and preserves all individual seed values in the live checkpoint CSV.

## E65R1 invalidation and corrected E67 treatment — 2026-07-27

- A paired pre-intervention telemetry check found that E65R1 was not literal
  E58 plus the singleton actuator as documented. All 12 E65R1 runtime logs
  reported `online_canonical_novelty_beta=0.0`, while every materialized E66
  control reported `0.50`. Countdown seed 43 already differed in novelty
  advantage and policy-gradient norm at optimizer update 1, before the
  64-update singleton gate warmup could complete.
- The cause was deterministic in E65R1's frozen execution branch: it
  explicitly overwrote the configured E58 novelty beta to zero. This is a
  non-outcome objective mismatch. A machine-readable invalidation audit
  confirms it across all 12 jobs. E65R1 is excluded from confirmatory gates
  and the 54-run denominator, but its artifacts remain preserved as disclosed
  engineering evidence.
- Stopped only the invalid E65R1 jobs `30127478,30127479,30127481` and
  `30127483--30127491`. E61-R1, E64, and E66 were untouched.
- Froze E67 before any three-seed-complete post-training E66 checkpoint. Its
  correction makes the repair variant inherit literal E58's configured bank
  alpha and novelty beta and adds a runtime expectation that aborts unless
  novelty beta remains exactly `0.50`. Relative to E66, only proposals and the
  singleton gate are enabled.
- Submitted and released the 12 corrected jobs: Graph
  `30128500--30128502`, Countdown `30128503--30128505`, Python
  `30128506--30128508`, and MathIR `30128509--30128511`. The held-job audit
  bound source, execution, protocol, launcher, manifests, seeds, controllers,
  `novelty_beta=0.50`, and the independent runtime expectation.
- A zero-step placement amendment moved only nine unmaterialized pending jobs
  to broader same-family A6000/RTX 3090 capacity under MLTheory; the artifact
  hashes the amendment and mutation script and records that no scientific
  setting changed. Low-priority preemption subsequently requeued the initial
  A100 allocations with checkpoint recovery enabled.
- Added a paired pre-intervention equivalence audit over request seeds,
  rewards, novelty advantages, replay state, entropy controllers, and
  optimization telemetry. The terminal campaign cannot pass unless all 12
  E66/E67 pairs clear that audit before the first E67 intervention.
- The live five-row figure, seed-level checkpoint CSV, combined audit, and
  frozen causal gate now use E67 rather than E65R1. The valid denominator
  remains 54 runs: E61-R1 24, E64 6, E66 12, and E67 12.

## E67 pre-optimizer invalidation and E68 separated support — 2026-07-27

- E67 exposed a second, independent contract problem before optimization.
  Literal E58 novelty beta `0.50` and proposal admission shared the same
  canonical count table. The source validator correctly rejected this because
  off-policy proposal support could otherwise alter a subsequent neutral
  rollout's on-policy novelty advantage. All 12 E67 logs reproduce the common
  validator, and none wrote `train_metrics.jsonl`.
- Canceled only E67 jobs `30128500--30128511`; all logs and frozen identities
  remain preserved. The machine-readable E67 invalidation audit is
  `confirmed`. E67 is excluded from performance evidence and from the valid
  denominator.
- Implemented a default-off separation contract. Proposal outcomes can expand
  replay-exemplar support but cannot enter the on-policy count table used by
  canonical entropy or novelty advantages. A proposed outcome remains novel
  to E58 until the neutral policy produces it; it then receives the ordinary
  E58 novelty bonus and replaces the proposal exemplar with its neutral
  exemplar. Proposal admissions emit an objective-outcome delta that must be
  exactly zero.
- Added checkpoint schema v3 for the separated proposal-only support and
  fail-closed resume validation. Unit tests prove replay support can grow from
  one to two while objective support remains one, exact E58 novelty is
  retained on later neutral discovery, proposal-only state resumes exactly,
  and capacity failures are atomic. The relevant bank, argument, mechanism,
  E58, and reporting suites pass 154 tests.
- Froze E68 before submission with the same data, seeds, optimizer,
  request-seeding, evaluation, ten ModeBench checkpoints, and 12-pass horizon
  as E66. Relative to E66 it enables proposals, the singleton entropy gate,
  and the explicit proposal/objective separation contract. Runtime beta
  remains `0.50`; proposal rows sent to PPO remain zero.
- Scheduler-only probes found startable paired capacity on A6000
  node103/node104, RTX 3090 node022 within the registered five-node pool, and
  A100 node302. The probe artifact is hash-bound in E68's identity.
- Submitted and released Graph `30130469--30130471`, Countdown
  `30130472--30130474`, Python `30130475--30130477`, and MathIR
  `30130478--30130480`. All three MathIR jobs allocated immediately and
  crossed argument validation. Their first optimizer records report beta
  `0.50`, separated support `1`, objective-outcome delta `0`, proposal PPO
  rows `0`, and no integrity violation.
- The live figure, checkpoint table, causal gate, and 54-run audit now use E68
  and require both E65R1 and E67 invalidation audits to remain confirmed.
  E66/E68 pre-intervention equivalence is checked over all 12 paired seeds
  before any actuator admission.
- A post-release scheduler-only probe suggested earlier `lowprio` starts for
  the nine unmaterialized Graph/Countdown/Python jobs. A hash-bound zero-step
  amendment moved them without changing any scientific setting. Actual batch
  estimates contradicted the interactive probe and became later, so a second
  hash-bound zero-step amendment restored `pvl-lowprio`. Both records are
  required by the E68 auditor; the three running MathIR jobs were untouched.

## E68 first intervention and durable checkpoint — 2026-07-27

- All three MathIR E68 seeds matched their E66 controls before actuation. The
  fail-closed equivalence audit compared 318, 305, and 337 updates before the
  first seed-specific interventions and found zero mismatches.
- Each seed independently crossed the 64 usable-observation entropy warmup and
  later activated only below its own reference. The first interventions
  occurred at steps 319, 306, and 338 for seeds 43, 44, and 45.
  Every audited proposal group has warmup complete, below-reference entropy,
  multiplier greater than one, singleton support, at most one admission,
  zero proposal rows sent to PPO, zero projection, and objective-outcome delta
  exactly zero.
- At the registered pass-1 checkpoint, E68 MathIR mean pass@8 is `0.339`
  versus E66's `0.330` (`+0.0085`); mean@8 is `+0.0026`, distinct@8 is
  `+0.0104`, and greedy accuracy is tied. This is an interim fixed-checkpoint
  result and does not satisfy the frozen pass-12 gate.
- Added a cached machine audit of each latest durable E68 checkpoint. All
  three step-384 states use separated-support schema v3. Their 5, 9, and 7
  proposal-only outcomes are present in replay exemplars, with zero overlap
  in the on-policy objective count table. The checkpoint audit is now required
  by the combined terminal gate.
- After preemption, the nine pending non-Math E66 controls had poor or absent
  single-node `lowprio` start estimates. A pre-recorded, hash-bound
  infrastructure amendment moved them to `pvl-lowprio` and broadened only
  within their original accelerator families (A6000 for Graph, RTX 3090 for
  Countdown/Python). Run IDs, frozen settings, metrics, and checkpoints are
  retained; the E66 audit fails if any trace regresses below its recorded
  pre-amendment step. Running MathIR controls were untouched.
- At the registered MathIR pass-2 checkpoint, the E68-versus-E66 mean deltas
  are positive on every reported metric: pass@8 `+0.0143`, mean@8 `+0.0103`,
  distinct@8 `+0.0150`, and greedy `+0.0156`. Pass@8 improves for seeds 43
  and 44 and is nearly tied (`-0.0039`) for seed 45. The pass-1 advantage has
  therefore persisted at a second prospectively fixed checkpoint.
- The step-768 checkpoint audit found 40, 62, and 39 current proposal-only
  outcomes for seeds 43, 44, and 45. All 141 are present in replay exemplars,
  none overlap the on-policy objective count table, and all three checkpoints
  retain schema v3 with the unprojected entropy controller. Five additional
  proposal discoveries have graduated: the neutral policy later produced
  those modes, at which point they entered on-policy counts through ordinary
  E58 novelty rather than through proposal admission.
- The first non-initial held-out MATH-500 checkpoint landed across all six
  E64 runs. Relative to GRPO at pass 2, E58 changes three-seed greedy accuracy
  by `+0.0087`, mean@8 by `-0.0063`, and pass@8 by `-0.0133`. This is mixed
  interim evidence but remains inside the frozen terminal realism margins:
  mean@8 loss no worse than `0.02` and at least one of greedy/mean@8 positive.
  The raw-trace sensitivity audit has zero identical-response reward conflicts.
- Twenty-one of 24 E61-R1 jobs became terminal or running after scheduler
  recovery. The last three pending checkpoint-resume jobs were transparently
  moved to `pvl-lowprio` under a hash-bound same-family amendment: Graph stays
  on A6000 and Countdown stays on RTX 3090. Their recorded pre-amendment steps
  are 1930, 1953, and 2450; the E61-R1 audit fails on any trace regression.

## E68 paired prompt uncertainty — 2026-07-27

- The retained E66/E68 evaluation sidecars contain the same prompt identities,
  ordering, references, fixed draw indices, and fixed evaluation seeds for all
  three complete MathIR pairs. Added a post-specified descriptive uncertainty
  surface that resamples training seeds and prompts as crossed paired units.
  The four K=8 draws are averaged before resampling and never counted as
  independent training replicates.
- The initial local v1 analysis centered greedy intervals on the sidecar's
  second temperature-zero evaluation. That repeated call differed slightly
  from the earlier primary greedy result used by the paper figure. Before
  integrating the analysis into reporting, froze a source-aligned v2:
  `greedy` uses primary `eval_results`, while sampled metrics continue to use
  the four fixed sidecar draws. Both versions and their hash-bound identities
  remain archived.
- At MathIR pass 2, the source-aligned E68-minus-E66 estimates and descriptive
  crossed-bootstrap 95% intervals are: greedy `+0.0156`
  `[-0.0313, +0.0625]`, mean@8 `+0.0103`
  `[-0.0029, +0.0303]`, pass@8 `+0.0143`
  `[-0.0241, +0.0547]`, and distinct@8 `+0.0150`
  `[-0.0326, +0.0664]`. All intervals are explicitly secondary and cannot
  alter the frozen terminal/AUC gates.
- The repeated greedy sensitivity changed 13 of 768 prompt scores across the
  six arm-seed evaluations at pass 2; per-run mean shifts were between
  `-0.0156` and `+0.0156`. The live report now exposes this rather than
  treating a second temperature-zero call as bitwise reproducible.
- The analysis tests pass, the live monitor refreshes it before rendering, and
  the report labels the central limitation: prompt resampling does not turn
  three independently trained seeds into more than three independent runs.

## Paired Graph A6000 drain recovery — 2026-07-27

- Slurm marked both registered prospective Graph nodes as draining after node
  health checks reported overheated A6000s: 9 on node103 and 7 on node104.
  E68 Graph had no start estimate; all three partial E66 Graph controls were
  also pending on that pool.
- A scheduler-only probe found a schedulable same-family A6000 placement under
  non-MLTheory account `allcs`, partition `lowprio`, on
  `node205,node206,node207`. No GPU was allocated by the probe.
- Froze and applied one paired infrastructure amendment to E66 jobs
  `30128394--30128396` and E68 jobs `30130469--30130471`. All six stayed on
  one A6000 each. Job IDs, run stamps, checkpoints, metrics, source,
  objective, seeds, and scientific settings are unchanged.
- The amendment records E66's pre-move steps `492/356/383` and E68's
  zero-update state. Both cohort auditors now verify the document and script
  hashes, exact affected job set, paired move, accelerator family, and
  no-regression checkpoints. E66, E68, and the 54-run combined audit remain
  `in_progress` with zero violations after the move.

## E61-R1 second same-family resume recovery — 2026-07-27

- Eight additional E61-R1 Countdown/Python jobs were preempted after the first
  placement amendment froze. All were pending, materialized checkpoint
  resumes and had reached steps `2011`, `2344`, `2407`, `2824`, `3165`,
  `3203`, `2418`, and `2498`.
- Their original RTX 3090 `allcs/lowprio` estimates extended from July 30
  through August 2 or were unknown. A scheduler-only probe found an earlier
  same-family `mltheory/pvl-lowprio` backfill window; E68 retained higher
  scheduler priority.
- Froze and applied a second placement-only amendment to jobs `30126339`,
  `30126340`, `30126341`, `30126343`--`30126347`. Node eligibility was
  broadened to `node020,node021,node022,node023,node024,node026`; every job
  remains on one RTX 3090 with its original job ID, run stamp, checkpoint,
  frozen source, arm, seed, optimizer state, and scientific settings.
- The E61-R1 auditor verifies both amendment document/script hashes, exact
  affected sets, same-family mutation, and per-job no-regression steps. The
  E61-R1 and combined audits remain `in_progress` with zero violations.

## Paired Graph A6000 contamination recovery — 2026-07-27

- E68 Graph jobs `30130469` and `30130470` were assigned node206 before any
  optimizer step. Both failed while vLLM tried to load the 0.5B model.
- Inside job `30130469`'s allocation, Slurm exposed physical GPU 6 with
  `48,323 MiB` already used. Five listed compute processes belonged to users
  `mi9937` and `rj5498`, not the E68 owner. This establishes
  cross-allocation GPU contamination rather than model memory demand.
- Requeue-held both zero-step attempts and held all six paired E66/E68 Graph
  jobs. Froze and applied a same-A6000 recovery to
  `node103,node104,node805` under `mltheory/pvl-lowprio`; job IDs,
  checkpoints, sources, arms, seeds, objectives, evaluation cadence, and
  scientific settings are unchanged.
- The amendment records E66's retained steps `492/356/383`, E68's zero-step
  state, and exact pre-amendment stdout/stderr byte lengths for both failed
  jobs. Auditors exempt uncaught signatures only before those offsets; any
  later OOM remains a hard failure.
- E66, E68, equivalence, checkpoint-separation, and combined audits returned
  to `in_progress` with zero violations after the move.

## Five-domain cadence and fixed-checkpoint coverage hardening — 2026-07-27

- Added a fail-closed evaluation-cadence audit over durable
  `eval_results/*.json` artifacts for all 54 registered runs. Every
  materialized run with optimizer progress must have an evaluation gap no
  larger than one prompt epoch and an exact terminal evaluation. The first
  integrated audit checked 44 progressed runs plus two pre-optimizer runs and
  found zero violations. Held-out MATH-500 is landing evaluations every 96
  updates, four times per 384-update epoch.
- Added an independent seed-level fixed-checkpoint coverage audit. It requires
  all registered metrics and seeds 43/44/45 at each paper point, enforces no
  more than ten checkpoints per domain, and cannot pass until every E61,
  E64, E66, and E68 cohort is terminal. The initial surface contains 77/174
  complete arm/domain checkpoint cells and 912/2046 required seed-metric
  values with zero integrity violations.
- Both audits are required by the combined terminal gate, are rendered in the
  live report, and run in durable monitor job `30131521`. Reporting and audit
  tests pass.
- E61 jobs `30126338` and `30126350` had already written their terminal
  evaluations, exported model directories, and clean
  `TRAINING_COMPLETE.json` markers but remained as stale pending watchdog
  resumptions. Those two scheduler entries were canceled without deleting or
  changing any result artifact.

## Legitimate-result readiness gate — 2026-07-27

- Added `audit_e65_legitimate_result_readiness.py` as the final fail-closed
  claim gate. It requires the exact five-domain surface, nonempty primary and
  all-epoch diagnostic figures, at most ten registered paper checkpoints per
  domain, passing per-epoch cadence, complete three-seed fixed-checkpoint
  coverage, a passing 54-run integrity audit, and all four frozen scientific
  gates.
- The primary paper figure now uses ten registered ModeBench anchors and seven
  registered MATH-500 anchors. A separate diagnostic figure grows at every
  completed integer MATH-500 epoch, so monitoring remains granular without
  silently changing the paper estimand.
- The first readiness audit correctly reports `in_progress`, with 7/54
  terminal runs, 78/174 complete fixed-checkpoint cells, seven pending
  requirements, zero failed requirements, and zero integrity violations.
  Evaluation-at-least-once-per-epoch already passes.
- The combined reporting tests now contain 17 checks and pass. Readiness is
  not allowed to report success merely because the figures exist or an
  interim effect looks promising.
