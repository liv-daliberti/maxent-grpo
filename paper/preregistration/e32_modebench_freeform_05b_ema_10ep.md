# E32: clean matched 0.5B free-form rerun with EMA Haarnoja control

**Status: FROZEN — prospective, before submission**

**Launch correction (2026-07-22, before E32-v3 submission):** the initial v2
execution surface inherited the repository-wide five-pass safety ceiling even
though the held-job record requested ten passes. That mismatch was visible in
the runtime log before any run reached one complete pass on Countdown (graph
runs had only early, nonterminal data). The v2 allocation and its trajectories
are excluded. The first v3 held submission was rejected before allocation when
its audit showed that the shared submitter had not serialized the new ceiling
into the Slurm environment. The v4 held submission was also rejected before
allocation because the site submit filter rewrote `all` to `mltheory`; its
placement audit failed closed. E32-v5 explicitly freezes and forwards
`OAT_ZERO_MAX_PROMPT_EPOCHS=10`, normalizes and re-audits the effective held-job
partition, uses a new run-stamp and identity, and runs on non-preempting `all`.

## Question

Does conditional-token MaxEnt improve free-form exploration over a matched
Dr.GRPO control on Countdown and graph coloring when resume synchronization,
evaluation replication, and dual-controller noise handling are fixed in the
training code itself?

## Cohort

- Model: the local immutable Qwen2.5-0.5B-Instruct snapshot ending in
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Tasks: `exact_countdown_easy3_probe` (384 training prompts) and
  `exact_answer_mode_probe` (192 training prompts).
- Arms: matched free-form Dr.GRPO and conditional-token MaxEnt with a
  base-preserving Haarnoja log-alpha Adam controller.
- Training seeds: 43, 44, and 45.
- Budget: 10 complete passes over each prompt pool, G=16 completions per
  prompt, one prompt group and one optimizer update per step.
- Shared optimization: learning rate 2e-7, one PPO epoch, beta 0, max norm 1,
  temperature 1, top-p 1, and 192 generated tokens.

## MaxEnt treatment fixed in advance

- Objective: mean conditional token entropy.
- Entropy targets: 1.622718550885717 for graph coloring and
  1.347109432487438 for Countdown (the prior 125% calibration).
- Alpha starts and is bounded below at 0.000075, is bounded above at 0.00060,
  and uses Adam with learning rate 0.010 after 64 warmup observations.
- The controller observes an entropy EMA with decay 0.7:
  `ema_t = 0.7 * ema_(t-1) + 0.3 * entropy_t`. Both raw and EMA entropy are
  logged. No other controller knob differs from the prior aggressive target.

## Evaluation and retained evidence

- Evaluate at initialization, every quarter prompt-pool pass, and the terminal
  boundary.
- Retain deterministic greedy pass@1.
- At every evaluation boundary, run four reproducible K=8 sampled evaluations
  with seeds 1001, 1002, 1003, and 1004 at temperature 1.
- Log and plot mean@8, pass@8, coverage@8, and distinct@8. For every metric,
  retain the four raw draw values plus mean, sample SD, SE, minimum, and maximum.
- Retain prompt, reference answers, raw response text, parsed answer, reward,
  training step, evaluation seed, and draw index in the evaluation trace JSONL.
- Figures show unsmoothed seed trajectories and raw evaluation draws. A heavy
  mean is drawn only where all three training seeds have reached the boundary.

## Resume and storage contract

- A restored learner checkpoint must synchronize its weights to every actor
  before evaluation or rollout.
- At duplicate resume boundaries, the predecessor evaluation is authoritative;
  the pre-synchronization resumed evaluation is rejected during curve parsing.
- Write a rolling optimizer-resumable checkpoint once per prompt-pool pass,
  retain the newest two, and do not prune them automatically on success.
- Export only the terminal inference checkpoint. Watchdog requeues resume the
  same immutable source, execution surface, protocol identity, and run stamp.

## Analysis

The primary plots contain pass@1, mean@8, pass@8, coverage@8, and distinct@8
against training passes. There is no smoothing, no replacement of raw draws,
and no cross-cohort control reuse. The E32 cohort supersedes E22/E27 only for
the newest 0.5B free-form comparison; historical artifacts retain their
original labels and provenance.
