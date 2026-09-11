# E33: clean matched 3B free-form rerun with EMA Haarnoja control

**Status: FROZEN — prospective, before submission**

**Placement amendment (frozen before v2 submission, 2026-07-22):** E33-v1's
12 A5000 jobs all failed during actor initialization, before step 0 or any
evaluation, because vLLM could not allocate a KV-cache block at ratio 0.25 on
24 GiB GPUs. They contain no scientific trajectories. E33-v2 retains every
scientific setting below under fresh `*_v3_a100` identities and uses the
previously validated node302 A100-80GB placement.

The v2 A100 submission was cancelled while still held with zero runtime after
Slurm represented its typed GPU request as `TresPerNode=gres/gpu:a100:1`
rather than the launcher's expected `gres/gpu:a100=1`; v3 corrects only that
attestation spelling.

## Question

Does conditional-token MaxEnt improve free-form exploration over a matched
Dr.GRPO control at 3B scale when the complete E32 training, evaluation,
resume, and visualization contract is applied prospectively?

## Cohort

- Model: the local immutable Qwen2.5-3B-Instruct snapshot ending in
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
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
  logged. No treatment knob is selected from E33 outcomes.

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
- Figures show unsmoothed seed trajectories. Heavy means appear only where all
  three training seeds have reached the boundary, with light 95% t-confidence
  intervals over the four fixed draw-level, all-seed means.

## Resume, storage, and placement contract

- A restored learner checkpoint must synchronize its weights to every actor
  before evaluation or rollout.
- At duplicate resume boundaries, the predecessor evaluation is authoritative;
  the pre-synchronization resumed evaluation is rejected during curve parsing.
- Write one rolling optimizer-resumable checkpoint per prompt-pool pass, retain
  the newest two, and do not prune them automatically on success.
- Export only the terminal inference checkpoint. Watchdog requeues resume the
  same immutable source, execution surface, protocol identity, and run stamp.
- Each run requests one A100-80GB, 16 CPUs, and 96 GiB RAM from node302 in
  non-preempting `mltheory`, using ZeRO-2, optimizer and activation offload,
  and vLLM ratio 0.25. Five jobs can run concurrently under node302's host-RAM
  limit; the remaining jobs stage without changing their scientific budget.
  All twelve jobs are submitted held and released only after a full invariant
  and aggregate-capacity audit.

## Analysis

The primary plots contain pass@1, mean@8, pass@8, coverage@8, and distinct@8
against training passes. There is no smoothing, no replacement of raw draws,
and no reuse of the contaminated E25/E28 trajectories in the E33 matched
comparison. Historical artifacts retain their original labels and provenance.
