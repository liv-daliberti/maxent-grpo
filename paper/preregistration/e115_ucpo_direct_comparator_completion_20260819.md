# E115 preregistration: UCPO direct-comparator completion

Date frozen: 2026-08-19, before submission of any E115 smoke or scientific
cell.

## Question and estimand

Does UCPO preserve verified-answer breadth relative to the matched Dr.GRPO
control across every registered model/domain block? The estimand is the paired
terminal-pass difference, UCPO minus Dr.GRPO, within model, domain, and seed.
E115 extends the E97 treatment without tuning it on the new outcomes.

## Frozen cells

- Qwen2.5-0.5B: Countdown and MathIR, seeds 43--47 (10 cells), paired to E78.
- Qwen2.5-3B: all five ModeBench domains, seeds 70--74 (25 cells), paired to
  E80-R1.
- Together with terminal E97 and E99, these 35 cells complete five seeds for
  all five domains at all three scales.
- Each cell inherits its paired control's model revision, dataset, native
  prompt surface, optimizer, rollout group of 16, decoding, 384 prompts x 8
  passes, and 192-step checkpoint schedule.
- The only active treatment change is the already-frozen E97 UCPO advantage
  redistribution with `tau = 0.2`. RLEP is disabled. Existing passive replay
  remains compute-only exactly as in the matched Dr.GRPO control.

## Gates and reporting

One fixed 32-update learner smoke per model family must complete before that
family's scientific cells become eligible. All jobs are submitted held,
identity-audited, recorded atomically, and only then released. Report all paired
seed effects and failures. No model-, domain-, or seed-specific tuning is
permitted.

