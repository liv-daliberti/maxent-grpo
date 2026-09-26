# E50E — exact OAT 384-train/MATH-500 E46 extension

**Status: PREREGISTERED BEFORE E50 MATCHED-TOY RESULTS — 2026-07-26**

**SERVICE-CONTINUITY AMENDMENT PREREGISTERED BEFORE E50F/E50H/E50G OR E50D
TERMINAL RESULTS — 2026-07-26.**  The already calibrated node302 Qwen72
allocation has a fixed 24-hour limit that cannot be extended by the user and
therefore cannot guarantee coverage of full materialization plus 1,152
updates.  Only after E50D passes, start a fresh 48-hour four-way
Qwen2.5-72B-AWQ service on node302 port 8771 using the identical checkpoint revision,
model-file hashes, 32,768-token context, eight-sequence limit, eager mode,
and guidance structured-output backend.

Before that service may submit E50E materialization, rerun both immutable
E49T finite-menu controls against its exact endpoint record.  Require at
least 75% positive route acceptance, at least ten dual-route groups, and
zero duplicate false-new, wrong-route, answer-only, open-set, declared-route
mismatch, and judge-format errors.  A deterministic continuity certificate
binds both passing results, frozen cohorts, frozen canonicalizer, scorers,
checkpoint configuration, endpoint record, and Slurm job.  Materialization,
both full training arms, and both full terminal route probes must use that
same certified endpoint.  Any identity drift or service/calibration failure
prevents downstream submission.

## Advancement and source selection

E50E materializes only after the complete E50D matched toy passes every task,
route-support, canonical-bank, and normalized-Haarnoja advancement gate.
E50D is the sole eligible source; quarantined earlier E50 route-discovery
variants cannot become fallbacks.  Failed toy menus cannot be repaired,
relabeled, or substituted.

## Exact data contract

- Training is exactly all 384 rows, in frozen order, from
  `var/data/math12k_384_math500/train`.
- Evaluation is exactly all 500 MATH-500 rows, in frozen order, from
  `var/data/math12k_384_math500/eval`.
- The passing toy's 50 training and 50 held-out evaluation rows are overlaid
  only on exact source-ID, original-problem, and answer matches.
- The ten selected dual training menus and the one accessible dual
  evaluation menu remain byte-identical at the canonical-menu level.
- Each of the other 784 source rows receives one answer-blind singleton
  action combo.  Two independent temperature-zero Qwen2.5-72B audits must
  literally execute it, independently derive the auditor-only answer, find
  every action sound and sufficient, and find no hidden decisive step or
  final-answer leakage.  Failures retry only with fresh answer-blind
  generation; answer-bearing feedback is forbidden.

The terminal artifact must have 384 train rows, 500 evaluation rows, 884
parseable nonzero-support menus, exactly 100 toy overlays, exactly 11 dual
menus (10 train and 1 eval), and 873 singleton menus.  Source order and
answers must be unchanged, and every formatted prompt must fit 2,048 input
tokens.

## Matched three-epoch comparison

Run exactly two Qwen2.5-0.5B-Instruct arms, seed 45, 16 samples per update,
learning rate 2e-7, one A100 per arm, and exactly three prompt epochs: 1,152
updates.  Evaluate on all 500 MATH-500 rows at steps 0, 384, 768, and 1,152.

- Control: ordinary Dr.GRPO behind the same exact-answer plus exact-route
  validator.
- Treatment: the exact passing-toy E46 normalized canonical-bank Haarnoja
  implementation, with novelty beta 0.50, alpha initialized at 0.10 and
  projected to [0.10, 0.50], normalized target 0.80, log-alpha Adam learning
  rate 0.003, EMA 0.90, and policy-entropy adaptation disabled.

Both arms use the passing toy's source-identity checked scalar-`solve(Eq)`
grader overlay while importing the byte-frozen E49T canonicalizer and E46
controller.  No other source or objective change is permitted between toy
and full training.  The overlay tree SHA-256 remains
`ba032e9ca300c25385be9650582556f6c8f833ce1f4f5a7197c4259ce5e44a1e`.

At shared initialization and both terminal checkpoints, draw 64 unforced
samples on the ten no-gradient dual training probes.  Report MATH-500 task
quality, answer-plus-route acceptance, bank support, normalized entropy,
signed alpha-control telemetry, and route preservation beside the frozen
successful graph-coloring and Countdown E46 mechanism ranges.  A successful
treatment must retain natural support on at least eight probes, where each
route has at least two accepted responses and at least eight responses are
accepted in total.  No claim of success is allowed from task accuracy alone.
