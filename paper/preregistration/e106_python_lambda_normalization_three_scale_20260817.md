# E106: Python LaTeX-lambda normalization repair at three scales

Frozen: 2026-08-17, before any E104 post-update evaluation result was opened.
PointMaze and every other interactive environment are excluded.

## Why this is a repair rather than a new method

The outcome-blind E104 mechanism audit reached a terminal failure for the
Qwen2.5-3B Python-factor cell: it completed 64/64 optimizer steps with the v6
group-centered estimator active, but the verified bank remained empty and no
replay update was applied.  Its training telemetry reported zero admitted
canonical rows.  The matched 0.5B Python cell used the same scheduler and did
populate replay.

Inspection of a previously analyzed E80R1 step-0 response file localized the
interface defect.  Qwen2.5-3B writes the requested Python lambda in the native
boxed mathematical form `\boxed{\lambda n: ...}`.  The ModeBench extractor
unboxed this to `\lambda n: ...`, while the restricted Python parser requires
the literal keyword `lambda`.  Replaying the repaired extractor over those 128
already-analyzed responses converts all 128 to the intended Python surface and
recovers three programs that pass the unchanged executable validator.  The
machine-readable diagnosis is
`var/artifacts/e106_python_lambda_normalization_diagnosis.json`.

The sole scientific/runtime change relative to E104 is a formatting alias in
`src/oat_drgrpo/math_grader.py`: an exact leading `\lambda n:` or
`\lambda\,n:` becomes `lambda n:`.  The exact one-argument signature,
restricted AST language, isolated execution, divisor checks, endpoint key, v6
semantic estimator, replay objective, and optimizer are unchanged.  Unsafe
calls and alternate argument names remain rejected by regression tests.

## Frozen replacement design

E106 reruns only `python_factors`, once at each E104 scale:

| scale | seed | inherited E104 model/config |
|---|---:|---|
| Qwen2.5-0.5B-Instruct | 43 | exact E104 Python cell |
| Falcon3-1B-Instruct | 55 | exact E104 Python cell |
| Qwen2.5-3B-Instruct | 70 | exact E104 Python cell |

Each cell uses 64 training prompts, one prompt pass, rollout group size 16,
temperature 1, the E104 scale-specific learning rate and memory recipe,
semantic coefficient 0.1, replay coefficient 0.1, one scheduled global replay
group per update, no counterfactual proposals, and the same Python train/eval
data and prompt template as its E104 counterpart.  Checkpoints/evaluations stay
at steps 0, 32, and 64.  E106 changes only the immutable source snapshot noted
above and uses new output directories; no E104 artifact is overwritten.

For the eventual full cohort, the three E106 Python cells supersede the three
E104 Python cells.  The other twelve static-domain E104 cells remain the exact
mechanism evidence.  The original pending E104 Falcon-Python job may be
cancelled before allocation once the E106 ledger is durably written, because
its parser snapshot is already known to be superseded.

## Outcome-blind release gate

No post-update accuracy, score, pass@k, distinct@k, or coverage result may be
read to decide this gate.  Each E106 cell must:

1. finish with scheduler state `COMPLETED`, exit code zero, and reach step 64;
2. run the versioned `python-factor-response-v2-latex-lambda` snapshot;
3. keep v6 group-centered semantic advantages active, legacy v5 off, the RMS
   controller off, the effective within-group mean within `1e-8`, and every
   semantic advantage within the fixed coefficient bound;
4. admit at least one verified Python row and populate the online bank;
5. apply at least one verified-replay actuator group with a nonzero score
   gradient; and
6. contain no traceback, assertion, OOM, or non-finite marker.

The combined gate passes only after those three E106 cells and all twelve
non-Python E104 cells pass their mechanism checks.  Only that combined pass may
unlock E105.  Any failed cell remains a failed repair; the criteria will not be
weakened post hoc.

## Blinding and scope

The prior E80R1 file used for diagnosis was already part of the analyzed paper
record.  No E104 post-update result was inspected.  E106 itself remains blinded
until its mechanism gate is final.  This repair adds no PointMaze data, jobs,
figures, or claims.
