# ConstructiveCode executable slate v5 — prospective source-count repair

Frozen before v5 materialization or checker execution on 2026-07-30.

## Prior outcome and admissible repair

The immutable v4 source-gate job `30201021` failed during materialization, before any candidate program or checker replay was executed. Its exact error was `only 52 held-out Python-3 programs; 64 required`. No v4 slate, replay ledger, checker-equivalence receipt, or gate audit exists.

V5 changes only the per-task, per-label source requirement from exactly 64 to exactly 48. The threshold 48 is below the observed limiting source count of 52 while retaining a larger executable gate than the earlier 1,600-replay v2 design. No task, split, source revision, program ordering rule, checker, suite policy, label, acceptance rule, model, prompt, or training decision changes.

## Frozen slate

- The same 12 v3/v4 tasks and the same fixed 4/4/4 train/development/evaluation assignment are used.
- The same pinned full CodeContests Plus and CodeContests-O source revisions are used.
- Only explicitly labeled Python-3 submissions are eligible.
- V1 overlap is allowed and reported; selection is unique program SHA-256 in ascending order.
- Each task must supply exactly 48 correct and exactly 48 incorrect programs. Failure is fail-closed.
- The evaluation split remains unloaded, and no language-model sampling occurs in this gate.

## Frozen executable gate

Every selected program is executed against both frozen official checker suites, producing exactly 2,304 submission-suite execution records: 12 tasks × 96 programs × 2 suites.

The task-level suite policy remains: use CodeContests-O when admissible; otherwise use Plus-5x. Admission requires all 12 tasks to meet the existing source, execution, checker-equivalence, and known-label replay criteria. A pass permits only split construction and development-only Qwen2.5-Coder-0.5B viability sampling. It does not authorize a paper training cell.

## Immutability

The launcher must identity-bind the v4 identity and exact v4 error log, snapshot all executable sources, and release the held job only after recording a v5 identity. Any v5 failure is retained as an outcome; there is no post-execution task, program, or checker substitution.
