# E47R-CAL: reasoned proof and strategy-equivalence calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47Q is the first live semantic preflight to pass all four frozen checks:

- a valid anchor is admitted;
- two answer-matched invalid derivations are rejected;
- per-revolution and per-time parameterizations of one computation merge; and
- coefficient expansion and strategic point evaluation remain distinct.

E47R is the full 50-problem calibration of that exact executable. It uses two
answer-integrity classifications with a retained `brief_check`, followed by
two strategy partitions and the conservative component-incidence admission
rule. Its persistent schema is
`math_strategy_canonicalizer_reasoned_equivalence_v13`.

The data, 3,200 Qwen0.5B samples, final-answer validator outcomes, injected
controls, group-16 online schedule, Qwen72 checkpoint, seeds, temperature,
finite JSON bounds, and failure semantics are unchanged from E47L/M.

The coverage gate is the prospectively amended 50% threshold defined before
E47M: proof-invalid answer matches are intentionally rejected and cannot be
required to receive a strategy key. All precision gates remain unchanged.
If no naturally occurring human-different pair survives the proof screen,
the manually authored, answer-validated E47Q different-route regression
supplies the false-merge check.

E47R advances only if:

- its own rerun of all four semantic regressions passes;
- exact-duplicate false-new is zero;
- overall injected false-new is at most 5%;
- lexical-paraphrase false-new is at most 10%;
- at least 50% of answer-matched policy responses receive a key;
- both known invalid witnesses receive no key;
- blinded same-pair false-new is at most 5%;
- blinded different-pair false-merge is at most 20%, or the E47Q-style
  frozen different-route regression passes when no natural human-different
  pair exists; and
- every bounded problem completes without structural failure.

Passing licenses a hard-but-solvable toy only. The level-five calibration set
itself is not the toy because E47L showed it contains no trustworthy
multi-policy-strategy support after false boundaries are removed.
