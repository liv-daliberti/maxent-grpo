# E50B — observed-route hard-MATH matched toy

**Status: PREREGISTERED BEFORE E49AB/E50A/E49AC TERMINAL ROUTE RESULTS — 2026-07-26**

## Route-source gate

E50B materializes only after a conservative observed-route calibration
passes with ten bidirectionally executable Qwen2.5-0.5B problems.  The fixed
source preference is E49AA's shortest-sixteen calibrated pairwise-veto
result, followed by E49AB's preregistered all-observed persistent
pairwise-veto result, followed by E50A's exact-assignment two-partition
consensus result, followed by E49AC's independently confirmed singleton
result.  E50B uses the first passing source in that order.  E49AC was added
to this frozen preference before E49AB and E50A had terminal results.  If
none passes, E50B does not train.  Failed routes may not be relabeled,
repaired, or substituted after forced-generation outcomes are observed.

## Data

Training has 50 exact OAT hard-MATH rows:

- ten selected, double-audited, bidirectionally executable dual-route rows;
- the E49T singleton training rows that do not duplicate a selected row; and
- when deduplication removes an E49T singleton, the earliest nonoverlapping
  E49T dual row, pruned to the route with the largest frozen E49U forced
  success count (ties choose the lower strategy ID), until 40 singleton
  rows remain.

All 50 training row IDs must be unique.  A separate no-gradient route-probe
dataset contains exactly the ten selected dual-route rows.  Strategy IDs and
listing order are explicitly described as non-preferential, but the model
still chooses its own route on every unforced rollout.

Task evaluation retains the same 50 held-out E49T MATH-500 rows.  The nine
E49U-inaccessible dual menus are pruned by the frozen E49U rule and the one
bidirectionally executable menu remains dual.  This evaluation measures task
quality; it is not used as the main route-preservation claim.

Before training, 64 independent unforced base-model samples are drawn for
each of the ten route-probe prompts.  A response counts only when the exact
answer validator and frozen E49T route canonicalizer both accept it.  At
least eight of ten prompts must have at least two accepted responses from
each route and at least eight accepted responses in total; every retained
prompt must have at least one accepted response.  Otherwise E50B does not
train.

## Matched training

Run two exactly matched Qwen2.5-0.5B-Instruct arms on one A100 each, seed 45,
16 samples per update, learning rate 2e-7, and exactly three prompt epochs
(150 updates):

- ordinary Dr.GRPO control behind the same answer-plus-route gate;
- unchanged E46 normalized canonical-bank Haarnoja treatment with novelty
  beta 0.50, alpha initialized at 0.10 and projected to [0.10, 0.50],
  normalized target 0.80, log-alpha Adam learning rate 0.003, EMA 0.90, and
  policy-entropy adaptation disabled.

Initialization, data order, verifier, judge, rollout settings, checkpoint
cadence, and evaluation draws are otherwise identical.

The training runtime imports the byte-frozen E49T canonicalizer and E46
controller.  Before importing its frozen answer grader, a source-identity
checked startup overlay installs only the already-tested scalar
`solve(Eq)` compatibility repair used by the successor route calibrations.
The overlay must verify the frozen grader and canonicalizer SHA-256 values
and may not change any reward, bank, controller, or route decision.
Its frozen source-tree SHA-256 is
`ba032e9ca300c25385be9650582556f6c8f833ce1f4f5a7197c4259ce5e44a1e`.

## Advancement evidence

At base initialization and both terminal checkpoints, draw 64 unforced
samples on the ten no-gradient route probes and score exact answer plus exact
route.  The toy advances only when:

1. both arms finish all 150 updates with identical step-zero task metrics;
2. both arms have nonzero gated answer-plus-route training reward;
3. E46 records nonzero canonical-bank observations and signed Haarnoja
   optimizer steps, support at least two, and nonzero normalized entropy;
4. E46 loses no more than five task-accuracy points to Dr.GRPO in terminal
   greedy or sampled-mean evaluation;
5. E46 terminal mean normalized route coverage is no lower than its shared
   base initialization and no lower than Dr.GRPO terminal coverage; and
6. at least eight of ten E46 terminal route probes retain the same natural
   support threshold: two accepted responses per route and eight accepted
   responses total.

Only a passing E50B may seed the exact 384-train/MATH-500 three-epoch run.
