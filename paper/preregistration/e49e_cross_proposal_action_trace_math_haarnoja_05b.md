# E49E — Superseded Prelaunch Cross-Proposal Draft

**Status: SUPERSEDED BEFORE ANY E49E JUDGE REQUEST OR TRAINING LAUNCH — 2026-07-24**

This draft made no requests and launched no policy job. The sole canonical
protocol is
`paper/preregistration/e49e_trace_executed_bank_math_haarnoja_05b.md`.

The historical text below is retained only to make the prelaunch design
history auditable.

## Why E49D cannot train

E49D correctly stopped before training, but manual inspection of its first
retained multi-route examples exposed a calibration failure in the actual menu
domain that E47W's injected-pair calibration did not cover:

- a purported CRT route determined only a residue class and then imported
  excluded actions `A1`–`A3` to select the exact answer;
- stars and bars for distinguishable boxes was incorrectly repaired by an
  auditor when the problem's boxes were indistinguishable;
- a sphere/cube route equated the sphere diameter to the cube space diagonal;
- a determinant with a fixed unit vector was treated as the full magnitude of
  a three-dimensional cross product; and
- Binet's irrational closed form was manipulated modulo 4 as though Euler's
  theorem directly applied.

Both E49D auditors sometimes called these routes sound. Therefore neither the
old 20% count nor a passing support-recall count is sufficient evidence.
No E49D policy job may launch.

E49E retains the current E46 normalized canonical-bank Haarnoja learning
method. It changes only the preprocessing and exact execution contract so the
canonical object is materially executed, as it is in Countdown and graph
coloring.

## Exact cohorts and matched schedule

- Toy: the exact frozen E49B/E49D 50 hard training rows and 50 hard MATH-500
  evaluation rows.
- Full: the exact OAT 384-row MATH training cohort and exact 500-row MATH-500
  evaluation cohort.
- Qwen2.5-0.5B-Instruct, seed 45, group size 16, learning rate `2e-7`, one PPO
  epoch, temperature 1, top-p 1, one A100 per arm, and exactly three prompt
  epochs.
- Both arms receive identical prompts, menus, deterministic parsing, two
  runtime 72B audits, task-reward gate, sampling, and compute.
- Only treatment receives the unchanged E46 normalized Haarnoja controller
  and first-discovery bonus.

## No new proposal generation

E49E uses the append-only E49D proposal evidence as a finite candidate bank.
It issues no new route-ideation or menu-generation request.

For each successful E49D proposal, recompute its answer-bound v4 maximal
certified subset. A singleton contributes its one retained route; an old
multi-route clique may contribute each route, because every candidate is
re-certified below. Exact duplicate operation/plan candidates collapse.

Before any model call, reject a candidate if its plan or any included action
explicitly references an action ID outside its own combo. Reindex all retained
action references deterministically. Choose at most three candidates and at
most twelve total actions by the frozen score:

1. maximum candidate count;
2. maximum number of distinct source proposals;
3. minimum total action count; and
4. earliest source-evidence order.

This is the prompt-local candidate bank, not yet eligible support.

## Action-by-action certification

Every candidate receives exactly two temperature-zero, answer-bound audits:

1. `literal_executor`, seed 492111, must literally execute every declared
   action in order and state the fact it produces.
2. `adversarial_proof_checker`, seed 492112, tries to falsify every identity,
   count, case, domain, uniqueness claim, dependency, and sufficiency step.

Each structured response must contain the exact ordered action IDs, a concrete
calculation and output fact for each, booleans certifying that only declared
actions were used and the route is self-contained, and the derived answer.
Local deterministic validation requires:

- exact action IDs in exact combo order;
- all free-text fields present and bounded;
- every action status `valid`;
- both closure booleans true;
- route status `sound`;
- the auditor's answer-match flag true; and
- an independent local `math_verify`-based comparison of the stated derived
  answer to the frozen reference.

Malformed completed output is a terminal failed audit and consumes the one
request. Only a request that receives no response after the transport retries
is incomplete and may resume. Durable records reuse every completed audit, so
restarting a job cannot resample a completed decision.

If at least two candidates pass both soundness audits, the whole candidate
menu and its action traces receive exactly two temperature-zero novelty
vetoes:

1. `equivalence_attack_a`, seed 492121;
2. `equivalence_attack_b`, seed 492122.

Equivalent is the default. An edge exists only when both auditors call the
pair distinct, say it is not reducible by routine algebra, certify both routes
self-contained, and name different nonempty decisive operations. E49E takes
the deterministic maximum clique in candidate order. A singleton is allowed;
zero sound routes blocks the stage.

## Exact policy execution contract

Every completion must begin at character zero with the exact menu-bound:

```text
<strategy_id>Sj</strategy_id>
<action_combo>Aa>Ab>...</action_combo>
```

It must then contain `<action_trace>` and exactly one nonempty
`<action_step id="Ai">...</action_step>` block for every declared action in
the exact combo order, followed by `</action_trace>` and a boxed final answer.
Missing, extra, duplicate, empty, unknown, or reordered blocks fail
deterministically before any judge call.

For answer-positive completions, two independently permuted frozen 72B runtime
audits must verify that each block actually executes its named action, the
facts connect in order, no omitted action or other strategy is imported, and
the blocks suffice for the answer. A header or action tag alone proves
nothing. Rejected task reward is set to zero in both arms before ordinary
Dr.GRPO centering.

The canonical outcome is the finite prompt-local action combo. Surface
paraphrases merge; an unexecuted declaration cannot enter the bank.

## Learning method

Matched execution-gated Dr.GRPO is the passive control. Treatment uses the
unchanged current E46 settings:

- initial/minimum alpha 0.10, maximum alpha 0.50;
- normalized target `H(q_x)/log |B_x^+| = 0.80`;
- log-alpha learning rate 0.003 and EMA decay 0.90;
- pseudocount 1 and surprisal clip 5; and
- first audited discovery bonus beta 0.50.

Singleton prompts do ordinary task learning and do not update the normalized
entropy dual. No token entropy, semantic Shannon, uncertainty controller,
DIAYN, XDR, curriculum, or other exploration method is active.

## Pre-training calibration and launch gate

Before toy training:

1. all 100 rows must have a recomputable trace-certified menu;
2. at least 20/100 overall and at least 10/50 evaluation rows must retain two
   or more routes;
3. every retained toy multi-route pair receives a blinded manual audit for
   route soundness, self-containment, and genuine distinction;
4. manual false-new count must be exactly zero; and
5. prompt length must be at most 2048 tokens for every rendered row.

Any failure blocks training and remains immutable.

## Toy advancement and full success

Evaluate raw greedy and fixed-seed pass@8 at epochs 0, 1, 2, and 3. Re-audit
terminal stored traces under the exact action-block contract. Full may launch
only if:

- both matched jobs complete exactly 150 steps and four eval points;
- epoch-0 raw metrics are identical;
- reward accounting, deterministic format gate, judge format, controller
  direction, alpha bounds, and monotone-bank invariants pass;
- treatment terminal raw and execution-gated pass@8 are each at least control;
- treatment raw greedy is within five percentage points of control;
- on multi-route-eligible prompts, treatment support-at-least-two is at least
  control plus 10 points; and
- treatment mean audited correct strategy count is at least control plus 0.20.

The full 384/500 stage runs exactly three epochs. It succeeds under the same
mechanism and quality invariants, with the eligible support-at-least-two
margin set to treatment at least control plus five points.

Implementation hashes and the immutable E49E source snapshot are recorded
after focused tests and before the first E49E judge request.
