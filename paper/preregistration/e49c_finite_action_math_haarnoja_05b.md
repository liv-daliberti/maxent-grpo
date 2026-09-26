# E49C — Finite-Action, Execution-Gated MATH Haarnoja (0.5B)

**Status: FAILED PREPROCESSING; NO TRAINING LAUNCHED; SUPERSEDED BY E49D MAXIMAL-CERTIFIED SUPPORT — 2026-07-24 08:54 EDT**

## Motivation

E49B showed that the calibrated 72B canonicalizer is conservative and can
separate genuine proof routes, but an unscaffolded Qwen2.5-0.5B policy almost
never discovers a second route for the same hard MATH prompt. Its verified
support was 50 outcomes over 48 tracked prompts at step 146. The treatment
therefore could not exercise the normalized-support controller and later
terminated on a malformed judge response.

E49C tests the finite-action version proposed before E49B: canonicalize the
available proof routes *before* training, expose them as a small prompt-local
action space, and require the generated derivation to execute its declared
action sequence. This makes MATH structurally analogous to canonical graph
coloring and Countdown while retaining free-form mathematical derivations and
the calibrated semantic validity gate.

## Immutable hypothesis

Given a frozen menu of at least two sound, substantively distinct solution
routes per problem, the E46 normalized canonical-bank Haarnoja objective will
increase verified strategy coverage relative to a matched, execution-gated
Dr.GRPO control without materially reducing ordinary answer accuracy.

## Data

### Hard toy

- Source train/eval rows are exactly E49B's immutable 50-row level-3/4
  hard-solvable training subset and 50-row level-3/4 MATH-500 evaluation
  subset.
- The source row order, answers, and split membership may not change.

### Full stage

- Training is exactly the 384-row OAT MATH training cohort in
  `var/data/math12k_384_math500/train`.
- Evaluation is exactly its 500-row MATH-500 split.
- Both stages use three prompt epochs.

The only data transformation is appending a frozen strategy menu and its
execution instructions to each problem. Original problem text and reference
answer are retained in separate fields for audit. Ground-truth answers are
never supplied to the 72B menu generator or route ideator. They are supplied
only to the two frozen soundness/novelty auditors, are hashed into the
evidence binding, and are never relayed in regeneration feedback or embedded
in policy prompts.

## Frozen menu construction

For each problem, frozen Qwen2.5-72B proposes:

- a problem-local action vocabulary `A1..An`; and
- preferably three, but at least two, strategies `S1..Sk`, each an ordered,
  non-repeating action sequence.

Actions must be concrete mathematical operations. Strategies must not contain
or leak the final answer. Two independently prompted, temperature-zero audit
passes, with seeds 491711 and 491712, have complementary frozen roles:
`soundness_execution` independently solves the problem and literally executes
each route against the auditor-only reference answer; `equivalence_attack`
tries to reduce every pair to a shared mathematical core under routine
algebra. Both must certify:

1. every action sequence is sound, sufficient, and actually derives an answer
   equivalent to the reference answer; and
2. every pair survives the adversarial equivalence attack and has different,
   explicitly named route-exclusive decisive operations rather than prose,
   distributivity, fixed arithmetic notation, or routine algebra.

Failed menus may be regenerated at most four times using only answer-blind
failure codes and generic replace-route directives as feedback, followed by
at most four separately elicited route-rescue attempts. Auditor derivations
and derived/reference answers never enter feedback. Every row must pass both
final audits; otherwise the stage does not launch. The exact v3 rationale and
the two v2 false admissions that motivated it are preserved in
`e49c_reference_answer_adversarial_audit_amendment_20260724.md`.

## Exact policy contract

Every completion must start at character zero with exactly:

```text
<strategy_id>Sj</strategy_id>
<action_combo>Aa>Ab>...</action_combo>
```

The ID and action combo must be one exact pair from that problem's menu.
Missing, unknown, reordered, or mismatched declarations fail deterministically.

For an answer-positive response that passes the parser, two independently
permuted temperature-zero Qwen2.5-72B audits check both:

- mathematical integrity of the written derivation; and
- material execution of every declared action, in order, as the decisive
  route.

Name-dropping, switching routes, skipping an essential action, or appending
the declared route as an irrelevant afterthought fails. Any invalid,
ambiguous, disagreeing, missing, malformed, or truncated audit receives no
canonical key. In both arms, an answer-positive response keeps its task reward
only if it receives a key; otherwise its task reward is set to zero before
ordinary Dr.GRPO centering.

After those gates, the canonical outcome is the prompt-local audited action
combo. Thus formatting paraphrases merge exactly, while a label cannot create
novelty without executing its associated route.

## Matched arms

Both arms use:

- Qwen2.5-0.5B-Instruct, same immutable revision;
- seed 45;
- group size 16;
- learning rate `2e-7`;
- one PPO epoch;
- temperature 1 and top-p 1;
- the same menu-augmented train and eval rows;
- the same execution-gated task reward;
- the same frozen 72B endpoint and two judge seeds;
- three prompt epochs;
- one A100 each on node302.

Arms:

1. **Matched execution-gated Dr.GRPO** — passive audited strategy tracking,
   no exploration advantage.
2. **E49C finite-action normalized canonical Haarnoja** — E46 objective:
   initial/minimum alpha 0.10, maximum alpha 0.50, normalized entropy target
   0.80, log-alpha learning rate 0.003, EMA decay 0.90, pseudocount 1,
   surprisal clip 5, and first audited discovery bonus beta 0.50.

No other entropy, semantic, DIAYN, XDR, or curriculum objective is active.

## Evaluation

The ordinary frozen MATH verifier reports greedy accuracy and fixed-seed
pass@8 at steps 0, 50, 100, and 150 for the toy. A post-hoc audit on the exact
stored evaluation responses additionally reports:

- execution-gated greedy accuracy;
- execution-gated pass@8;
- mean audited correct strategies per prompt at 8 samples; and
- fraction of prompts with at least two audited correct strategies.

Raw and execution-gated metrics must both remain visible.

## Hard-toy advancement gate

The full stage may launch only when all conditions hold:

1. all 100 toy menus pass both frozen audits;
2. source, code, protocol, menu evidence, and endpoint identities are pinned;
3. both jobs complete all 150 optimizer steps without judge-format crashes;
4. reward accounting is exact for every group:
   accepted + contract-rejected + integrity-rejected + ambiguous-rejected +
   disagreement-rejected = answer-positive;
5. both arms have identical step-0 raw evaluation metrics;
6. by the terminal checkpoint the treatment has mean verified training support
   at least 2.0 per tracked prompt and at least 50% of tracked prompts have
   support at least 2;
7. terminal treatment execution-gated pass@8 is at least the matched control;
8. terminal treatment raw pass@8 is at least the matched control;
9. terminal treatment raw greedy accuracy is no more than 5 percentage points
   below the matched control;
10. the logged dual direction is correct whenever alpha is not projected at a
    bound, and alpha remains finite and inside `[0.10, 0.50]`.

If the toy fails, no full run launches. A successor may alter prompting,
menu granularity, or reward scale only under a new protocol/run stamp; failed
artifacts remain immutable.

## Full-stage success criterion

The 384-train/500-eval, three-epoch stage is successful when both jobs finish,
all accounting and controller invariants pass, treatment execution-gated
pass@8 and raw pass@8 are each at least their matched controls, treatment raw
greedy is within 5 percentage points of control, and treatment has materially
higher audited multi-strategy coverage (at least +5 percentage points of eval
prompts with two or more audited correct strategies at K=8).

## Frozen implementation identities

The following SHA-256 identities were recorded before any E49C menu request or
training submission:

```text
c3e477f26183f917f6804405a7d607e2241d68cc44cb81c1e8f1f0a6cdb7e1a3  src/oat_drgrpo/math_strategy_menu.py
f9855ce3842d30bcfad58f6e1f8c0e151bf8e97b3fc158f7ab2f10bba549bf32  src/oat_drgrpo/math_strategy_canonicalizer.py
4e984c3cba54dbec50984b3c571c53adbf99ce5606ed8e42efd1f76bdfbbece7  src/oat_drgrpo/online_canonical_bank.py
d5fed3bbcea20807e72015ae88a2ef00de0af0e5ec5ca6049cb04c07d147ce0e  src/oat_drgrpo/online_canonical_controller.py
58976324034691fe001ded0c9a84ddb8c76a1f416b2de17106da5c0e99b39a01  src/oat_drgrpo/learner/grpo.py
58d98b326a07316a2d873b3d9e20ebdb04f44ff519303735ce0b2f10873f2261  src/oat_drgrpo/args.py
8111b291345d0a88c9fe98cbfa47eb18f190a8242525330d1540bf1ae37d2dbc  ops/math_strategy_calibration/materialize_e49c_strategy_menu_data.py
bb1fc918cd5bbd90a957d2666698b1e4facbd153edbaf80dd37e639dbb0c4275  ops/exp_scaling/launch_e49c_finite_action_math_05b.sh
0defade3800a91e3e56fbbfde508cda26b8e95283e1878ee556cda709465f04b  ops/train.sh
9c81ca954723c4a9e9a94eb83a26df2fd66588f697350a90067e0664b52b03a1  ops/submit_countdown_comparative.sh
1d4a113c9a014d2b7b3c9d52e0c5ddc86342a37df3f467bd6dcf900ab20a912d  paper/preregistration/e49c_menu_format_amendment_20260724.md
08f4264d644ddfd166016108143a9a17faf2953c078520f775f0fe3ee287ac42  paper/preregistration/e49c_route_ideation_amendment_20260724.md
ca431c15ba18981f34090b2a70a17ffe110ed09ccd30e0cb8f7ea7b7d1ca71ba  paper/preregistration/e49c_conservative_menu_veto_amendment_20260724.md
c565e64590421e239a31b128571fe7f65e19484eeb36ee580c2ee1fab4489a07  paper/preregistration/e49c_reference_answer_adversarial_audit_amendment_20260724.md
dfdc9a0646af84fb84200ae2d1bd60c28dc7c1ce9d4ddecca10886125a613b59  tests/test_e49c_menu_audit.py
e1e6a9fc09e2e3ac79e9a97bda4d980ae8d82d958a0918d18c9d247f1c586abd  ops/exp_scaling/babysit_e49c_finite_action_math.sh
```
