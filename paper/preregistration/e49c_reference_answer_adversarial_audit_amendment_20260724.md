# E49C reference-answer/adversarial menu-audit amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49C TRAINING LAUNCH**

A second spot audit of preprocessing evidence found two v2 false admissions:

- computing non-Honda percentages before multiplying by the total versus
  multiplying each percentage first and subtracting the counts was labeled
  distinct, although the routes differ only by distributivity; and
- a cone-height route using the sine of half the paper sector's central angle
  was labeled sound, although that angle does not give the cone's axial
  cross-section relation.

Menu job `30073111` was canceled after six durable v2 records. No E49C policy
training job or materialized menu dataset existed. Those records remain as
failed calibration evidence and are ineligible for training.

The menu audit contract is therefore versioned
`reference_answer_adversarial_novelty_veto_v3`. The generator and route
ideator remain answer-blind. Only the two audit passes receive the reference
answer, and audit derivations or answers are never relayed to generation
feedback or embedded in policy data.

The two frozen audit passes now have different fail-closed roles:

1. `soundness_execution` independently solves the problem and literally
   executes each declared action sequence; it records the answer actually
   derived and must certify equivalence to the reference answer.
2. `equivalence_attack` first tries to collapse each strategy pair to a
   shared decisive equation, identity, search space, or invariant under
   routine algebra. It may authorize novelty only if that attack fails and
   it names a genuinely route-exclusive decisive operation on each side.

Both passes still use frozen Qwen2.5-72B and seeds 491711/491712. Every
strategy must be sound, have failure code `none`, and derive the reference
answer. Every pair must be called distinct by both roles and provide
nonempty, unequal route-specific decisive operations. Missing, malformed,
ambiguous, equivalent, or answer-mismatched output vetoes the menu.

The equivalence instructions now explicitly merge:

- subtracting percentages before multiplying by a total versus multiplying
  the components first and subtracting their counts;
- a fixed list of repeated multiplications versus exponent notation for the
  same product; and
- any alleged recurrence/closed-form split that has no genuine state
  relation or induction argument.

Direct versus complementary counting remains potentially distinct only when
the two routes count genuinely different configuration sets. Routine
distributive arithmetic is not a distinct counting method.

All earlier v1/v2 rows require v3 re-audit or regeneration; none is
grandfathered. The online execution validator, exact task-reward gate,
canonical keys, E46 controller, matched arms, training data cohorts, three
epochs, and advancement criteria are unchanged.
