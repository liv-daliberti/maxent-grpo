# Tiny draft: open-bank MaxEnt should be replay, not advantage credit

Status: the initial matched cells (Slurm 30579748--30579750) were canceled at
zero GPU time after the mechanism smoke exposed missing replicated-sampling
plumbing; no run directory or metric was used. E101m r1 (30582184, node204)
completed eight updates and E101m2 (30582404, node205) completed 32 updates.
Both validated isolation and ordinary replay actuation, but untouched-prompt
proposal sampling found no validator-positive row. The repaired matched screen
therefore remains gated. E101m3 (30582661, node205) completed 32 updates in
242 seconds via `all`/`mltheory`. Its exact-grammar actuator was active, but the
one same-seed singleton anchor had no valid radius-two neighbor. E101m3 is a
mechanism-only result, not a replacement comparison arm.

## Claim

Keep semantic diversity outside PPO's task advantage. Use task-only Dr.GRPO, a
verified exemplar bank, and two direct replay derivatives: one that preserves
common verified mass and one that equalizes scores among modes already in the
bank. Handle genuinely new modes with a separate verifier-gated explorer that
can add an exemplar to replay support but can never add a PPO row. On easy3
Countdown, the clean candidate is a fixed local neighborhood in the public
three-position binary-expression grammar. Legacy sign rewrites remain disabled
because the validator accepts their unary-negative keys outside the dataset's
declared exact support.

For bank B, the clean objective is

    L = L_PPO_task
        + lambda_m [-mean_{y in B} s_theta(y)]
        + lambda_b KL(U_B || softmax_B(s_theta)).

The explorer changes B, not L_PPO_task. This is important because an unseen
mode has no sequence to differentiate: an unseen bucket is a sensor, not an
actuator. Once a verified sequence is admitted, direct replay supplies the
gradient.

## What we already knew

The adaptive combo is also already in flight. The current program ledger records
34 terminal and 21 active adaptive-semantic-plus-replay cells, while E90's
bank-normalized adaptive replay has 9 terminal and 16 active cells. E88's first
RMS controller target was unreachable in MathIR and Python factors, pinning eta
at .40 for 97--98% of applied updates. Another adaptive-advantage/replay cohort
would duplicate an existing program rather than answer the objective-placement
question.

Yes, E72 already compared replay mass against replay mass plus direct known-bank
balance. On Countdown, five-seed terminal distinct@8 was 1.8836 for mass-only
B1b and 1.8461 for mass+balance B1a. The paired mass-minus-balance mean was
+0.0375 with bootstrap interval [-0.0383, 0.1215]. Thus E72 did not show that
adding known-bank balance improves Countdown; scaling that same ablation would
repeat existing evidence.

Across E72 domains, the mass-minus-balance paired effects were heterogeneous:
-0.066 on Graph coloring, +0.037 on Countdown, +0.992 on Python factors,
-0.051 on MathIR, and -0.469 on PantryPlan. The bootstrap interval excluded
zero in opposite directions for Python factors ([+0.733, +1.324]) and
PantryPlan ([-0.677, -0.265]). A fixed balance coefficient is therefore not a
universally beneficial replay add-on; any adaptive version needs a bank-local
mechanism gate rather than another task-advantage combination.

The missing comparison is open support: matched mass+balance with versus without
a separate new-mode admission path, with every semantic advantage coefficient
exactly zero.

## Score-space toy

Start with bank scores (-0.2, -2.2), so the rare mode has conditional bank mass
0.119. Twelve unit score steps at lambda_m=lambda_b=0.1 give:

| objective | initial gap | final gap | final rare-mode bank mass |
|---|---:|---:|---:|
| mass only | 2.000 | 2.000 | 0.119 |
| mass + balance | 2.000 | 1.205 | 0.231 |

Mass alone raises both scores equally, so their gap is unchanged. Immediately
after admission, the combined derivative gives the rare mode 7.39 times the
upward score pressure of the common mode. Before admission, balance on the
singleton is exactly zero. This isolates why discovery and balance must be
separate operations.

Concrete Countdown example: numbers [2, 3, 4], target 10, and initial bank
`{2*3+4}`. Its balance derivative is zero. A fixed radius-two edit in the
public action grammar proposes `3*4-2`; the validator accepts this second exact
canonical key, and only its token sequence enters replay. At scores (-0.2, -2.2),
the combined loss gives upward score pressures 0.0119 to the incumbent and
0.0881 to the new weak mode. The fresh mode therefore gets 7.39 times the
push, while PPO still trains only on the ordinary task-rollout group.

## Better direct MaxEnt candidate

A second score-space toy exposes a weakness in the unprojected direct loss. For
mass weight lambda_m, requested balance weight lambda_b, and a bank of size n,
use the largest bank-local balance coefficient that is retention-safe:

    g_i = -lambda_m/n + lambda_b (q_i - 1/n),
    lambda_b_safe = min(lambda_b, lambda_m / (n q_max - 1)),
    g_i_safe = -lambda_m/n + lambda_b_safe (q_i - 1/n).

The middle rule applies when `n q_max > 1`; at a uniform bank, keep the requested
coefficient. This is the maximum KL strength for which every `g_i_safe <= 0`.
It therefore preserves the mass term's total score gradient while never directly
assigning a verified bank row downward score pressure. With equal raw mass and
balance weights, the unprotected gradient lowers mode i exactly when `q_i > 2/n`;
this is impossible for two modes but possible as soon as n is at least three.
That regime is not merely synthetic: the prior E56 0.5B Countdown sentinel had
one replay actuator group with four modes at step 192 and three modes in its
terminal snapshots.

For q=(0.9, 0.0333, 0.0333, 0.0333) and requested weights 0.1/0.1, the raw
gradients are (+0.0400, -0.0467, -0.0467, -0.0467): raw balance anti-trains the
common verified mode. The safe cap reduces only the bank-local balance weight
to 0.03846, giving (0, -0.0333, -0.0333, -0.0333). Weak modes retain more upward
score pressure without a direct downward derivative on the incumbent. The
bank-local cap, the earlier projection toy, and the mechanism auditor pass the
canonical replay and E101 contract suites (35 tests). The exact-grammar
transform has six additional focused tests. E101 remains frozen on the unprotected objective;
retention-safe balance is a separately named follow-up, not an after-the-fact
arm replacement. It is cleaner than another adaptive semantic-advantage controller.

## Fail-fast mechanism trials

| trial | placement | opportunity | result | interpretation |
|---|---|---:|---|---|
| E101m r1 | node204 A5000, `all/mltheory` | 1 group / 16 raw rows | 0 task-positive, 0 validator-positive, 0 admissions; 4 replay-actuation updates | proposal path and isolation work; no new sequence |
| E101m2 | node205 A6000, `all/mltheory` | 3 groups / 48 raw rows | 0 task-positive, 0 validator-positive, 0 admissions; 4 replay-actuation updates | more blind sampling is not the clean next move |
| exact-grammar CPU audit | 32 frozen train prompts | 136 valid anchor keys | radius 1 expands 89/136; radius 2 expands 113/136; zero support mismatches | dense support-blind local search inside the exact grammar |
| E101m3 | node205 A6000, `all/mltheory` | 1 exact-grammar opportunity plus 1 fallback raw group | 0 local candidates, 0/16 correct fallback rows, 0 admissions; 4 replay-actuation updates | safe radius-two search cannot expand this anchor |

E62's older evidence explains the raw-sampling result. Answer-conditioned
proposals copied the anchor (56/96 positive rows, all the same outcome), while
anti-copy conditioning made 233/240 rows invalid and admitted nothing. An
untouched-prompt sweep eventually found one novel Python outcome only after
hundreds of extra rows. The explorer, not the bank loss, is the current
bottleneck.

A follow-up CPU audit tested target-blind, fixed-seed distance-three code
subsets. Adding 16 long-jump codes expands 130/136 anchors (95.6%), but recovers
only 17/23 anchors that radius two misses. Reaching 135/136 requires 32
distance-three codes on top of the radius-two neighborhood. Since easy3 has
only 108 public grammar codes, that becomes near-enumerative, task-specific
solver search. It is not a clean general explorer and is not queued.

## E101 tiny Countdown screen

All cells use Qwen2.5-0.5B-Instruct, 32 train and 32 disjoint exact multi-answer
Countdown prompts, seed 101, four passes (128 updates), G=16, fixed replay
weights 0.1, and evaluation at 0/64/128 with K=8 over two draws. Each allocation
is capped at 55 minutes.

| arm | mass | balance | explorer | endpoint distinct@8 | endpoint pass@8 | mechanism |
|---|---:|---:|---:|---:|---:|---|
| mass | yes | no | no | not launched | not launched | gated |
| balance | yes | yes | no | not launched | not launched | gated |
| open | yes | yes | singleton-only raw sample | not launched | not launched | gated by discovery |

E101m, E101m2, and E101m3 are not comparison arms. Accuracy and diversity from
them are not estimates. E101m3 succeeds only if a singleton prompt triggers an
exact-grammar local proposal, a validator-positive novel mode is admitted with
zero sampled proposal groups and zero PPO/feedback leakage, and replay actuates
at or after that admission. E101m3 did not meet that gate.

## Decision

Do not return semantic advantage to PPO, and do not launch the frozen
raw-sampling screen. E101m/E101m2 identify model proposal quality as its limiting
step; E101m3 shows that a support-safe local mutation is still incomplete. The
next method should be a separately trained or asynchronously sampled proposal
distribution with a fixed validator budget, not more task-grammar enumeration.
It may add a verified sequence to replay support, but cannot add PPO rows or
change neutral counts.

Once that explorer demonstrates admission and later replay actuation, screen
`task PPO + replay mass + retention-safe bank balance` against mass-only before
adding adaptive controllers. If admission succeeds but replay does not actuate,
add fresh-admission scheduling. If safe balance does not beat mass despite
admission/actuation, focus on replay dose and scheduling. Adaptive gains cannot
repair a missing new-mode sequence, and semantic advantage combination should
remain retired.
