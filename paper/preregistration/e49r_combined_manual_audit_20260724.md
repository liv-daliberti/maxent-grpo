# E49R combined blinded manual audit and toy-bank materialization

**Status: FROZEN BEFORE E49R PACKET GENERATION OR LABELING — 2026-07-24**

## Inputs and purpose

E49R is the final fail-closed audit over the toy support bank. It combines:

- the four pairs already retained by the earlier blinded E49E manual audit;
- nine strict E49H survivors;
- two strict E49K survivors;
- four strict E49L survivors;
- one strict E49P survivor; and
- four E49Q visible-trace recoveries.

The relevant candidate-record SHA-256 values are:

- E49H: `42da9d325425f6a5939d132e0886ec7ca1acc220eb2830ab7ba4457a2d489098`;
- E49K: `f099377ead4226427b2b0661e53aa1d7217123669c488202e1e7f929458ead0f`;
- E49L: `948408214521f84e35021f4637def2d42b0636e347958e920a3c24fcdec23d0f`;
- E49P: `00c34f4cd1b03c4212dd8c3f3b5a8634247b01f47c9bf293dbd067b00e2268b2`;
- E49Q: `88cc58a2d19cf7c2ce89dcb09e113ef250e5a2f029385226f2caddd2d743414c`.

The base 100-row audited-record SHA-256 is
`b945211f0baa20800e322958ef9095af034c354cc123043fcaa2eaf63ef9e484`.
Its earlier decision correctly blocked training and is not rewritten.

## Frozen packet

Before reading any private mapping, E49R writes an order-randomized packet
containing exactly:

- 20 new distinct-strategy claims;
- one zero-support singleton-repair check for `eval:0010`, presented as two
  equivalent renderings of its sign-bound strategy; and
- five hidden equivalent/paraphrase controls sampled from the new claims.

Thus the packet contains 26 blinded pairs. Each row shows the problem,
reference answer for soundness, both declared action programs, and every
independent execution trace used upstream. Display order and pair IDs are
deterministic from the frozen packet seed but do not reveal source kind.

Labels are written before the private mapping is read. For every pair the
auditor must state whether each displayed route is sound and self-contained,
whether the decisive algorithms are genuinely distinct, and a nonempty
rationale.

## Advancement rule

Training is authorized only if all of the following hold after labels are
joined to the sealed private mapping:

1. manual false-new is exactly zero;
2. all hidden equivalent controls are labeled non-distinct and sound;
3. the singleton repair is sound and non-distinct, and its validated
   sign-bound route gives the old zero-support row a singleton menu;
4. every retained pair has two manually sound, genuinely distinct routes;
5. unique multi-route support is at least ten train and ten evaluation rows;
6. all 100 rows retain at least one validated route; and
7. every rendered prompt is at most the frozen context limit.

Support is counted by row, never by duplicate contract. Rejected claims fall
back to the previously audited singleton menu for that row. The resulting
dataset is the only toy dataset eligible for the matched three-epoch E46
normalized-Haarnoja versus Dr.GRPO run.
