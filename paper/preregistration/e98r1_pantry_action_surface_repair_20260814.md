# E98-R1 Pantry canonical action-surface execution repair

Date: 2026-08-14. This amendment was written after Pantry seeds 43 and 44
failed deterministically and before any repaired Pantry training cell ran.
Seeds 45--47 were held before their original frozen jobs started.

## Failure and scope

E98-R1's frozen offline pool stores the verifier-ready PantryPlan allocation
witnesses emitted by its collection policy, for example
`carrots=50;pumpkin_seeds=100;sunflower_seeds=100`. The registered E98-R1
Pantry learner instead acts through the six-position binary
`pantry_support_mask` policy surface. The RLEP materializer tokenized the full
allocation as ordinary text, then the canonical scorer correctly rejected
those token IDs as outside the binary action support.

This is a missing representation adapter, not evidence that a sampled
trajectory is invalid and not a change to the E98-R1 objective.

## Frozen repair

For each sampled offline Pantry row, the repaired runtime:

1. validates the original allocation against its exact frozen prompt reference;
2. takes the validated allocation's selected ingredient IDs;
3. emits one binary inclusion decision in the prompt's public six-row order;
4. maps each decision through the frozen canonical action-space token ID; and
5. adds no EOS or ordinary-language tokens to the six-action trajectory.

The conversion fails closed if the reference is malformed, the allocation is
not verifier-valid, the prompt does not contain six rows, or the support cannot
round-trip through the registered mask.

Every sampled row remains present, including duplicates and distinct quantity
witnesses with the same support. Therefore the frozen pool's empirical action
frequency is preserved: there is no deduplication, mode balancing, row
filtering, resampling, reward change, or replay-dose change. Quantity is not a
policy action in this registered Pantry arm; the trusted environment already
projects a selected support to a verifier-ready quantity allocation.

Only three files differ from the original E98-R1 snapshot:
`pantry_support_action.py`, `canonical_actions.py`, and
`learner/grpo.py`. Their old and new hashes are stored in the repair ledger.

## Execution gate and replacement

A non-scientific 32-step Pantry seed-44 smoke runs first. Seed 44 is used
because its failed trace reached two ordinary fallback updates before the first
eligible replay prompt, so the prefix exercises both registered branches. The
existing E98-R1 smoke audit must observe both (fallback, replay) dose pairs and
finite RLEP telemetry before any scientific replacement becomes eligible.
The non-scientific smoke may use healthy-in-Slurm A6000 nodes 103, 104, 205,
206, 207, or 805 in the eligible `all` partition; an initially audited node201
constraint was corrected while the job was still pending because node201
belongs only to `vertaix`. Node206 is allowed only for this disposable gate,
not for checkpoint-bearing scientific resumes.
The smoke uses a one-hour limit in the short-job `all` lane; the preceding
E98-R1 32-step smoke completed in 6 minutes. Its 15.9 GB measured peak host
memory also supports a 32 GB request with a 2x margin, replacing the original
64 GB request.

All five Pantry jobs are replaced so the domain uses one runtime. The two
failed jobs restart from initialization; neither produced a checkpoint. The
three not-yet-started jobs are cancelled only after held replacements are
audited. Seeds 43--44 move from drained node105 to healthy nodes 202--204 while
retaining the A5000 GPU class. Seeds 45--47 retain A100 node302. This placement
change is operational only.

The original job IDs and terminal states remain in both the primary run repair
history and the dedicated repair ledger. E98's earlier feasibility failure and
E98-R1's sparse estimand remain unchanged.
