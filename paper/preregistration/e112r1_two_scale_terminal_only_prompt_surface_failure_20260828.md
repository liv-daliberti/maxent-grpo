# E112-R1 two-scale terminal-only amendment after trajectory audit failure

Superseded on 2026-08-28 by A7 after the alleged prompt drift was traced to
outcome-dependent `answer_keys` in the nominally response-free hash. This file
is retained as an audit incident record and is not an analysis authorization.


Recorded: 2026-08-28, after executing the existing registered full-trajectory
builder on the frozen 50-cell two-scale membership.

## Fail-closed trigger

The builder aborted before writing a result because the Qwen2.5-0.5B Graph
seed-43 treatment run changed its response-free prompt surface across the
17-checkpoint grid. This violates the frozen A3 trajectory identity contract.
The registered trajectory AUC, baseline-centered trajectory sensitivity, and
any claim requiring a constant prompt bank across checkpoints are therefore
unavailable. They must not be approximated, partially pooled, or silently
redefined.

This failure was detected by response-free metadata hashes. It does not depend
on the sign or magnitude of an endpoint effect. The terminal endpoint values
had already been viewed in the author-requested private 50-cell look.

## Terminal-only admissibility check

The exploratory public disclosure may proceed for update 3,072 only if every
one of the 50 frozen treatment/comparator pairs passes all of the following:

1. both registered completion markers reach the target horizon;
2. all four fixed sampled-K terminal draws exist and match the frozen row
   contract;
3. the response-free ordered prompt projection is constant across the four
   terminal draws;
4. request-seed identities are distinct across the four draws; and
5. the complete terminal prompt/request identity object is exactly equal
   between treatment and its registered ReplayDr.GRPO comparator.

If any pair fails, no two-scale public endpoint result is emitted. If all pass,
report only the terminal pass@8 and terminal correctness-adjusted-breadth
paired effects, five-seed family means, and paired 95% Student-t intervals.
The machine-readable result may retain raw terminal distinct@8 and per-draw
Monte Carlo diagnostics.

## Disclosure

The paper must say that trajectory AUC was planned but failed its prompt-
surface identity audit. The terminal result remains an exploratory bundled
historical-comparator contrast after repeated looks. The registered 75-cell
confirmatory criteria and Qwen2.5-3B scale remain unevaluated.
