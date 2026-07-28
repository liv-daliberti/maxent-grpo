# E47-CAL: hard-MATH strategy-canonicalizer calibration for E46

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Scope

E47-CAL is a calibration gate, not a training result and not a replacement for
E46.  It tests whether a blinded Qwen2.5-72B strategy judge can safely provide
canonical reasoning-strategy keys for a later MATH arm that otherwise uses the
current E46 normalized online-canonical Haarnoja method.

The later arm, if the gate clears, retains E46 exactly:

- `rho_x = H(q_x) / log |B_x^+|`, observed only for `|B_x^+| >= 2`;
- target `rho*=0.80`;
- canonical coefficient `alpha_0=alpha_min=0.10`, `alpha_max=0.50`;
- log-alpha Adam learning rate `0.003`, betas `(0.9, 0.999)`, epsilon
  `1e-8`, and entropy EMA decay `0.90`;
- novelty coefficient `0.50`, bank pseudocount `1`, and surprisal clip `5`;
- group size `16`, group-snapshot scoring, validator-first bank admission,
  and post-update controller observation.

The proposed policy-token-uncertainty controller is excluded.  E47-CAL must
not use policy entropy, confidence, verbal uncertainty, judge confidence, or
any other model-generated uncertainty to set alpha.

## Frozen data

The source is the existing immutable E39 training artifact
`var/data/math12k_384_math500/train`, whose Arrow file has SHA-256
`359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8`.
MATH-500 is evaluation-only and is not touched.

The pilot uses 50 level-5 problems.  Within subject, rows are sorted by
`sha256("e47-cal-selection-v1" || unique_id)` and the following fixed quotas
are taken:

| Subject | Rows |
|---|---:|
| Intermediate Algebra | 15 |
| Geometry | 8 |
| Algebra | 7 |
| Number Theory | 6 |
| Prealgebra | 5 |
| Precalculus | 5 |
| Counting & Probability | 4 |

The materialized ordered problem IDs and source indices are written before
generation and hashed into the pilot manifest.

## Frozen policy sampling and validation

For every problem, the immutable local
`Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`
produces 64 independent samples from the ordinary neutral `qwen_math` prompt.
Sampling uses temperature `1`, top-p `1`, maximum response length `1,024`, and
seed `470064`.  There is no diversity, strategy, action-list, or judge-facing
instruction in the policy prompt.

All 3,200 samples are graded with the repository's full
`boxed_reward_fn(..., fast=False)` MATH validator.  A policy sample reaches the
semantic judge only when this validator returns reward one.  Invalid,
unparseable, and merely plausible prose is retained in the grade audit but is
never admitted to a strategy bank.

## Blinded same-strategy injections

Each problem contributes a validator-positive official-solution anchor and
three same-strategy controls:

1. an exact duplicate;
2. a formatting/whitespace-only rendering;
3. a conservative lexical paraphrase of connective prose that does not alter
   mathematical expressions.

Every rendered control is independently revalidated.  Any problem whose four
controls are not all validator-positive fails closed and is excluded from the
denominator while being reported as a construction failure.

Opaque IDs are assigned by salted SHA-256.  The semantic judge receives a
seeded permutation containing validator-positive policy samples and controls;
it is not told item source, injection type, sample index, correctness score,
gold status, expected pair label, or E46 bank history.  The hidden injection
key is stored separately and opened only by analysis.

## Frozen 72B strategy judgment

The judge is
`Qwen/Qwen2.5-72B-Instruct-AWQ@698703eae6604af048a3d2f509995dc302088217`
served through the local OpenAI-compatible vLLM endpoint.  Frozen identity
hashes are:

- `config.json`:
  `ec4813b56d971f7a9c4490c7c148f997c468e97e9e8ae39e80f380d51a6dda1f`;
- `tokenizer.json`:
  `22bfef58b74d2fa40f402ac0c8d638d9e54d12137da2e2c7e9f32a1e7534e42f`;
- `model.safetensors.index.json`:
  `9f9e509271b4a2da37444f8a7ac3b3758279f6044c9c2f6eb40edc171329a65a`.

For one problem at a time, the judge partitions all validator-positive
solutions by essential mathematical route.  Rewording, notation,
formatting, reordered routine algebra, and skipped or expanded routine steps
must remain in one cluster.  A new cluster requires a different central
mathematical idea, construction, theorem, substitution, case decomposition,
or proof route.  Final-answer validation is not delegated to the judge, but it
must mark a path ambiguous when the written reasoning is internally
incoherent or does not actually establish the validated answer.  Ambiguous
items fail closed and cannot earn a new-strategy reward.

Judgment is repeated twice at temperature zero after independent item
permutations with seeds `470721` and `470722`.  Cluster names are irrelevant;
analysis uses only pairwise co-clustering.  For prospective novelty admission,
an item is a new strategy only when both blinded passes support the same
separation.  Disagreement is not new.

The AWQ checkpoint's enforced context length is `32,768` tokens.  Before any
judge call, the exact pinned tokenizer measures the complete chat-formatted
request.  Solution renderings are symmetrically compacted (preserving both
the beginning and final answer) until the request is at most `24,000` tokens;
`8,192` tokens are reserved for the JSON partition.  The per-item character
cap, exact preflight prompt-token count, and IDs whose renderings were
compacted are retained.  Full unabridged texts remain in the manual-audit
packet.  No unsafe positional-length override is permitted.

The node105 service uses tensor parallelism four over A5000s and
`max_num_seqs=8`, twice the frozen four-call judge concurrency.  This prevents
vLLM's irrelevant 256-sequence sampler warm-up from consuming memory while
leaving the model, context, request contents, and decoding contract unchanged.
It uses eager execution with GPU-memory utilization `0.88`; CUDA graphs are a
throughput optimization and exceeded A5000 headroom during pre-outcome
startup.  Disabling them does not alter the checkpoint, tokenizer, request,
temperature, or strategy-equivalence rule.

## Manual audit

Before opening judge labels, the pipeline writes a blinded audit packet:

- 50 same-strategy pairs, one independently permuted injected pair per problem;
- up to 50 validator-positive policy-policy pairs, balanced between judge
  same/different calls where the available pool permits.

The packet contains the problem and two complete solution texts, but no model
cluster, injection type, or expected label.  A human records
`same`, `different`, or `uncertain`, plus a short reason.  The immutable
completed labels and packet hash are inputs to final analysis; `uncertain`
pairs are reported and excluded from binary rates.  Injection labels remain a
separate, mechanically known stress test and are not substituted for the
manual audit.

## Primary and secondary measurements

The primary endpoint is false-new rate on the 150 anchor/control comparisons:

`FNR_new = count(control not co-clustered with anchor) / valid comparisons`.

It is reported overall and separately for exact duplicates, formatting
variants, and lexical paraphrases, with Wilson 95% intervals.  We also report:

- per-problem any-false-new rate;
- pairwise partition disagreement across the two permutations;
- judge parse, missing-assignment, ambiguity, and truncation rates;
- on manually labeled pairs, same-pair false-new rate and different-pair
  false-merge rate;
- number and fraction of validator-positive 0.5B samples;
- number of apparent policy strategy clusters, explicitly labeled as a judged
  sample estimate rather than exhaustive support.

As a mechanism diagnostic only, the judged policy stream is replayed in the
original sample order through E46's exact group-16 normalized bank-entropy
sensor and Haarnoja controller.  Injected controls never enter this replay.
Both judge passes are replayed separately; their alpha and normalized-entropy
divergence quantifies the controller sensitivity to canonicalization error.
No policy update is performed in E47-CAL.

## Frozen gate

The semantic canonicalizer clears the calibration gate only if all conditions
hold:

1. all 50 problems have four validator-positive injected controls;
2. exact-duplicate false-new is exactly zero in both judge passes;
3. overall injected false-new is at most `0.05`, and the upper Wilson 95%
   bound is at most `0.10`;
4. lexical-paraphrase false-new is at most `0.10`;
5. pairwise partition disagreement is at most `0.05`;
6. judge structural failure plus ambiguity is at most `0.05`;
7. among manually labeled same pairs, false-new is at most `0.05`;
8. among manually labeled different pairs, false-merge is at most `0.20`; and
9. the two E46 replays differ by at most `0.03` in terminal alpha and `0.05`
   in mean eligible normalized entropy.

Failure blocks new-strategy reward on MATH.  Passing supports only use of this
judge and prompt as a conservative sampled strategy canonicalizer for an
E46-style experiment; it does not establish exhaustive strategy support,
proof equivalence, or correctness independent of the symbolic validator.
