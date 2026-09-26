# E49B: proof-gated MATH strategy Haarnoja at 0.5B

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Question

Can the successful graph-coloring/Countdown E46 mechanism transfer to
free-form MATH when every bank entry has passed final-answer validation, two
independent derivation-integrity checks, and conservative strategy
canonicalization?

E49B has a hard-but-solvable toy followed by the exact existing
MATH12K-384/MATH-500 OAT split. Both stages run three prompt epochs and compare
the treatment with a contemporaneous, compute-matched Dr.GRPO control.

## Executable admission chain

The policy never names its own strategy. For each response:

1. the repository's full `math_verify` final-answer validator must return one;
2. two independently permuted, temperature-zero Qwen2.5-72B-AWQ calls must
   classify the written derivation `valid`, each retaining a brief decisive
   mathematical check;
3. two further independently permuted Qwen72 calls partition only the
   doubly-valid responses and existing valid representatives by essential
   mathematical route;
4. whenever those full-context calls propose more than one reduced component
   or a new component beside an existing representative, two schema-bound
   pairwise Qwen72 audits retain a decisive comparison and return
   `same`, `different`, or `ambiguous` for every proposed boundary; and
5. the frozen component-incidence rule either safely reuses one key, creates
   a new key only under unanimous separation, or rejects the response.

Any negative answer score, inactive row, no-EOS exclusion, invalid/ambiguous
derivation, omitted ID, conflicting representative incidence, transport
failure, parse failure, or uncertain new boundary receives no key and no
exploration reward. Raw prompt and response text used by the task validator
is the exact text sent through this chain.

The frozen semantic executable is E47W,
`math_strategy_canonicalizer_pair_veto_v15`. Its live regressions verify
valid-anchor admission, invalid-proof rejection, routine-equivalent merging
in both direct and adversarial multi-solution contexts, and different-proof
separation. Its SHA-256 is
`91a1a8fdc3b29fa49b1089154f5955ec2d9e759cc85e94b35bc7d101d81bf988`;
the launcher refuses any other source before snapshotting it for training.
The endpoint record must likewise name the calibrated Qwen72 checkpoint
revision `698703eae6604af048a3d2f509995dc302088217` and frozen serving
configuration. The strategy representative state and E46 count/controller
state are checkpointed separately.

The matched Dr.GRPO arm performs the identical answer validation, conditional
two/four/six-call judge chain, strategy tracking, checkpointing, and telemetry,
but has strategy entropy and novelty coefficients exactly zero.

## Frozen E46 objective

No policy-token or verbal uncertainty controller is used. The treatment is
the existing normalized canonical-bank Haarnoja method:

- group size `16`;
- `rho_x = H(q_x) / log |B_x^+|`, eligible only for `|B_x^+| >= 2`;
- target `rho*=0.80`;
- `alpha_0=alpha_min=0.10`, `alpha_max=0.50`;
- log-alpha Adam learning rate `0.003`, betas `(0.9,0.999)`, epsilon `1e-8`;
- entropy EMA decay `0.90`;
- novelty coefficient `0.50`;
- pseudocount `1`, surprisal clip `5`;
- immutable pre-group snapshots, row-order-independent commit, exploration
  advantage added after ordinary Dr.GRPO task centering, and controller
  observation after the bank update.

All other diversity objectives are zero.

## Model and optimizer

Both arms use the immutable local
`Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`,
neutral `qwen_math` prompting, learning rate `2e-7`, constant schedule,
Dr.GRPO, one PPO epoch, beta zero, max norm one, temperature one, top-p one,
and prompt/response/context limits `1024/1024/2048`. Each arm uses one A100 on
node302. The only treatment difference is the detached E46 strategy-bank
advantage and its alpha controller.

## Stage A: hard-but-solvable toy

The original level-five toy is retired: after invalid proofs and a false
routine boundary were removed, it contained no trustworthy prompt with two
policy strategies, so normalized bank entropy could not operate.

The prospective replacement is a fixed 50-row slice of the exact OAT train
set. Eligibility uses immutable, pre-E49 first-epoch Dr.GRPO rollouts from
seeds 43, 44, and 45:

- MATH difficulty level 3 or 4;
- mean answer reward in `[0.10, 0.55]`;
- nonzero answer reward in at least two of three seeds.

The 56 eligible rows are ordered by
`sha256("e49b-support-toy-v1" || unique_id)` and the first 50 are taken. This
is an explicit mechanism-feasibility screen, not a held-out quality claim.
The selected set has 27 level-3 and 23 level-4 problems, historical mean
group reward `0.2954`, and all seven subjects. Its frozen train tree hash is
`7583599fcfd71494001974035d30459fe11445bd56478a9089b7d8612b3929ec`.

Toy evaluation contains 25 deterministic held-out MATH-500 level-3 rows and
25 level-4 rows. Its tree hash is
`9cdd064a024df1036c2024d7f9f6f375800be888b1399a950897026210067ff3`.

Seed 45 runs exactly three prompt epochs (150 prompt groups). Greedy pass@1
and stochastic pass@8 are evaluated at initialization and after every epoch.

Stage B may launch only when Stage A shows:

- exact canonicalizer accounting and no validator-negative admission;
- nonzero accepted strategy coverage, terminal mean support of at least two
  validated strategies per solved prompt, and monotonically nondecreasing
  cumulative support;
- finite nonzero exploration advantage in treatment and exactly zero
  objective influence in matched Dr.GRPO;
- normalized entropy remains eligible after support growth, its terminal EMA
  is at least `0.50`, and alpha moves in the prescribed error direction;
- no late non-finite or support-collapse failure;
- terminal treatment greedy accuracy is within two points of matched Dr.GRPO;
  and
- treatment greedy improves, or pass@8 improves while matched Dr.GRPO does
  not improve more.

## Stage B: exact OAT MATH split

Training and evaluation reuse byte-for-byte:

- `var/data/math12k_384_math500/train`, 384 rows, Arrow SHA-256
  `359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8`;
- `var/data/math12k_384_math500/eval`, full MATH-500, Arrow SHA-256
  `2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7`.

Seed 45 runs exactly three prompt epochs (1,152 groups). Full MATH-500 greedy
pass@1 and pass@8 are measured initially and after each epoch. Checkpoints are
written after each epoch and the terminal model is exported.

## Success criterion

“Works like graph coloring and Countdown” requires the same mechanism pattern:
validated support grows, normalized entropy remains live after support reaches
two, alpha responds to entropy error rather than epoch number, and quality is
retained or improved against matched Dr.GRPO. E49B succeeds only if both full
arms finish all three epochs, no bank-integrity violation occurs, treatment
finishes with mean support of at least two and normalized-entropy EMA at least
`0.50`, improves full-MATH-500 pass@1 or pass@8 from initialization, and
terminal pass@1 is no more than two points below matched Dr.GRPO.
