# E50 amendment: executable Python-factor ModeBench extension

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Reason for the amendment

At the user's direction, E50 is extended with a third ModeBench domain after
the original graph-coloring and Countdown jobs were launched. The existing
twelve jobs, their source snapshot, manifests, and protocol identity remain
immutable. The Python cells form a separately frozen, internally matched E50
post-launch domain extension and are not represented as contemporaneous with
the two original domains.

## Frozen domain

Each prompt supplies four distinct composite integers. The model must return
one pure one-line function of the exact form `lambda n: EXPR`. An isolated
Python worker calls that function once on every frozen input. A response is
valid only when every returned value is an integer proper divisor:

`1 < d < n` and `n % d == 0`.

The canonical outcome is the complete integer return vector from the same
execution that established correctness. Source-text differences with identical
behavior are one mode; different valid vectors are different modes.

The syntax gate admits only the lambda argument `n`, bounded integer literals,
arithmetic `+`, `-`, `*`, `//`, `%`, comparisons, Boolean operators, unary
operators, and conditional expressions. Calls, imports, attributes,
subscripts, containers, comprehensions, assignment expressions, and all other
names are rejected before execution. The worker has a per-request timer and
the parent process can terminate and replace it.

## Frozen data

- Schema: `python_factor_modebench_v1`.
- Seed: `5100`.
- Train/evaluation rows: `384/128`.
- Inputs per prompt: `4`.
- Maximum input: `96`.
- Exact behavior-vector support: `16--3,600` modes per prompt.
- Train row digest:
  `bcfa9accfa3b5c7dd312c85683157af070e7f2741385e9b5a882fd57da037cf7`.
- Evaluation row digest:
  `be0f621c5a0ae84ca45ef4ab866ae12f4c64472eeed1a0ae4666918ab794d183`.
- Train and evaluation input tuples are disjoint.
- Every row has two distinct seed programs certified through the external
  worker; these programs are generation audits and do not initialize the
  online support bank.

## Matched extension

- Model:
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Arms:
  - `grpo`: ordinary Dr.GRPO with passive verified-discovery tracking;
  - `online_canonical_haarnoja`: the E50 uncapped normalized controller.
- Seeds: `43,44,45`.
- Group size: `G=16`.
- Budget: exactly 50 complete 384-prompt passes.
- Learning rate: `2e-7`.
- One PPO epoch, `beta=0`, maximum norm `1`.
- Rollout temperature `1`, top-p `1`, maximum response length `192`.
- Prompt template `qwen_boxed`, verifier version `fast`, and evaluation split
  `multi_answer`.
- Evaluation at initialization and every quarter pass: greedy pass@1 plus four
  deterministic temperature-1 `K=8` draws using seeds `440100--440103`.
- Checkpoint and resume interval: one complete prompt-pool pass.
- Placement: node302 A100, 8 CPUs, 64 GiB, seven-day limit.

Run prefix:

- `pye50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1`.

## Objective and admission contract

The treatment is unchanged from the original E50 protocol:

- verified prompt-local growing support;
- entropy coefficient initialized and lower-bounded at `0.10`;
- no upper projection;
- normalized sensor `H(q_x) / log |B_x^+|`, skipped for singleton banks;
- target `0.80`;
- Adam controller learning rate `0.003`, betas `(0.9,0.999)`,
  epsilon `1e-8`, and EMA decay `0.90`;
- novelty coefficient `0.50`, pseudocount `1`, and surprisal clip `5`;
- updates apply to the next optimizer round.

The learner revalidates the exact sampled lambda. Admission, task reward, and
the canonical key all come from the external worker's single result. A
disagreement between actor reward and learner admission aborts the run.

The Dr.GRPO arm uses the same verifier and bank as a passive observer. Its
entropy and novelty coefficients are exactly zero, and discovery state has no
effect on task reward or policy loss.

## Reporting

Python is added to the live E50 monitor as a third row. It receives its own
paired common-horizon estimate and is never silently pooled with graph coloring
or Countdown. Until all three seeds in both arms reach an evaluation boundary,
only seed trajectories and scheduler state are shown; a heavy paired mean is
withheld.
