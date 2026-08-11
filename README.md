# ModeBench and Verified Replay

This repository studies correct-mode collapse in reinforcement learning with
verifiable rewards. The paper is **Mode Collapse under GRPO: ModeBench and
Verified Replay**:

- source: [`paper/main.tex`](paper/main.tex)
- built PDF: [`paper/main.pdf`](paper/main.pdf)
- evidence and build notes: [`paper/README.md`](paper/README.md)

The maintained method is verified replay: a prompt-local bank stores one
policy-generated exemplar for each observed validator-positive execution mode,
and a deterministic recurrent schedule rehearses those exemplars with uniform
teacher-forced likelihood. This replay likelihood is the only auxiliary
derivative. Passive bank bookkeeping and replay traversal remain enabled in the
compute-matched control,
whose applied replay derivative is exactly zero.

Published prior-stack endpoints are historical evidence only. They must not be
attributed to verified replay. The clean replay-only comparison is currently
running on the frozen eight-pass, half-pass-checkpoint protocol.

## ModeBench

ModeBench currently has five execution-bound domains:

| Domain | Validator | Canonical mode |
|---|---|---|
| Graph coloring | Checks fixed colors and every graph edge | Complete color vector |
| Countdown | Parses and exactly executes an operand-valid arithmetic AST | Normalized executed AST |
| Python factors | Calls a restricted lambda in an isolated interpreter | Returned integer vector |
| MathIR | Executes a finite equation-action menu | Exact state trajectory |
| PantryPlan | Solves the declared feasibility constraints | Ingredient support |

Correctness and identity always come from the same execution. Training support
is never initialized from a valid-answer catalogue.

The Python task is the third benchmark environment. Each prompt requests
`lambda n: EXPR` over four frozen inputs. The generated function must return a
nontrivial proper divisor on every external tool call. The deterministic
384/128 train/evaluation materialization has 16--3,600 exact modes per prompt
and two externally certified distinct programs per row.

```bash
var/seed_paper_eval/paper310/bin/python \
  ops/make_python_factor_mode_data.py \
  --output-root var/data/python_factor_modebench_v1

var/seed_paper_eval/paper310/bin/python \
  ops/verify_python_factor_mode.py \
  --candidate 'lambda n: 2 if n % 2 == 0 else 3' \
  --reference \
  '{"verifier":"python_factor_function","python_version":"factor-v1","cases":[6,10,15]}'
```

## Base model families

The training surface supports two instruction-tuned base-model families, so a
ModeBench result can be shown not to be an artifact of one tokenizer or one
instruction-tuning style:

| Family | Model | Chat surface |
|---|---|---|
| Qwen | `Qwen/Qwen2.5-0.5B-Instruct` | `<\|im_start\|>role` / `<\|im_end\|>` |
| Falcon | `tiiuae/Falcon3-1B-Instruct` | `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` |

Every prompt contract exists once per family as a `qwen_*`/`falcon_*` twin pair
(`*_boxed`, `*_graph_digits`, `*_countdown_digits`,
`*_pantry_support_mask`, `*_math`, `*_math_route`). The two members of a pair
share their system instruction and canonical answer rewrite verbatim and differ
only in role markers, which is what makes a cross-family comparison a clean
swap rather than a second prompt design. Selecting a family is a single
`OAT_ZERO_PROMPT_TEMPLATE` change; the objective, validator, canonical action
space, and reward path are surface-independent.

Three invariants are enforced rather than assumed, in
[`tests/test_falcon_prompt_surface.py`](tests/test_falcon_prompt_surface.py):

1. each Falcon prompt is a pure role-marker swap of its Qwen twin;
2. the Falcon surface reproduces `Falcon3-1B-Instruct`'s own published chat
   template byte for byte; and
3. the canonical action space resolves to the same horizon, sequence count,
   and maximum entropy under either tokenizer.

Argument validation keys off a template's *role* rather than its name, and a
canonical run whose materialized rows were rendered on the other family's
surface fails closed instead of training against role markers its base model
never saw.

## Evidence status

- **Running clean comparison:** Qwen2.5-0.5B-Instruct, all five domains,
  control versus verified replay, seeds 43--47, exactly eight passes.
- **Registered measurements:** pass 0 through pass 8 on the half-pass grid;
  pass 8 is the only primary endpoint.
- **Scientific difference:** the control performs the same bank, schedule,
  scoring, and backward traversal but applies an exact-zero replay derivative.
- **Historical program:** prior-stack results remain available as provenance
  but are not estimates of the replay-only method.

The frozen protocol is
[`paper/preregistration/e78_verified_replay_only_05b_20260804.md`](paper/preregistration/e78_verified_replay_only_05b_20260804.md),
and submitted jobs are bound in
[`var/artifacts/e78_verified_replay_only_05b_jobs.json`](var/artifacts/e78_verified_replay_only_05b_jobs.json).

## Main implementation surfaces

- [`src/oat_drgrpo/math_grader.py`](src/oat_drgrpo/math_grader.py) — unified
  ModeBench reward and canonical-key admission boundary.
- [`src/oat_drgrpo/templates.py`](src/oat_drgrpo/templates.py) — per-family
  chat surfaces, the `qwen_*`/`falcon_*` twin registry, template roles, and the
  fail-closed prompt materialization check.
- [`src/oat_drgrpo/canonical_actions.py`](src/oat_drgrpo/canonical_actions.py)
  — finite action grammars resolved into one-token, one-to-one tokenizer IDs.
- [`src/oat_drgrpo/online_canonical_bank.py`](src/oat_drgrpo/online_canonical_bank.py)
  — verified exemplar storage and deterministic replay scheduling state.
- [`src/oat_drgrpo/canonical_replay.py`](src/oat_drgrpo/canonical_replay.py)
  — uniform teacher-forced verified-replay objective.
- [`src/oat_drgrpo/python_modebench.py`](src/oat_drgrpo/python_modebench.py)
  — bounded Python syntax and factor-task contract.
- [`src/oat_drgrpo/python_modebench_process.py`](src/oat_drgrpo/python_modebench_process.py)
  and
  [`src/oat_drgrpo/python_modebench_worker.py`](src/oat_drgrpo/python_modebench_worker.py)
  — killable external execution boundary.
- [`ops/make_python_factor_mode_data.py`](ops/make_python_factor_mode_data.py)
  — deterministic third-domain materializer.
- [`ops/plot_modebench_paper.py`](ops/plot_modebench_paper.py) — frozen
  paired-common-horizon result builder.

## Build and validate

Cluster runs use the pinned environment at `var/seed_paper_eval/paper310`.
For repository-local commands:

```bash
source ops/repo_env.sh
export LD_LIBRARY_PATH="$PWD/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The expected stack is Python 3.10, PyTorch 2.6.0, Transformers 4.51.3,
vLLM 0.8.4, OAT 0.1.3.post1, and DeepSpeed 0.16.8.

Run the maintained validation surface:

```bash
make check
```

Build the paper:

```bash
make paper
```

The deterministic benchmark generators and operational workflow are
documented in [`ops/README.md`](ops/README.md). The broader objective and
historical evidence boundary are documented in
[`docs/maxent_program.md`](docs/maxent_program.md).
