# ModeBench and Verified Replay

This repository studies correct-mode collapse in reinforcement learning with
verifiable rewards. The paper is **There’s More Than One Way: Mode Collapse in RLVR & ModeBench**:

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
attributed to verified replay. The clean replay comparison is complete on the frozen eight-pass,
half-pass-checkpoint protocol. Its three-scale analysis contains 74 admissible
seed pairs; one conflicting Falcon Countdown endpoint is excluded. ReplayMaxRL
and the harder Level-2 tasks extend that comparison.

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

The main version for new Level 3 work is the [neutral-Python revision](docs/modebench_current_prompts.md): Python uses `python_level3_neutral_v1`, while Countdown, Graph Coloring, MathIR and Pantry retain their admitted V3 data and wording. Fresh neutral-dataset confirmation passed on September 12 (22.34% pass@1, 70.51% pass@8), and the queued Python jobs use CLI-tested immutable runtimes. New native training configurations should use `neutral_python_training_environment` from `ops/modebench_current_training.py` to bind the admitted data and runtime together. Historical results retain their original dataset and prompt identities.

The [Level 4 / Qwen-7B and Level 5 / Qwen-14B guide](docs/modebench_scale_calibration.md)
describes calibration, admission requirements, and loading the train/test splits.

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

The [September 11 completion report](paper/results/completed_3b_and_level2_20260911.md)
now presents all five finished 3B MaxRL/ReplayMaxRL comparisons. The
[audited census](paper/results/latest_results_20260911.json) admits exact
pass-8 endpoints with all four registered evaluation draws.

| Evidence | Admitted terminal cells | Complete paired blocks |
|---|---:|---:|
| E118 MaxRL / ReplayMaxRL | 150/150 | 15/15 |
| E119 Level-2 four-method factorial | 90/100 | 4/5 |
| E120 uniform versus frequency-weighted replay | 44/45 | 8/9 |

**3B is complete:** all 50 endpoints and 25 seed pairs across the five
domains. ReplayMaxRL has higher mean pass@8 and distinct@8 in every domain;
Graph and Python correctness intervals remain inconclusive. Level 2 has
complete Graph, Countdown, Python, and MathIR factorials (80/80 endpoints);
Pantry remains at 10/20. The original ReplayDr.GRPO comparison retains
14 complete five-seed blocks plus Falcon Countdown at four admissible seeds.
Partial blocks retain exact seed counts without five-seed intervals.

The control and replay arms share bank bookkeeping, recurrent traversal,
scoring, and backward traversal; the control applies an exact-zero replay
derivative. Historical multi-component results remain provenance rather than
estimates of this intervention. The original protocol is
[`paper/preregistration/e78_verified_replay_only_05b_20260804.md`](paper/preregistration/e78_verified_replay_only_05b_20260804.md).

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
