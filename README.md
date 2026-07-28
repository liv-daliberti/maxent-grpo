# ModeBench and Online Verified MaxEnt

This repository studies solution-mode discovery in reinforcement learning with
verifiable rewards. The paper is **ModeBench: Executable Outcome Discovery for
Maximum-Entropy Reasoning**:

- source: [`paper/main.tex`](paper/main.tex)
- built PDF: [`paper/main.pdf`](paper/main.pdf)
- evidence and build notes: [`paper/README.md`](paper/README.md)

The final method maintains a prompt-local bank containing only outcomes that
the current policy generated and the task validator accepted. It adds:

1. a leave-one-out entropy advantage over verified canonical outcomes;
2. one total bonus for each newly discovered mode;
3. a controller over entropy normalized by discovered support size; and
4. identical passive discovery telemetry in the reward-only control.

The long-horizon comparison uses a lower-bounded but upper-unprojected
coefficient. Earlier sequence-entropy, candidate-projection, semantic-shaping,
latent-option, finite-action, and strategy-canonicalization experiments remain
in the repository as the design evidence behind this choice; they are not
pooled into the final treatment estimate.

## ModeBench

ModeBench currently has three execution-bound domains:

| Domain | Validator | Canonical mode |
|---|---|---|
| Graph coloring | Checks fixed colors and every graph edge | Complete color vector |
| Countdown | Parses and exactly executes an operand-valid arithmetic AST | Normalized executed AST |
| Python factors | Calls a restricted lambda in an isolated interpreter | Returned integer vector |

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

## Evidence status

- **Original active cohort:** fresh matched Dr.GRPO and online verified MaxEnt
  on graph coloring and Countdown, paired at seeds 43--45 with a fixed 50-pass
  budget.
- **Frozen interim snapshot:** graph coloring at 8.25 common passes and
  Countdown at 2.75 common passes. This is explicitly not the terminal result.
- **Python extension:** materialized, externally audited, and separately
  frozen for matched Dr.GRPO/online verified MaxEnt runs at seeds 43--45. Its
  later launch provenance is retained explicitly.
- **MATH realism track:** a separate external-validity experiment trains only
  on a frozen MATH12K subset and evaluates on disjoint MATH-500. It is not
  counted as another multi-mode ModeBench domain because final-answer grading
  does not certify distinct reasoning strategies.
- **Historical program:** retained as mechanism and provenance evidence, not
  independent replications of the final method.

Machine-readable interim results live in
[`paper/results/modebench_long_horizon_interim.json`](paper/results/modebench_long_horizon_interim.json).
The leakage-free MATH transfer contract is frozen in
[`paper/preregistration/e64_math500_realism_transfer_05b.md`](paper/preregistration/e64_math500_realism_transfer_05b.md).

## Main implementation surfaces

- [`src/oat_drgrpo/math_grader.py`](src/oat_drgrpo/math_grader.py) — unified
  ModeBench reward and canonical-key admission boundary.
- [`src/oat_drgrpo/online_canonical_bank.py`](src/oat_drgrpo/online_canonical_bank.py)
  — verified growing-support state and canonical advantages.
- [`src/oat_drgrpo/online_canonical_controller.py`](src/oat_drgrpo/online_canonical_controller.py)
  — support-normalized Haarnoja-style control.
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
