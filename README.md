# Group-Level Exploration in GRPO (xDr.GRPO)

This repository holds the matched comparative for the paper under `paper/`
(*Exploration Should Act at the Group Level in RL LLM Fine-Tuning*): Dr.GRPO,
xDr.GRPO (candidate-level tempered aggregation), SEED-Dr.GRPO, and
Token-MaxEnt Dr.GRPO, trained and evaluated on the exact multi-answer
ModeBench pair (Countdown arithmetic and graph-coloring completion).

The live code is under `src/oat_drgrpo/`:

- `train_zero_math.py`: the patched OAT learner
- `learner/`: Dr.GRPO and Dr.X learner mixins split out from the
  original monolithic trainer
- `listwise.py`: prompt-group compatibility facade for listwise helpers
- `drx_targets.py`, `semantic_remix.py`, `semantic_utility.py`: Dr.X target
  construction and semantic utility/remix helpers
- `stats_utils.py`, `logging_utils.py`, `runtime.py`, `templates.py`: shared
  runtime, metrics, and prompt-format support
- `math_grader.py`: the verifiable-math reward/grader

The active launch surface is:

- `ops/submit_countdown_comparative.sh`: the paper's matched comparative
  (Dr.GRPO vs xDr.GRPO tau sweep plus the Token-MaxEnt control) on the
  ModeBench data, with `ops/run_countdown_comparative_eval.sh` and
  `ops/analyze_countdown_comparative.py` for the seed-matched evaluation and
  prompt-clustered regression analysis
- `ops/run_oat_zero_exact_1p5b_upstream.sh`
- `ops/run_oat_zero_exact_drx_1p5b_upstream.sh`
- `ops/run_oat_zero_tiny_probe.sh`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_exact_drx_node302.slurm`
- `ops/slurm/train_tiny_probe_node302.slurm`

See:

- `docs/drgrpo_vs_drx.md` for the comparison contract.
- `ops/README.md` for the active launcher/evaluation scripts.

Training and evaluation use the exact multi-answer ModeBench pair — Countdown
arithmetic and graph-coloring completion — generated deterministically on
first use into:

- `var/data/exact_countdown_easy3_probe` (via `ops/make_exact_countdown_mode_data.py`)
- `var/data/exact_answer_mode_probe` (via `ops/make_exact_answer_mode_data.py`)

Select the domain with `OAT_ZERO_TASK=countdown|graph_coloring` (default
`countdown`).

## Paper

The NeurIPS-format draft of the xDr.GRPO paper (*Exploration Should Act at
the Group Level in RL LLM Fine-Tuning*) lives under `paper/`:

- `paper/main.tex` — the paper source
- `paper/example_paper.bib` — bibliography
- `paper/Makefile` — build (`make` in `paper/` produces `main.pdf`)

See `paper/README.md` for build requirements and provenance notes.

## Environment Setup

There are two supported ways to use this repo.

### 1. Canonical training runtime

The launchers are written around the canonical `paper310` environment:

- Python: `var/seed_paper_eval/paper310/bin/python`
- Source root: `src/`
- Trainer module: `oat_drgrpo.train_zero_math`

The exact launcher validates the key runtime versions before starting training:

- Python `3.10.x`
- `torch==2.6.0`
- `transformers==4.51.3`
- `vllm==0.8.4`
- `oat-llm==0.1.3.post1`
- `deepspeed==0.16.8`
- `math-verify==0.7.0`
- `fire==0.7.0`

The Slurm scripts default to that environment automatically. If you want to use a different interpreter, override:

```bash
export OAT_ZERO_PYTHON=/path/to/python
export OAT_ZERO_PYTHON_LIB_DIR=/path/to/python/lib
```

### 2. Local development environment

If you just want to edit code, import modules, and run smoke checks locally:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

This gives you the repo package from `src/oat_drgrpo/`. For real training, you will still want the canonical runtime above unless you intentionally reproduce all training dependencies yourself.

### Repo environment variables

Before running ad hoc commands, source the repo environment helper:

```bash
source ops/repo_env.sh
```

This keeps caches and runtime state under `var/` instead of spilling into home-directory defaults:

- Hugging Face caches under `var/cache/huggingface`
- pip cache under `var/cache/pip`
- W&B state under `var/wandb`
- temporary files under `var/tmp`

## Data Layout

The kept launchers generate and consume the ModeBench datasets under:

```text
var/data/
  exact_countdown_easy3_probe/
    train/
    eval/
  exact_answer_mode_probe/
    train/
    eval/
```

Each root is a HuggingFace `DatasetDict` written by the corresponding
`ops/make_exact_*_data.py` generator (regenerated automatically by the
launchers when missing; generation is deterministic given the seed). The
training prompt column is `problem`; the `answer` column holds the JSON
verifier spec (task, numbers/graph, target, and the prompt's number of valid
answer modes), which the grader in `src/oat_drgrpo/math_grader.py` consumes
directly.

## Running Jobs

Baseline Dr.GRPO:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm
```

Dr.X-GRPO:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_exact_drx_node302.slurm
```

The launcher logs the resolved config at startup, including:

- Python path
- source root
- prompt/eval datasets
- batch geometry
- Dr.X-GRPO knobs such as semantic token advantage source, token length normalizer, and semantic clustering method

Logs are written to:

- `var/artifacts/logs/*.out`
- `var/artifacts/logs/*.err`

## Quick Checks

Minimal syntax/import checks:

```bash
source ops/repo_env.sh
python -m py_compile src/oat_drgrpo/train_zero_math.py src/oat_drgrpo/listwise.py src/oat_drgrpo/math_grader.py
python -m ruff check src tests
bash -n ops/run_oat_zero_exact_1p5b_upstream.sh ops/run_oat_zero_exact_drx_1p5b_upstream.sh
```

If you want to sanity-check the exact launcher without submitting a long job, inspect the startup banner in the Slurm log. The exact launcher also bootstraps `flash-attn==2.7.4.post1` on-node when needed.
