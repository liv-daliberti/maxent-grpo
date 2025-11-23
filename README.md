# MaxEnt-GRPO: Maximum‑Entropy Group‑Relative Policy Optimization

A clean, maximum‑entropy variant of GRPO for sequence‑level (candidate‑level) learning. We form a listwise target distribution over ranked candidates, blend it with a reference policy via a tempered geometric mean, and obtain a closed‑form per‑context optimizer with a soft improvement guarantee for a regularized potential.


## Quick Start

- Environment: `make conda-local && conda activate ./var/openr1` (wraps `configs/environment.yml`, pins conda/pip caches + TMPDIR + HF_HOME under `./var/`; if you prefer the raw command, export those env vars first, then `cd configs && conda env create -p ../var/openr1 -f environment.yml`)
- Install (core): `pip install -c configs/constraints.txt -e .`
- Install (dev): `pip install -c configs/constraints.txt -e .[dev]`
- Authenticate with Hugging Face (`huggingface-cli login` or `export HF_TOKEN=…`) so gated models/datasets can be pulled inside the Slurm job.
- Train on Slurm (recommended):  
  `sbatch ops/slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config math --accelerator zero3`
  - Use `--task maxent` to launch `src/maxent-grpo.py`.
  - Pass extra trainer flags via `--args "--run_name demo --report_to wandb"`.
  - Inspect `sbatch ops/slurm/train.slurm --help` for all knobs (dp/tp, ports, accelerator config).
- Training telemetry / proof of work: public run stats at [wandb.ai ↗](https://api.wandb.ai/links/ogd3-princeton-university/aw6ecc9b).
- Evaluation (quick): set `do_eval: true` in your recipe to run a fast subsample eval (see `src/grpo.py`).
- Inference: evaluate trained checkpoints on the `math_500` split locally via `src/inference`. Example:
  ```python
  from inference import InferenceModelSpec, run_math500_inference

  specs = [
      InferenceModelSpec(
          model_name_or_path="od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1",
          style="grpo",
          system_prompt="...",  # reuse the recipe prompt to match training
      ),
      InferenceModelSpec(
          model_name_or_path="od2961/Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-math-v1",
          style="maxent",
          system_prompt="...",
      ),
  ]
  results = run_math500_inference(specs)
  for res in results:
      print(res.label, res.accuracy)
  ```

Notes
- The one-liner keeps conda/pip caches and the env under `var/` in this repo (no writes to $HOME).
- Adjust the recipe via the `--config` flag (default: `configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`).
 - For LightEval benchmarks via vLLM/Slurm, see `src/core/evaluation.py` (benchmarks list and launcher helper).


## CLI Entry Points

Hydra CLI console scripts are generated on install:

- `maxent-grpo-baseline command=train-baseline ...`
- `maxent-grpo-maxent command=train-maxent ...`
- `maxent-grpo-generate command=generate ...`
- `maxent-grpo-inference command=inference ...`

Examples (override with Hydra-style dotted keys):

```bash
# Baseline GRPO with inline overrides
maxent-grpo-baseline command=train-baseline training.output_dir=var/data/out

# MaxEnt-GRPO using a YAML recipe
GRPO_RECIPE=configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml \
  maxent-grpo-maxent

# Generation
maxent-grpo-generate command=generate \
  generate.args.hf_dataset=open-r1/OpenR1-Math-220k \
  generate.args.model=Qwen/Qwen2.5-1.5B-Instruct

# Inference (math_500)
maxent-grpo-inference command=inference \
  inference.models='[ {model_name_or_path: od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1} ]'
```

The legacy Slurm launchers (`ops/slurm/train.slurm`, `ops/slurm/maxent-grpo.slurm`) remain available; consider migrating recipes to Hydra configs under `configs/recipes/`.

Hydra recipes:
- Baseline: `configs/recipes/hydra/baseline_math.yaml`
- MaxEnt-GRPO: `configs/recipes/hydra/maxent_math.yaml`

You can also set `GRPO_RECIPE=<path>` to point at any YAML recipe under `configs/recipes/...`; the Hydra CLI will load it directly.


## Environment Notes

- Transformers/TRL require `huggingface-hub < 1.0`. The launcher repins Hub into a compatible range and installs small CLI deps (`yq`, `rich`, `markdown-it-py`).
- vLLM requires a CUDA‑enabled PyTorch. The launcher provisions a matching CUDA stack inside the repo‑local env.


## Troubleshooting

- vLLM not healthy / empty server log:
- Check `var/logs/slurm_%j.out` for `nvidia-smi -L`. If it prints “No devices found”, update the SBATCH partition/GRES to match your site.
- “Invalid generic resource (gres) specification”:
  - Ensure your cluster supports `--gres=gpu:a100:7`. If not, change GRES to your GPU type or switch to `#SBATCH --gpus=7`.
- Transformers complaining about `huggingface-hub==1.1.1`:
  - The launcher repins Hub to `<1.0`. If you installed Hub 1.x outside the launcher, run: `pip install 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0'` inside the env.

MaxEnt‑GRPO
- Use `src/maxent-grpo.py` to run the maximum‑entropy variant that forms the per‑context target `π* ∝ q^{1/(τ+β)}·π_ref^{β/(τ+β)}` and trains via weighted MLE over K candidates.
- It reuses your existing GRPO recipe. Optional knobs via environment variables:
  - `MAXENT_TAU` (default 0.2) — sequence‑level entropy weight τ
  - `MAXENT_Q_TEMPERATURE` (default 1.0) — temperature when turning utilities into listwise q via softmax
  - `MAXENT_Q_EPS` (default 1e-6) — epsilon floor to ensure full support
  - `MAXENT_LENGTH_NORM_REF` ("1"/"0", default 1) — length‑normalize reference log‑probs
  - The trust‑region weight β is taken from `init_kl_coeff` in your recipe.


## Configuration
- All knobs live in the selected YAML recipe and are parsed by TRL (`GRPOScriptArguments`, `GRPOConfig`, `ModelConfig`). Start from:
  - `configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`
- Typical fields to adapt:
  - `model_name_or_path`, `model_revision`
  - `dataset_name` and prompt/solution columns
  - `use_vllm`, batch sizes, steps, KL settings
- `output_dir`, `hub_model_id`, `push_to_hub`

## Training Flow Overview
1. **Generation** – `CompletionGenerator` (local HF model or vLLM) produces grouped completions per prompt using the knobs in `GenerationSettings`.
2. **Rewards & Reference Scoring** – `training.pipeline.prepare_training_batch` computes reward statistics, gathers reference log-probs, and builds policy scores.
3. **Loss / Optimizer** – `training.weighting.loss` turns the sequence scores into weighted objectives; `training.loop` applies gradient accumulation, controllers, and optimizer/scheduler steps.
4. **Logging & Checkpointing** – Metrics flow through `training.metrics` while controller state + HF/DeepSpeed checkpoints are managed via `training.state` + `training.zero_utils`.

```
Prompts/Dataloader
       │
       ▼
Generation (CompletionGenerator)
       │
       ▼
Reward & Reference Prep (pipeline.prepare_training_batch)
       │
       ▼
Loss / Optimizer (training.weighting.loss + training.loop)
       │
       ├──► Logging (training.metrics)
       └──► Checkpoints (training.state/zero_utils)
```

See ``docs/architecture`` for a more detailed breakdown and module references.


## Repository Layout
```
.
├─ configs/                     # repo-wide config + recipes (constraints, env, pytest, pyright, YAMLs)
│  └─ recipes/                  # task/model YAMLs + accelerate configs
├─ src/                         # core package
│  ├─ grpo.py                   # GRPO entrypoint
│  ├─ maxent-grpo.py            # MaxEnt‑GRPO entrypoint
│  ├─ maxent_grpo/config/       # configuration helpers
│  ├─ utils/                    # utilities
│  └─ rewards.py                # reward shaping / scoring
├─ ops/                         # execution tooling (slurm/, scripts/, tools/, sitecustomize.py)
│  ├─ slurm/                    # cluster launchers
│  ├─ scripts/                  # automation helpers (env bootstrap, dataset prep, etc.)
│  ├─ tools/                    # local shell helpers (PATH management, etc.)
│  └─ sitecustomize.py          # repo-local sys.path/test shims
├─ var/                         # runtime artifacts (openr1 env, logs, caches, Sphinx build, datasets)
│  ├─ openr1/                   # repo-local conda env
│  ├─ logs/                     # Slurm/stdout, training + vLLM logs
│  ├─ data/                     # training/eval outputs and checkpoints
│  └─ _build/                   # Sphinx HTML
├─ docs/                        # Sphinx sources
└─ setup.py                     # package metadata (install at repo root)
```


## Development
- Install dev tooling (ruff, pylint, pytest, etc.) with `pip install -r configs/requirements-dev.txt` (the file already enforces `configs/constraints.txt`).
- Run `pytest -q -c configs/pytest.ini` (or `make test`) so the relocated config is picked up.
- Run `pyright --project configs/pyrightconfig.json` for static type checking.
- Optional commit hooks (ruff + pylint + pytest + sphinx docs):
  - `pre-commit install`
  - Run on demand: `pre-commit run -a`


## Documentation
- Online: https://maxent-grpo.readthedocs.io/en/latest/
- Local build: `pip install -c configs/constraints.txt -r docs/requirements.txt && sphinx-build -b html docs var/docs/_build/html`


## Citation
If you use this work, please cite: “MaxEnt‑GRPO: Maximum‑Entropy Group‑Relative Policy Optimization (2025).”

BibTeX
```
@misc{MaxEntGRPO2025,
  title        = {MaxEnt-GRPO: Maximum-Entropy Group-Relative Policy Optimization},
  author       = {Liv d'Aliberti},
  year         = {2025},
  publisher    = {GitHub},
  note         = {Code: this repository}
}
```


## License
Apache 2.0 — see `LICENSE`.
