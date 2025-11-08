# MaxEnt-GRPO: Maximum‑Entropy Group‑Relative Policy Optimization

A clean, maximum‑entropy variant of GRPO for sequence‑level (candidate‑level) learning. We form a listwise target distribution over ranked candidates, blend it with a reference policy via a tempered geometric mean, and obtain a closed‑form per‑context optimizer with a soft improvement guarantee for a regularized potential.


## Quick Start

- Environment: `make conda-local && conda activate ./openr1`
- Install (core): `pip install -e .`
- Install (dev): `pip install -e .[dev]`
- Authenticate with Hugging Face (`huggingface-cli login` or `export HF_TOKEN=…`) so gated models/datasets can be pulled inside the Slurm job.
- Train on Slurm (recommended):  
  `sbatch slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config math --accelerator zero3`
  - Use `--task maxent` to launch `src/maxent-grpo.py`.
  - Pass extra trainer flags via `--args "--run_name demo --report_to wandb"`.
  - Inspect `sbatch slurm/train.slurm --help` for all knobs (dp/tp, ports, accelerator config).
- Training telemetry / proof of work: public run stats at [wandb.ai ↗](https://api.wandb.ai/links/ogd3-princeton-university/aw6ecc9b).
- Evaluation (quick): set `do_eval: true` in your recipe to run a fast subsample eval (see `src/grpo.py`).

Notes
- The one‑liner keeps conda/pip caches and the env under this repo (no writes to $HOME).
- Adjust the recipe via the `--config` flag (default: `recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`).
- For LightEval benchmarks via vLLM/Slurm, see `src/utils/evaluation.py` (benchmarks list and launcher helper).


## Training via Slurm Launcher

- Entry point: `slurm/train.slurm` (multi‑node aware, optional dedicated vLLM head node).
- Minimal command:

  ```bash
  sbatch slurm/train.slurm \
    --model Qwen2.5-1.5B-Instruct \
    --task grpo \
    --config math \
    --accelerator zero3 \
    --args "--run_name math-demo --report_to wandb"
  ```

- Key arguments:
  - `--model` maps to `recipes/<model>/…` (e.g., `Qwen2.5-1.5B-Instruct`).
  - `--task` selects `src/grpo.py` (`grpo`) or `src/maxent-grpo.py` (`maxent`).
  - `--config` picks `recipes/<model>/<task>/config_<name>.yaml`.
  - `--accelerator` selects an Accelerate/DeepSpeed config from `recipes/accelerate_configs/`.
  - `--dp/--tp`, `--vllm-port`, `--vllm-group-port` tune vLLM server topology.
  - `--args` passes raw flags to the training script (quoted string).
- What the launcher does:
  - Boots your conda or venv (configurable via `ENV_MODE`, `CONDA_ENV`, `ENV_ACTIVATE`).
  - Pins CUDA/toolkit modules and redirects all HF/pip caches into the repo.
  - Parses your recipe to detect `use_vllm`; if enabled it either dedicates GPU 0 (single node) or a full node (multi‑node) to `trl.scripts.vllm_serve`.
  - Derives Accelerate world size from `#SBATCH --nodes/--gres`, including inline vLLM layouts (training uses remaining GPUs).
  - Injects vLLM server host/port back into the trainer unless you override it.
- Logs in `logs/`:
  - Slurm stdout/err: `logs/<job-name>-<jobid>.out|err` (defaults to `open_r1`).
  - Training stream: `logs/train_<jobid>.log`.
  - vLLM server: `logs/vllm-<jobid>.out` (or inline server tails in the Slurm log).
- Prompts longer than 2,048 characters are automatically truncated before sending requests to vLLM; override via `MAX_PROMPT_CHARS` if you need a different limit.
- Customise the SBATCH header at the top of `slurm/train.slurm` for your cluster (account, partition, GPU type/count, walltime).
- Run `sbatch slurm/train.slurm --help` to see every flag and the expected directory layout.


## Environment Notes

- Transformers/TRL require `huggingface-hub < 1.0`. The launcher repins Hub into a compatible range and installs small CLI deps (`yq`, `rich`, `markdown-it-py`).
- vLLM requires a CUDA‑enabled PyTorch. The launcher provisions a matching CUDA stack inside the repo‑local env.


## Troubleshooting

- vLLM not healthy / empty server log:
  - Check `logs/slurm_%j.out` for `nvidia-smi -L`. If it prints “No devices found”, update the SBATCH partition/GRES to match your site.
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
  - `recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`
- Typical fields to adapt:
  - `model_name_or_path`, `model_revision`
  - `dataset_name` and prompt/solution columns
  - `use_vllm`, batch sizes, steps, KL settings
  - `output_dir`, `hub_model_id`, `push_to_hub`


## Repository Layout
```
.
├─ src/                         # core code
│  ├─ grpo.py                   # GRPO entrypoint
│  ├─ maxent-grpo.py            # MaxEnt‑GRPO entrypoint
│  ├─ configs.py                # configuration helpers
│  ├─ utils/                    # utilities
│  └─ rewards.py                # reward shaping / scoring
├─ recipes/                     # task/model YAMLs
│  └─ accelerate_configs/       # Accelerate/DeepSpeed configs
├─ slurm/
│  └─ train.slurm               # multi-node SLURM launcher (env bootstrap + vLLM + Accelerate)
├─ environment.yml              # minimal conda spec (installs this package editable)
└─ setup.py                     # package metadata and dependencies
```


## Development
- Optional commit hooks (ruff + pylint + pytest + sphinx docs):
  - `pre-commit install`
  - Run on demand: `pre-commit run -a`


## Documentation
- Online: https://maxent-grpo.readthedocs.io/en/latest/
- Local build: `pip install -r docs/requirements.txt && sphinx-build -b html docs _build/html`


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
