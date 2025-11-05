# MaxEnt-GRPO: Maximum‑Entropy Group‑Relative Policy Optimization

A clean, maximum‑entropy variant of GRPO for sequence‑level (candidate‑level) learning. We form a listwise target distribution over ranked candidates, blend it with a reference policy via a tempered geometric mean, and obtain a closed‑form per‑context optimizer with a soft improvement guarantee for a regularized potential.


## Quick Start

- Environment: `make conda-local && conda activate ./openr1`
- Install (core): `pip install -e .`
- Install (dev): `pip install -e .[dev]`
- Train on Slurm (recommended): `bash scripts/sbatch_training.sh`
  - Pass extra sbatch args as needed, e.g. `bash scripts/sbatch_training.sh --time=04:00:00`
- Evaluation (quick): set `do_eval: true` in your recipe to run a fast subsample eval (see `src/grpo.py`).

Notes
- The one‑liner keeps conda/pip caches and the env under this repo (no writes to $HOME).
- Adjust the recipe via `CONFIG` in `training.sh` (default: `recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`).
- For LightEval benchmarks via vLLM/Slurm, see `src/utils/evaluation.py` (benchmarks list and launcher helper).


## Slurm Usage & Logging

- Submit via `scripts/sbatch_training.sh`. This wrapper:
  - Ensures `logs/` exists before submission.
  - Sets `--chdir` to the repo so relative paths resolve.
  - Forces Slurm stdout/err to `logs/slurm_%j.out` and `logs/slurm_%j.err`.
- All application logs are anchored to the repo at `logs/`:
  - vLLM server: `logs/liv_vllm_<RUN>_<TIMESTAMP>.log`
  - Trainer:     `logs/liv_train_<RUN>_<TIMESTAMP>.log`
- The training script runs vLLM on the first allocated GPU and trains on the remaining GPUs. No nested `srun` is used to avoid “step creation temporarily disabled” throttling on busy clusters.


## Environment Notes

- Transformers/TRL require `huggingface-hub < 1.0`. The launcher repins Hub into a compatible range and installs small CLI deps (`yq`, `rich`, `markdown-it-py`).
- vLLM requires a CUDA‑enabled PyTorch. The launcher checks `torch.cuda.is_available()` and fails fast if CUDA is not available. To fix the env on a login/interactive node:
  1) `conda activate /path/to/repo/openr1`
  2) Remove conflicting CPU torch/triton: `pip uninstall -y torch triton torchtriton pytorch-triton || true` and `conda remove -y pytorch pytorch-cuda torchtriton triton || true`
  3) Clean caches: `conda clean -a -y`
  4) Install CUDA wheels for the pinned Torch: `pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0'`
  5) Verify: `python -c "import torch; print('cuda?', torch.cuda.is_available(), 'n=', torch.cuda.device_count())"`


## Troubleshooting

- Slurm logs not under `logs/`:
  - Always submit with `bash scripts/sbatch_training.sh`. Submitting `training.sh` directly may cause Slurm to create `slurm-%j.out` in the submit CWD if `logs/` didn’t exist at submission.

- “step creation temporarily disabled” from `srun`:
  - The script does not use nested `srun`; everything runs in the main batch step. If you re‑enable `srun`, you may hit throttling on busy nodes.

- vLLM “Device string must not be empty” or “UnspecifiedPlatform”:
  - Ensure the CUDA PyTorch build is installed (see Environment Notes) and that `CUDA_VISIBLE_DEVICES` is set by Slurm. The launcher binds vLLM to the first GPU in the job’s mapping and training to the remainder.

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
├─ training.sh                  # SLURM‑friendly launcher (vLLM + Accelerate)
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
