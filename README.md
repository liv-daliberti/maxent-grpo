# MaxEnt-GRPO: Maximum‑Entropy Group‑Relative Policy Optimization

A clean, maximum‑entropy variant of GRPO for sequence‑level (candidate‑level) learning. We form a listwise target distribution over ranked candidates, blend it with a reference policy via a tempered geometric mean, and obtain a closed‑form per‑context optimizer with a soft improvement guarantee for a regularized potential.


## Quick Start

- Environment: `make conda-local && conda activate ./openr1`
- Install (core): `pip install -e .`
- Install (dev): `pip install -e .[dev]`
- Train (SLURM + vLLM): `./training.sh`
- Evaluation (quick): set `do_eval: true` in your recipe to run a fast subsample eval (see `src/grpo.py`).

Notes
- The one‑liner keeps conda/pip caches and the env under this repo (no writes to $HOME).
- Adjust the recipe via `CONFIG` in `training.sh` (default: `recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`).
- For LightEval benchmarks via vLLM/Slurm, see `src/utils/evaluation.py` (benchmarks list and launcher helper).


## Configuration
- All knobs live in the selected YAML recipe and are parsed by TRL (`GRPOScriptArguments`, `GRPOConfig`, `ModelConfig`). Start from:
  - `recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`
- Typical fields to adapt:
  - `model_name_or_path`, `model_revision`
  - `dataset_name` and prompt/solution columns
  - `use_vllm`, batch sizes, steps, KL settings
  - `output_dir`, `hub_model_id`, `push_to_hub`


## Repository Layout
- `src/` — core code: `grpo.py` (entrypoint), configs, utils, rewards
- `recipes/` — task/model YAMLs; `recipes/accelerate_configs/` for Accelerate/DeepSpeed
- `training.sh` — SLURM‑friendly launcher (vLLM + Accelerate)
- `environment.yml` — minimal conda spec (installs this package editable)
- `setup.py` — package metadata and dependencies


## Development
- Optional commit hooks (ruff + pylint + pytest):
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
