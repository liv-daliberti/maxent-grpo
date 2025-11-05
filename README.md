# MaxEnt-GRPO: Maximum-Entropy Group-Relative Policy Optimization

This repo contains a maximum-entropy variant of Group-Relative Policy Optimization (GRPO) at the sequence (candidate) level. We work with a listwise target distribution over ranked candidates and a reference policy, derive a closed-form per-context optimizer that blends the two via a tempered geometric mean, and show a soft improvement guarantee for a regularized potential.



## Code Layout
- `src/`: core training code (entrypoint `grpo.py`, utilities, rewards)
- `recipes/`: model- and task-specific YAML configs
- `recipes/accelerate_configs/`: DeepSpeed / Accelerate configs
- `training.sh`: launcher script
- `environment.yml`: minimal Conda environment
- `setup.py`: package metadata and dependencies



## Quickstart

### 1) Environment (local-only installs)
- Recommended: `scripts/bootstrap_env.sh` ensures Conda env, pip cache, and temp dirs live under the repo.
  - Run: `bash scripts/bootstrap_env.sh` then `conda activate ./openr1`
- If you prefer manual steps, export these before creating the env so pip/conda stay local:
  - `export CONDARC=$PWD/.condarc`
  - `export CONDA_PKGS_DIRS=$PWD/.conda_pkgs CONDA_ENVS_DIRS=$PWD/.conda_envs`
  - `export PIP_CACHE_DIR=$PWD/.pip_cache PIP_CONFIG_FILE=$PWD/.pip/pip.conf`
  - `export TMPDIR=$PWD/.tmp`
  - Then: `conda env create -p $PWD/openr1 -f environment.yml && conda activate $PWD/openr1`
- Install PyTorch (pick one):
  - GPU (Conda): `conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.4`
  - GPU (pip, CUDA 12.4): `pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0`
  - CPU only: `pip install torch==2.6.0`


### 2) Install the package
- Standard: `pip install -e .`
- Dev setup: `pip install -e .[dev]` (includes pytest, pylint, Sphinx)
- Local-only install under repo (no user dirs):
  - With local user base: `make install-local` then add `export PATH="$PWD/.local/bin:$PATH"`
  - With local venv: `make install-venv` then `source .venv/bin/activate`
  - Auto-append PATH to your shell rc: `make ensure-path`

### 2.1) Commit hooks (ruff + pylint + pytest)
- Enable hooks: `pre-commit install`
- Run manually on all files: `pre-commit run -a`
- Hooks run: `ruff check .`, `pylint` (via `.pylintrc`) on `src/`, and `pytest -q`.


### 3) Configure and train
- Pick a recipe (e.g., `recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`) and an Accelerate config (e.g., `recipes/accelerate_configs/zero3.yaml`).
- Launch training: `./training.sh`
- To change the model, dataset, or hyperparameters, edit the selected recipe YAML.

### Eval Sampler
- If `do_eval: true` and a test/validation split is available, the trainer evaluates on a subsample of the eval split for speed: 10% of examples up to 1,000 (min 1), shuffled with the run seed.
- This behavior is in `src/grpo.py` and does not override any other YAML parameters.


## Documentation
- Online docs: https://maxent-grpo.readthedocs.io/en/latest/
- Build locally: `pip install -r docs/requirements.txt && sphinx-build -b html docs _build/html`
- Read the Docs is configured via `.readthedocs.yaml` (Sphinx autodoc with heavy deps mocked).
- Make convenience: `make docs` (see `Makefile` for more targets)

## Recipes → Parameters
- All training/runtime parameters come from the selected recipe YAML via TRL’s parser (`GRPOScriptArguments`, `GRPOConfig`, `ModelConfig`).
- The entrypoint `src/grpo.py` does not hardcode training knobs; it only:
  - builds prompts from `dataset_prompt_column` with the optional `system_prompt`,
  - ensures PAD token behavior is sane for causal LMs,
  - enables `return_reward` if it isn’t already set (so rewards are logged/available),
  - applies the small eval subsampler described above.


## Citation (informal)
If you use this code or ideas, please cite as “MaxEnt-GRPO: Maximum-Entropy Group-Relative Policy Optimization (2025)”.

## BibTeX
Author: Liv d'Aliberti.

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
Apache License 2.0. See `LICENSE` for the full text.
