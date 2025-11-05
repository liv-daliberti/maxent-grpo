# MaxEnt-GRPO: Maximum-Entropy Group-Relative Policy Optimization

This repo contains a maximum-entropy variant of Group-Relative Policy Optimization (GRPO) at the sequence (candidate) level. We work with a listwise target distribution over ranked candidates and a reference policy, derive a closed-form per-context optimizer that blends the two via a tempered geometric mean, and show a soft improvement guarantee for a regularized potential.

- Paper title: MaxEnt-GRPO: Maximum-Entropy Group-Relative Policy Optimization
- Focus: Concave, per-context sequence-level program with an explicit Shannon-entropy term and reverse-KL trust region; closed-form optimizer and soft improvement lemma.


## Abstract (high-level)
We study a maximum-entropy GRPO at the sequence level. Given a listwise target distribution \(q(\cdot\mid x)\) over \(K\) ranked candidates and a reference policy \(\pi_{\mathrm{ref}}(\cdot\mid x)\), we propose the per-context concave program that adds Shannon entropy \(H(\pi)\) to a reverse-KL trust region and show it admits a closed-form maximizer
\[\pi^\* \propto q^{1/(\tau+\beta)}\,\pi_{\mathrm{ref}}^{\beta/(\tau+\beta)}.\]
This clarifies the geometry of GRPO-style updates, introduces a principled exploration knob \(\tau\), and yields a target distribution over candidates that can be projected to token-level training via weighted MLE. We prove a soft improvement lemma and situate MaxEnt-GRPO with respect to KL-regularized and maximum-entropy RL.


## Objective and Closed-Form Solution
For \(\tau\ge 0\) and \(\beta\ge 0\) (with \(\tau+\beta>0\)), consider the per-context concave program
\[\max_{\pi\in\Delta^K}\; \mathbb{E}_{a\sim\pi}[\log q(a\mid x)] + \tau\,H(\pi) - \beta\,\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}}).\]
Under full support, the unique maximizer is
\[\boxed{\;\pi^\*(a_i\mid x)\propto q_i(x)^{\frac{1}{\tau+\beta}}\,\pi_{\mathrm{ref},i}(x)^{\frac{\beta}{\tau+\beta}}\;}\quad (i=1,\dots,K).\]
Interpretation: a tempered geometric mean of \(q\) and \(\pi_{\mathrm{ref}}\) with temperature \((\tau+\beta)^{-1}\). Limits: \(\tau\downarrow 0\) recovers GRPO-like (no sequence-level entropy); \(\beta\downarrow 0\) yields pure max-entropy fit to \(q\).


## Soft Improvement Lemma (sketch)
Define the regularized potential \(\Phi_\tau(\pi) = \mathbb{E}_x\big[\mathbb{E}_\pi[\log q]+\tau H(\pi)\big]\). The one-step update
\[\pi_{t+1} = \arg\max_\pi\; \mathbb{E}_\pi[\log q]+\tau H(\pi)-\beta\,\mathrm{KL}(\pi\,\|\,\pi_t)\]
satisfies
\[\Phi_\tau(\pi_{t+1})-\Phi_\tau(\pi_t)\;\ge\; \beta\,\mathbb{E}_x\big[\mathrm{KL}(\pi_{t+1}\,\|\,\pi_t)\big]\;\ge\;0.\]
Thus, for \(\beta>0\), the potential strictly improves unless stationary.


## Code Layout
- Source: `src/open_r1` (training loops, utils, GRPO variants)
- Recipes: `recipes/*` (model- and task-specific YAML configs)
- Accelerate: `recipes/accelerate_configs/*` (DeepSpeed / process configs)
- Launcher: `training.sh` (wrapper) → `training-math-grpo.sh` (SLURM + vLLM + Accelerate)


## Quickstart

### 1) Create Conda env
- Create and activate:
  - `conda env create -f environment.yml`
  - `conda activate openr1`

- Install PyTorch (choose one):
  - GPU (Conda, recommended):
    - `conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.4`
  - GPU (pip, CUDA 12.4 wheels):
    - `pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0`
  - CPU only:
    - `pip install torch==2.6.0`

- Optional developer tools and extras:
  - `pip install -e '.[dev]'`  (ruff, isort, pytest, eval extras, etc.)


### 2) Install the package
- Already installed in editable mode via `environment.yml` (if you created it via the file). If you skipped that, run:
  - `pip install -e .`


### 3) Configure and Train
- Choose a recipe under `recipes/` (e.g., `recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml`) and an Accelerate config (e.g., `recipes/accelerate_configs/zero3.yaml`).
- Launch training:
  - `./training.sh`

Notes:
- The default script spawns a vLLM server on one GPU and runs training on the remaining GPUs using Accelerate+DeepSpeed. It logs to `logs/` and uses caches under local `.cache/` by default.
- To change the model, dataset, or hyperparameters, edit the `CONFIG` and `CONFIG_FILE` variables inside `training-math-grpo.sh` or point them to other YAMLs under `recipes/`.
- Ensure you have access to Hugging Face models (login may be required: `huggingface-cli login`).

### Eval Sampler
- If `do_eval: true` and a test/validation split is available, the trainer evaluates on a subsample of the eval split for speed: 10% of examples up to 1,000 (min 1), shuffled with the run seed.
- This behavior is in `src/open_r1/grpo.py` and does not override any other YAML parameters.


## Environment Details
- Minimal Conda env is defined in `environment.yml` and installs this repo in editable mode. It omits `torch` so you can choose the correct GPU/CPU wheels for your system.
- Core runtime deps (pinned in `setup.py`): `accelerate`, `deepspeed`, `transformers`, `trl[vllm]`, `datasets`, `bitsandbytes`, `einops`, `safetensors`, `sentencepiece`, `hf_transfer`, `wandb`, etc.
- Extras available: `[dev]`, `[eval]`, `[code]`, `[quality]`, `[tests]`, `[torch]`.

## Hugging Face Hub (push/pull)
- Pull datasets/models: configured via `datasets.load_dataset` and `transformers.from_pretrained`. Login if needed: `huggingface-cli login`.
- Push models: set in your recipe YAML (used directly by the trainer)
  - `push_to_hub: true`
  - `hub_model_id: <your-namespace/your-repo>`
  - Optional branch/tagging: `hub_model_revision`, `hub_strategy`
- The training script will call `trainer.push_to_hub(...)` when `push_to_hub: true`.

## Recipes → Parameters
- All training/runtime parameters come from the selected recipe YAML via TRL’s parser (`GRPOScriptArguments`, `GRPOConfig`, `ModelConfig`).
- The entrypoint `src/open_r1/grpo.py` does not hardcode training knobs; it only:
  - builds prompts from `dataset_prompt_column` with the optional `system_prompt`,
  - ensures PAD token behavior is sane for causal LMs,
  - enables `return_reward` if it isn’t already set (so rewards are logged/available),
  - applies the small eval subsampler described above.


## Citation (informal)
If you use this code or ideas, please cite as “MaxEnt-GRPO: Maximum-Entropy Group-Relative Policy Optimization (2025)”.


## License
Apache 2.0. See headers and `setup.py` for details.
