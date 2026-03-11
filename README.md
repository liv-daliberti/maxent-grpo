# MaxEnt-GRPO: Maximum‑Entropy Group‑Relative Policy Optimization

A clean GRPO training stack that supports vanilla GRPO, GRPO + entropy bonus, and maximum‑entropy weighting for sequence‑level learning. Hydra console scripts, TRL entrypoints, and Slurm launchers are wired to keep environments + caches inside the repo (`./var`).


## Quick Start

- Bootstrap the repo-local env + caches: `make conda-local` (uses `configs/environment.yml`, writes the env to `./var/openr1`, and keeps caches/tmp under `./var`).
- Activate: `conda activate ./var/openr1`; refresh installs via `pip install -c configs/constraints.txt -e .[dev]`.
- Authenticate with Hugging Face (`huggingface-cli login` or `export HF_TOKEN=…`) so gated models/datasets can be pulled by vLLM/TRL.
- Training telemetry / proof of work: public run stats at [wandb.ai ↗](https://api.wandb.ai/links/ogd3-princeton-university/aw6ecc9b).

Run the Hydra console scripts from the repo root:
```bash
# Baseline GRPO using the math recipe (static τ/β by default)
maxent-grpo-baseline baseline.recipe=configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml

# Inline overrides without a YAML recipe
maxent-grpo-baseline command=train-baseline \
  baseline.script.dataset_name=open-r1/OpenR1-Math-220k \
  baseline.training.output_dir=var/data/out

# Coding parity recipes (MBPP train + HumanEval eval) for 0.5B
maxent-grpo-baseline baseline.recipe=configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_code_mbpp.yaml
maxent-grpo command=train-maxent \
  maxent.recipe=configs/recipes/Qwen2.5-0.5B-Instruct/maxent-grpo/config_code_mbpp.yaml

# Optional: switch eval to APPS without changing the training dataset
maxent-grpo-baseline \
  baseline.recipe=configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_code_mbpp.yaml \
  baseline.script.eval_dataset_name=codeparrot/apps \
  baseline.script.eval_dataset_split=test \
  baseline.script.eval_dataset_prompt_column=question \
  baseline.script.eval_dataset_solution_column=input_output
```

Paired recipes (GRPO vs MaxEnt): the Qwen 0.5B/1.5B pairs under
`configs/recipes/<model>/grpo/config_math.yaml` and
`configs/recipes/<model>/maxent-grpo/config_math.yaml` keep the comparison
settings aligned on sampling, frozen-reference/KL, optimizer, and eval knobs.
For the cleanest three-way comparison, use the Hydra presets
`configs/recipes/hydra/grpo_custom_math.yaml`,
`configs/recipes/hydra/maxent_entropy_math.yaml`, and
`configs/recipes/hydra/maxent_listwise_math.yaml`, which layer
objective-specific overrides on top of the same GRPO base recipe.

The shipped Qwen 0.5B/1.5B `maxent-grpo` recipes default to
entropy-regularized MaxEnt (`objective: maxent_entropy`). In that mode the trainer always uses the
exact policy entropy gradient in the loss, so the recipe must keep
`maxent_policy_entropy_mode: exact`. The separate **GRPO + entropy bonus** mode
is still available through `policy_entropy_bonus_coef>0`; that reward-side path
adds a z-scored entropy bonus before the GRPO loss is built. The older Qwen 7B
`maxent-grpo` math recipe still uses that GRPO + entropy-bonus route.

For both native GRPO and entropy-regularized MaxEnt, the reference model stays
in play through the same beta-weighted KL term. The entropy term itself is
on-policy, but the trust-region anchor remains the frozen reference model via
`maxent_reference_logprobs_source: model` and `maxent_trl_reference_scoring: true`.

The paired flat recipes keep `controller_meta_enabled: false` by default so
GRPO vs MaxEnt comparisons stay static unless you explicitly opt in. The Hydra
convenience presets `configs/recipes/hydra/maxent_math.yaml` and
`configs/recipes/hydra/maxent_code_mbpp.yaml` do enable the analytic τ/β
meta-controller by default. Override with Hydra/CLI flags such as
`controller_meta_enabled=false` (fully static),
`controller_meta_method=first_order` (truncated gradients), or
`controller_meta_lr=0.02` for per-project retuning.

Examples:

```bash
# Disable meta-controller for ablations (MaxEnt pipeline)
maxent-grpo command=train-maxent maxent.training.controller_meta_enabled=false

# Enable first-order meta updates for baseline GRPO
maxent-grpo-baseline \
  baseline.training.controller_meta_enabled=true \
  baseline.training.controller_meta_method=first_order \
  baseline.training.controller_meta_lr=0.02
```

The MaxEnt runner uses the same training code path as baseline GRPO; select it with `maxent-grpo command=train-maxent` and MaxEnt-oriented recipe/overrides.

Slurm training (single-node dual launcher): `sbatch ops/slurm/train_dual_4plus4.slurm --model Qwen2.5-1.5B-Instruct --config math --accelerator zero3 --run-only grpo --grpo-args "--run_name demo --report_to wandb"`.
- Inspect `sbatch ops/slurm/train_dual_4plus4.slurm --help` for run mode, config overrides, and trainer arg forwarding. All caches stay under `var/` and run artifacts under `var/artifacts/` by default.

Notes
- Recipes live in `configs/recipes/` (Hydra shortcuts include `configs/recipes/hydra/{baseline,maxent}_math.yaml` plus MBPP coding variants `baseline_code_mbpp.yaml`, `grpo_custom_code_mbpp.yaml`, and `maxent_code_mbpp.yaml`).
- Coding reward: use `reward_funcs: [python_unit_tests]` (alias `mbpp_python_tests`) with test-based columns (`test_list` for MBPP, `test` for HumanEval, `input_output` for APPS). Tune execution with env vars `MAXENT_CODE_REWARD_TIMEOUT_S` and `MAXENT_CODE_REWARD_WORKERS`.
- Set `GRPO_RECIPE=<path>` to point any CLI at a YAML recipe; the TRL parser and Hydra CLI will load it automatically.
- For LightEval benchmarks via vLLM/Slurm, see `src/maxent_grpo/core/evaluation.py` (benchmark registry + launcher helper).
- Throughput knobs: `dataloader_num_workers`, `dataloader_pin_memory`, `dataloader_prefetch_factor`, `dataloader_persistent_workers` are honored by both GRPO and MaxEnt trainer entrypoints.
- Prompt cache: `maxent_prompt_cache_size` defaults to an auto-sized value (min 10k, max 50k based on batch size); set to 0 to disable or set a non-default size to override.
- Shared vLLM servers:
  - Every Accelerate rank derives a stable `client_tag` and forwards it via the JSON payload and `X-VLLM-Client-Tag` header so responses can be sharded per rank.
  - Override (`export VLLM_CLIENT_TAG=trainer-3`) when you run multiple independent trainers behind the same endpoint or when an external proxy wants to pin requests to a specific backend.
  - **Server requirement:** the vLLM server (or proxy) must copy the `client_tag` back into each prompt group of the JSON response (either at the result level or inside every output). Without that echo the client cannot filter, so you’ll see warnings such as “Skipping policy loss group due to empty log-probs,” `train/grpo_objective` stays at zero, and KL/entropy metrics remain `NaN`.
  - Guard rails: training now stops after `vllm_logprob_fail_after` consecutive steps with missing vLLM logprobs (default 3). Set `vllm_logprob_fallback=true` to switch to reference-model scoring instead, or `vllm_logprob_fail_after=0` to disable. `vllm_client_tag_fail_fast` controls abort-on-client_tag mismatch; env fallbacks are `MAXENT_VLLM_LOGPROB_FAIL_AFTER`, `MAXENT_VLLM_LOGPROB_FALLBACK`, `MAXENT_VLLM_CLIENT_TAG_FAIL_FAST`.
  - Weight sync: set `vllm_sync_interval_steps` to reduce sync stalls (e.g., sync every N optimizer steps; 0 disables sync).
  - **Verification:** when filtering works you’ll no longer see `vLLM raw groups=96 ...` immediately followed by `Score batch built | total_sequences=16`; the counts match and the controller metrics (`train/delta_tau`, `train/delta_beta`) move off zero even with chunking enabled. For a quick end-to-end sanity check, run `make smoke` and confirm client-tag filtering diagnostics appear before training steps.


## Code Organization
```
.
├─ configs/                     # env, constraints, recipes, prompts (Hydra shortcuts + TRL YAML + accelerate configs)
├─ sitecustomize.py             # repo-local Python bootstrapper (cache dirs)
├─ src/
│  └─ maxent_grpo/              # all code now lives under the namespaced package
│     ├─ cli/                   # Hydra console entrypoints (baseline/maxent training)
│     ├─ training/              # end-to-end GRPO/MaxEnt training + eval stack
│     │  ├─ generation/         # shared HF + vLLM generation helpers
│     │  ├─ patches/            # vLLM HTTP helpers used during rollout/scoring
│     │  ├─ runtime/ops/        # startup/health helpers for runtime services
│     │  └─ telemetry/          # wandb / metric logging wrappers
│     ├─ core/                  # dataset/model/evaluation utilities
│     ├─ rewards.py             # reward registry used by the baseline trainer
│     ├─ grpo.py                 # canonical trainer entrypoint
├─ docs/                        # Sphinx guides + API reference
├─ ops/                         # operational launchers/scripts (Slurm + smoke utilities)
├─ var/                         # runtime-only env, caches, and artifacts (gitignored)
```

Run artifacts (logs/results/outputs/wandb/details/controller_state.json) now live under `var/artifacts/`.


## Training Flow (Shared Baseline/MaxEnt Path)
1. **Entrypoint** — both variants launch `src/maxent_grpo/grpo.py` and call `maxent_grpo.training.baseline.run_baseline_training`.
2. **Data + prompt mapping** — `baseline.py` loads datasets, resolves prompt/answer columns, and maps to TRL chat-format prompts identically for GRPO and MaxEnt runs.
3. **Reward resolution** — `maxent_grpo.training.rewards.load_reward_functions` resolves reward callables/weights identically for both runs.
4. **Trainer objective** — `maxent_grpo.training.trl_trainer` is the objective boundary; this is where GRPO vs MaxEnt behavior diverges.
5. **Optimization/checkpointing** — TRL/HF trainer handles stepping, metrics, and checkpoint save/resume in the same way for both variants.

## Environment Notes
- Transformers/TRL require `huggingface-hub < 1.0`; the launchers repin Hub and small CLI deps (`yq`, `rich`, `markdown-it-py`) inside the repo-local env.
- vLLM requires a CUDA-enabled PyTorch. The Slurm launcher loads CUDA 12.6 by default and routes all caches/temp dirs into `var/`.
- MaxEnt recipes/CLI now invoke the same baseline training pipeline as GRPO; objective behavior differs only inside `maxent_grpo.training.trl_trainer`.


## Troubleshooting
- vLLM not healthy / empty server log: inspect `var/artifacts/logs/slurm_%j.out` for `nvidia-smi -L`. If it prints “No devices found”, adjust SBATCH partition/GRES.
- “Invalid generic resource (gres) specification”: ensure your cluster supports `--gres=gpu:a100:7` or switch to `#SBATCH --gpus=7`.
- Transformers complaining about `huggingface-hub==1.1.1`: reinstall inside the env with `pip install 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0'`.


## Development
- Install dev tooling with `pip install -r configs/requirements-dev.txt` (enforces `configs/constraints.txt`).
- Run `pytest -q` (or `make test`) from the repo root.
- Type check via `pyright --project configs/pyrightconfig.json`.
- Optional commit hooks: `pre-commit install` then `pre-commit run -a`.
- Quick offline smoke (paired GRPO + MaxEnt training) with isolated caches under `var/`: `make smoke`


## Documentation
- Online: https://maxent-grpo.readthedocs.io/en/latest/
- Local build: `pip install -c configs/constraints.txt -r docs/requirements.txt && sphinx-build -b html docs var/docs/_build/html`


## Citation
If you use this work, please cite: “MaxEnt‑GRPO: Maximum-Entropy Group-Relative Policy Optimization (2025).”

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
