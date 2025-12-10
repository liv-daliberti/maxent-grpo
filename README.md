# MaxEnt-GRPO: Maximum‑Entropy Group‑Relative Policy Optimization

A clean, maximum‑entropy variant of GRPO for sequence‑level (candidate‑level) learning. Hydra console scripts, TRL entrypoints, and Slurm launchers are wired to keep environments + caches inside the repo (`./var`).


## Quick Start

- Bootstrap the repo-local env + caches: `bash ops/scripts/bootstrap_env.sh` (wraps `configs/environment.yml`, writes the env to `./var/openr1` and all caches/tmp under `./var`). Equivalent: `make conda-local`.
- Activate: `conda activate ./var/openr1`; refresh installs via `pip install -c configs/constraints.txt -e .[dev]`.
- Authenticate with Hugging Face (`huggingface-cli login` or `export HF_TOKEN=…`) so gated models/datasets can be pulled by vLLM/TRL.
- Training telemetry / proof of work: public run stats at [wandb.ai ↗](https://api.wandb.ai/links/ogd3-princeton-university/aw6ecc9b).

Run the Hydra console scripts from the repo root:
```bash
# Baseline GRPO using the math recipe (static τ/β by default)
maxent-grpo-baseline baseline.recipe=configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml

# InfoSeed-GRPO using the math recipe (custom loop; τ/β meta-controller enabled)
maxent-grpo-infoseed infoseed.recipe=configs/recipes/Qwen2.5-1.5B-Instruct/infoseed/config_math.yaml

# Inline overrides without a YAML recipe
maxent-grpo-baseline command=train-baseline \
  baseline.script.dataset_name=open-r1/OpenR1-Math-220k \
  baseline.training.output_dir=var/data/out

# Generation + inference helpers
maxent-grpo-generate command=generate \
  generate.args.hf_dataset=open-r1/OpenR1-Math-220k \
  generate.args.model=Qwen/Qwen2.5-1.5B-Instruct
maxent-grpo-math-eval command=math-eval \
  inference.dataset=math_500 \
  inference.num_generations=8 \
  inference.seeds=[0,1,2,3,4] \
  inference.temperature=0.6 \
  inference.models='[ {model_name_or_path: od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1} ]' \
  inference.collect_generations=false \
  inference.artifacts.root_dir=var/artifacts/inference
```

All MaxEnt and InfoSeed recipes now enable the τ/β meta-controller (analytic mode) by default so weights stay on target without manual tuning. Override with Hydra/CLI flags such as ``controller_meta_enabled=false`` (fully static), ``controller_meta_method=first_order`` (truncated gradients), or ``controller_meta_lr=0.02`` for per-project retuning. The baseline GRPO recipes keep the controller disabled unless you explicitly flip the flag.

Examples:

```bash
# Disable meta-controller for ablations (MaxEnt pipeline)
maxent-grpo-maxent maxent.training.controller_meta_enabled=false

# Enable first-order meta updates for baseline GRPO
maxent-grpo-baseline \
  baseline.training.controller_meta_enabled=true \
  baseline.training.controller_meta_method=first_order \
  baseline.training.controller_meta_lr=0.02
```

The MaxEnt runner is provided as building blocks under `src/maxent_grpo/training/` and a YAML recipe; the convenience CLI (`maxent-grpo-maxent` / `--task maxent`) currently raises until the dedicated launcher is restored.

Multi-node training (recommended): `sbatch ops/slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config math --accelerator zero3 --args "--run_name demo --report_to wandb"`.
- Inspect `sbatch ops/slurm/train.slurm --help` for dp/tp, ports, vLLM mode, and accelerate config knobs. All caches/logs stay under `var/` by default.

Evaluate math benchmarks locally via `src/inference`:
```python
from maxent_grpo.inference import InferenceModelSpec, run_math_eval_inference

specs = [
    InferenceModelSpec(
        model_name_or_path="od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1",
        style="grpo",
        system_prompt="...",  # match the recipe prompt
    ),
    InferenceModelSpec(
        model_name_or_path="od2961/Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-math-v1",
        style="maxent",
        system_prompt="...",
    ),
]
for res in run_math_eval_inference(specs):
    print(res.label, res.accuracy)
```

Notes
- Recipes live in `configs/recipes/` (Hydra shortcuts: `configs/recipes/hydra/{baseline,maxent}_math.yaml`).
- InfoSeed runner: recipe at `configs/recipes/Qwen2.5-1.5B-Instruct/infoseed/config_math.yaml` with Hydra shortcut `configs/recipes/hydra/infoseed_math.yaml`; console alias `maxent-grpo-infoseed`.
- Inference presets cover `math_500`, `aime24`, `aime25`, `amc`, and `minerva`; override columns/splits via `inference.eval.*` if you use alternative mirrors. The math eval pipeline reports Pass@1, Pass@k (default k=8), and Avg@k averaged across seeds (default seeds: `[0,1,2,3,4]` at temperature 0.6).
  - Every math eval invocation now persists prompt-level artifacts under `var/artifacts/inference/<model>/<dataset>/<temp>/seed_<n>.jsonl`. Appends are flushed after each prompt so preempted jobs can resume automatically; rerunning `maxent-grpo-math-eval` reuses completed prompts and continues where the previous Slurm job stopped.
  - The table helper `python tools/math_eval_table.py --artifact-root var/artifacts/inference` renders three terminal tables (Pass@1 / Pass@8 / Avg@8) averaged over all seeds using the artifact JSON. Adjust `--precision` or `--min-datasets` to control formatting/filters.
- Preferred CLI alias for multi-benchmark math inference: `maxent-grpo-math-eval` (older `maxent-grpo-inference command=math-eval` remains for compatibility).
- Slurm helper to sweep all math benchmarks for a checkpoint: `sbatch ops/slurm/infer_math.slurm --model <HF_ID_OR_PATH> --datasets math_500,aime24,aime25,amc,minerva`. Pass `--revision <git_commit>` when pointing at Hugging Face repos (e.g., `--model od2961/Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-math-v1 --revision c929c65`) to evaluate specific training steps without syncing local checkpoints.
- Set `GRPO_RECIPE=<path>` to point any CLI at a YAML recipe; the TRL parser and Hydra CLI will load it automatically.
- For LightEval benchmarks via vLLM/Slurm, see `src/core/evaluation.py` (benchmark registry + launcher helper).
- Shared vLLM servers:
  - Every Accelerate rank derives a stable `client_tag` and forwards it via the JSON payload and `X-VLLM-Client-Tag` header so responses can be sharded per rank.
  - Override (`export VLLM_CLIENT_TAG=trainer-3`) when you run multiple independent trainers behind the same endpoint or when an external proxy wants to pin requests to a specific backend.
  - **Server requirement:** the vLLM server (or proxy) must copy the `client_tag` back into each prompt group of the JSON response (either at the result level or inside every output). Without that echo the client cannot filter, so you’ll see warnings such as “Skipping policy loss group due to empty log-probs,” `train/grpo_objective` stays at zero, and KL/entropy metrics remain `NaN`.
  - **Verification:** when filtering works you’ll no longer see `vLLM raw groups=96 ...` immediately followed by `Score batch built | total_sequences=16`; the counts match and the controller metrics (`train/delta_tau`, `train/delta_beta`) move off zero even with chunking enabled. For a quick end‑to‑end check on a local model, run `python tools/vllm_client_tag_smoke.py --model <HF_ID>` and confirm it logs “server echoed client_tag …” before launching training.


## Code Organization
```
.
├─ configs/                     # env, constraints, recipes (Hydra shortcuts + TRL YAML + accelerate configs)
├─ ops/                         # operations toolbox (slurm launchers, env bootstrap, PATH helpers, sitecustomize)
├─ src/
│  └─ maxent_grpo/              # all code now lives under the namespaced package
│     ├─ cli/                   # Hydra console entrypoints (baseline/maxent/generate/inference)
│     ├─ pipelines/             # end-to-end pipelines (baseline GRPO trainer, distilabel generation, math_500 inference)
│     ├─ training/              # MaxEnt runner: loop, generation, scoring/weighting, optim/state, types
│     ├─ generation/            # shared HF + vLLM generation helpers
│     ├─ core/                  # dataset/model/evaluation utilities
│     ├─ telemetry/             # wandb logging wrappers
│     ├─ patches/               # compatibility shims for TRL/vLLM/transformers
│     ├─ rewards.py             # reward registry used by the baseline trainer
│     ├─ grpo.py / maxent_grpo.py  # TRL-style entrypoints (baseline GRPO + MaxEnt shim)
│     └─ inference/             # math_500 inference helpers (re-exported by src/maxent_grpo/inference)
├─ tools/                       # local utilities (e.g., log validation)
├─ docs/                        # Sphinx guides + API reference
├─ var/                         # repo-local env, caches, logs, data outputs (gitignored)
└─ data/                        # optional staging area for large datasets/checkpoints
```


## Training Flow (MaxEnt Runner)
1. **Generation** — `training.generation.CompletionGenerator` wraps HF + vLLM; `GenerationContext` carries sampling + accelerator handles.
2. **Rewards & scoring** — `training.rewards` + `training.pipeline.prepare_training_batch` build grouped completions, reward stats, reference log-probs, and sequence scores.
3. **Weighting & loss** — `training.weighting.loss` + `training.weighting.logic` compute listwise targets/weights (entropy, KL) and loss scalars.
4. **Optimization** — `training.loop` drives gradient accumulation and schedules; `training.optim` + `training.state` handle LR schedules, controllers, checkpoints.
5. **Logging** — `training.metrics` and `telemetry.wandb` report metrics; logs/checkpoints land under `var/` by default.

## InfoSeed-GRPO (seed-aware auxiliary, optional)
- Enable via `info_seed_enabled=True` (Hydra now enforces this for `train-infoseed` recipes); suggested defaults: `info_seed_lambda=0.01`, `info_seed_alpha_entropy=0.5`, `info_seed_num_seeds=4–8`, `info_seed_prompt_template="\n[seed={seed}]"`.
- Losses: seed-aware contrastive (`info_seed_loss_type=infonce`) or CE over `seed_head` logits (`info_seed_loss_type=ce`); pooling via `info_seed_pooling` (`mean`/`last`).
- Metrics: logs `seed_pred_acc`, per-subset entropies (orig vs seed-aug), seed diversity (`seed_diversity_l2`), and optional eval-time seed metrics (`eval_seed/*`) when `EvaluationSettings.seed_eval` is configured.
- Failure modes: high `info_seed_lambda` can produce stylistic hacks. Mitigate by lowering `lambda`, capping entropy, adding dropout/perturbation to seed template tokens, and monitoring `seed_pred_acc` vs. diversity.


## Environment Notes
- Transformers/TRL require `huggingface-hub < 1.0`; the launchers repin Hub and small CLI deps (`yq`, `rich`, `markdown-it-py`) inside the repo-local env.
- vLLM requires a CUDA-enabled PyTorch. The Slurm launcher loads CUDA 12.6 by default and routes all caches/temp dirs into `var/`.
- The MaxEnt entrypoint (`training.run_maxent_training`) delegates to `pipelines.training.maxent.run_maxent_training`; compose a custom runner with `training.loop`/`training.pipeline` if you need finer control.


## Troubleshooting
- vLLM not healthy / empty server log: inspect `var/logs/slurm_%j.out` for `nvidia-smi -L`. If it prints “No devices found”, adjust SBATCH partition/GRES.
- “Invalid generic resource (gres) specification”: ensure your cluster supports `--gres=gpu:a100:7` or switch to `#SBATCH --gpus=7`.
- Transformers complaining about `huggingface-hub==1.1.1`: reinstall inside the env with `pip install 'huggingface-hub[cli,hf_xet]>=0.30.2,<1.0'`.


## Development
- Install dev tooling with `pip install -r configs/requirements-dev.txt` (enforces `configs/constraints.txt`).
- Run `pytest -q -c configs/pytest.ini` (or `make test`) so the relocated config is picked up.
- Type check via `pyright --project configs/pyrightconfig.json`.
- Optional commit hooks: `pre-commit install` then `pre-commit run -a`.
- Quick offline smoke (generation → training shim → inference) with isolated caches under `var/`: `make smoke`


## Documentation
- Online: https://maxent-grpo.readthedocs.io/en/latest/
- Local build: `pip install -c configs/constraints.txt -r docs/requirements.txt && sphinx-build -b html docs var/docs/_build/html`


## Citation
If you use this work, please cite: “MaxEnt‑GRPO: Maximum-Entropy Group-Relative Policy Optimization (2025).”

Quick eval parity check (shared math_500 pipeline):

```bash
# stubbed, offline delta comparison between baseline and candidate (math_500 runner)
make eval-math-delta  # uses tools/eval_math_delta.py with a fixed seed/dataset
# real models: swap stub runner for transformers and point at downloaded dataset
python tools/eval_math_delta.py --baseline /path/to/grpo --candidate /path/to/maxent --runner transformers --dataset /path/to/math_500.jsonl
```

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
