# Operations Toolkit

Utilities under `ops/` keep infrastructure bits out of the repo root and route everything into `var/`:

- `slurm/train.slurm` — primary multi-node launcher for GRPO (and legacy MaxEnt) with optional vLLM server orchestration; `--help` lists dp/tp/port/accelerate knobs.
- `slurm/maxent-grpo.slurm` — legacy MaxEnt-specific launcher; uses the same env wiring but the MaxEnt entrypoint is currently stubbed.
Runnable utilities live under `tools/` (e.g., `tools/bootstrap_env.sh`, `tools/ensure_local_path.sh`).
Repo-local Python bootstrapper: `sitecustomize.py` (at repo root).
Run commands from the repo root so relative paths to configs, logs, and `var/artifacts/` resolve correctly.
