# Operations Toolkit

Utilities under `ops/` keep infrastructure bits out of the repo root and route everything into `var/`:

- `slurm/train.slurm` — primary multi-node launcher for GRPO (and legacy MaxEnt) with optional vLLM server orchestration; `--help` lists dp/tp/port/accelerate knobs.
- `slurm/maxent-grpo.slurm` — legacy MaxEnt-specific launcher; uses the same env wiring but the MaxEnt entrypoint is currently stubbed.
- `sitecustomize.py` — repo-local Python bootstrapper that injects `src/` into `sys.path` and installs lightweight stubs when optional deps are missing.

Runnable utilities live under `tools/` (e.g., `tools/bootstrap_env.sh`, `tools/ensure_local_path.sh`).
Run commands from the repo root so relative paths to configs, logs, and `var/artifacts/` resolve correctly.
