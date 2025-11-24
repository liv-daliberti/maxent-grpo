# Operations Toolkit

Utilities under `ops/` keep infrastructure bits out of the repo root and route everything into `var/`:

- `scripts/bootstrap_env.sh` — bootstrap a repo-local conda env at `./var/openr1` with all caches/tmp under `./var`.
- `slurm/train.slurm` — primary multi-node launcher for GRPO (and legacy MaxEnt) with optional vLLM server orchestration; `--help` lists dp/tp/port/accelerate knobs.
- `slurm/maxent-grpo.slurm` — legacy MaxEnt-specific launcher; uses the same env wiring but the MaxEnt entrypoint is currently stubbed.
- `tools/` — lightweight shell helpers such as `ensure_local_path.sh`.
- `sitecustomize.py` — repo-local Python bootstrapper that injects `src/` into `sys.path` and installs lightweight stubs when optional deps are missing.

Run commands from the repo root so relative paths to configs, logs, and var/ artifacts resolve correctly.
