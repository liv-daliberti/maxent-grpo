# Operations Toolkit

Utilities under `ops/` keep infrastructure bits out of the repo root:

- `slurm/` – cluster launchers such as `train.slurm`, `maxent-grpo.slurm`, and helper jobs (evaluation, inference, etc.).
- `scripts/` – automation helpers for local development (e.g., `bootstrap_env.sh`, dataset prep scripts, cache cleaners).
- `tools/` – lightweight shell helpers such as `ensure_local_path.sh`.
- `sitecustomize.py` – repo-local Python bootstrapper that injects `src/` into `sys.path` and installs lightweight stubs when optional deps are missing.

Run everything from the repo root: commands in the docs/README already point to the new `ops/...` locations.
