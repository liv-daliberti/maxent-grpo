# Data Directory

This repository does not version datasets or model artifacts directly. Use this folder (or `var/data/`) as a staging area for downloaded benchmarks, generated samples, and training/eval outputs.

- Runtime outputs default to `var/data/...` in the recipes and Slurm launchers; keep the repo root clean by writing there.
- **Large blobs**: download or symlink them here instead of committing them. Document the source/commands alongside any helper scripts you add.
- **Tests/fixtures**: keep tiny fixtures under `tests/` (e.g., `tests/helpers/`); avoid placing test data here.
- Hugging Face caches are routed to `var/cache/huggingface/...` by the launchers. Use `data/` for local mirrors or one-off exports that you manage manually.
