# Data Directory

This repository does not version datasets or model artifacts directly. Use this folder (or `var/data/`) as a staging area for downloaded benchmarks, generated samples, and training/eval outputs.

- **Tests/fixtures**: keep small fixtures under `tests/` (e.g., `tests/helpers/`); do not place test data here.
- **Large blobs**: download or symlink them here via your own scripts instead of committing them. For reproducibility, document the source/commands next to the scripts you use.
- **Outputs/checkpoints**: point `output_dir` or similar paths to `var/data/...` to keep the repo root clean.
