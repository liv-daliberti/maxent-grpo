# Tests layout

This repository uses a hybrid layout: subpackages for discovery + pytest markers for selection.
Markers are auto-applied in `tests/conftest.py` based on path and filename tokens.

## Structure

- `tests/cli/`: CLI entrypoints and console scripts
- `tests/config/`: configuration and recipe validation
- `tests/core/`: core utilities and data/model helpers
- `tests/evaluation/`: inference, evaluation, scoring
- `tests/generation/`: generation helpers (see `tests/generation/vllm/` for vLLM)
- `tests/ops/`: ops/tooling, perf, validation scripts
- `tests/pipelines/`: pipeline orchestration/integration
- `tests/rewards/`: reward logic (see `tests/rewards/weighting/` for weighting)
- `tests/runtime/`: runtime/deps/setup/logging helpers
- `tests/training/`: training loop, optim, metrics, and helpers

## Markers

- `cli`, `config`, `core`, `evaluation`, `generation`, `pipelines`, `runtime`, `ops`, `training`, `rewards`, `vllm`
- `logging`, `setup` are added from subdirectories or filename tokens
- `integration` is added for pipeline tests and files containing "integration"

Examples:
- `pytest -m "generation and vllm"`
- `pytest -m "training and not slow"`
- `pytest tests/cli -m "integration"`
