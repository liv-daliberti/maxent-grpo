# Tests Layout

The test tree is organized around the GRPO/MaxEnt training + eval workflow.

## Structure

- `tests/cli/`: CLI entrypoints and console scripts
- `tests/config/`: configuration and recipe validation
- `tests/core/`: core utilities and data/model helpers
- `tests/evaluation/`: evaluation and scoring behavior
- `tests/rewards/`: reward logic (including weighting helpers)
- `tests/training/`: training-centric coverage
- `tests/training/generation/`: generation helpers used by training
- `tests/training/generation/vllm/`: vLLM integration behavior
- `tests/training/pipeline/`: training pipeline orchestration/parity checks
- `tests/training/runtime/`: runtime/dependency checks
- `tests/training/runtime/logging/`: training telemetry/logging behavior
- `tests/training/runtime/ops/`: training ops/tooling helpers
- `tests/training/runtime/setup/`: environment/setup behavior

## Markers

- `cli`, `config`, `core`, `evaluation`, `training`, `rewards`, `vllm`
- `generation`, `runtime`, `logging`, `ops`, `setup`, `pipelines` are retained for focused selection
- `integration` for cross-module and external-interface tests

Examples:
- `pytest tests/training/generation/vllm -q`
- `pytest -m "training and not slow"`
- `pytest tests/evaluation -q`
