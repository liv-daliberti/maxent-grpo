from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49ab_all_observed_persistent_pairwise_route_discovery.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49ab_all_observed_pairwise", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e49ab_replays_every_validator_positive_response():
    module = _load()
    from datasets import load_from_disk

    rows = load_from_disk(str(module.BASE.FULL / "train"))["train"]
    candidates = module._all_candidates(rows)
    assert len(candidates) == 74
    assert sum(len(row["exemplars"]) for row in candidates) == 1550
    assert all(
        row["exemplars"]
        == sorted(
            row["exemplars"], key=lambda exemplar: exemplar["response_sha256"]
        )
        for row in candidates
    )


def test_e49ab_protocol_keeps_pairwise_and_execution_gates():
    module = _load()
    protocol = module.PROTOCOL.read_text(encoding="utf-8")
    slurm = (
        ROOT
        / "ops/slurm/e49ab_all_observed_persistent_pairwise_node302.slurm"
    ).read_text(encoding="utf-8")
    assert "contain 1,550" in protocol
    assert "groups for one problem must remain" in protocol
    assert "sequential" in protocol
    assert "Wrong-route solutions do not count" in protocol
    assert "#SBATCH --gres=gpu:a100:1" in slurm
