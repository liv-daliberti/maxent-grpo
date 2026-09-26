from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49aa_calibrated_pairwise_observed_route_discovery.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49aa_pairwise_route_discovery", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e49aa_uses_exact_calibrated_pairwise_executable():
    module = _load()
    assert module._sha256(module.PAIRWISE_SOURCE) == module.EXPECTED["pairwise"]
    pairwise = module._load_pairwise_module()
    judge = pairwise.MathStrategyCanonicalizer(
        endpoint="http://judge.invalid/v1"
    )
    assert (
        judge.state_dict()["schema"]
        == "math_strategy_canonicalizer_pair_veto_v15"
    )


def test_e49aa_frozen_observed_cohort_has_74_candidates():
    module = _load()
    from datasets import load_from_disk

    rows = load_from_disk(str(module.FULL / "train"))["train"]
    candidates = module._candidates(rows)
    assert len(candidates) == 74
    assert all(int(row["level"]) >= 4 for row in candidates)
    assert all(4 <= len(row["exemplars"]) <= 16 for row in candidates)
    assert all(
        len(exemplar["text"]) <= 6000
        for row in candidates
        for exemplar in row["exemplars"]
    )


def test_e49aa_is_contingent_fail_closed_and_uses_one_a100():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e49aa_calibrated_pairwise_observed_route_discovery_20260726.md"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT
        / "ops/math_strategy_calibration/"
        "launch_e49aa_calibrated_pairwise_route_discovery.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        ROOT
        / "ops/slurm/"
        "e49aa_calibrated_pairwise_route_discovery_node302.slurm"
    ).read_text(encoding="utf-8")
    assert "only if the final E49Y result" in protocol
    assert "No failed route may" in protocol
    assert "be relabeled, simplified, or replaced" in protocol
    assert 'result.get("pass") is True' in launcher
    assert "#SBATCH --gres=gpu:a100:1" in slurm
