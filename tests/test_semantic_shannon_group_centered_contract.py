from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "ops/run_experiment.sh"
TRAIN = ROOT / "ops/train.sh"


def _case_body(name: str) -> str:
    text = RUN_EXPERIMENT.read_text(encoding="utf-8")
    match = re.search(
        rf"^  {re.escape(name)}\)\n(.*?)^    ;;$", text, re.M | re.S
    )
    assert match is not None
    return match.group(1)


def _exports(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        match = re.fullmatch(r"export ([A-Z0-9_]+)=(.*)", line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def test_group_centered_variant_changes_only_the_semantic_estimator_contract():
    legacy = _exports(_case_body("verified_replay_semantic_maxent"))
    repaired_body = _case_body(
        "verified_replay_semantic_maxent_group_centered"
    )
    repaired = _exports(repaired_body)

    semantic_diff = {
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL",
    }
    shared_keys = set(legacy) - semantic_diff
    assert {key: legacy[key] for key in shared_keys} == {
        key: repaired[key] for key in shared_keys
    }
    assert repaired[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE"
    ] == "0"
    assert repaired[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
    ] == "1"
    assert repaired["OAT_ZERO_SEMANTIC_RMS_CONTROL"] == "0"
    assert 'VARIANT_TAG="verified_replay_semantic_maxent_group_centered"' in (
        repaired_body
    )


def test_group_centered_flag_is_reset_and_forwarded_to_frozen_source():
    run_text = RUN_EXPERIMENT.read_text(encoding="utf-8")
    train_text = TRAIN.read_text(encoding="utf-8")

    assert (
        "export "
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=0"
        in run_text
    )
    assert (
        "--semantic-shannon-success-conditioned-group-centered-advantage"
        in train_text
    )
    assert (
        "--no-semantic-shannon-success-conditioned-group-centered-advantage"
        in train_text
    )
