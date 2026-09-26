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
        match = re.fullmatch(r"\s*export ([A-Z0-9_]+)=(.*)", line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def test_verified_support_variant_changes_only_the_estimator_identity() -> None:
    v6 = _exports(_case_body("verified_replay_semantic_maxent_group_centered"))
    v7_body = _case_body("verified_replay_semantic_maxent_verified_support")
    v7 = _exports(v7_body)
    estimator_keys = {
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK",
    }

    assert {key: value for key, value in v6.items() if key not in estimator_keys} == {
        key: value for key, value in v7.items() if key not in estimator_keys
    }
    assert v6[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
    ] == "1"
    assert v6[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE"
    ] == "0"
    assert v7[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
    ] == "0"
    assert v7[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE"
    ] == "1"
    assert v7[
        "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK"
    ] == "0"
    assert v7["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
    assert v7["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "verified_likelihood_per_rollout"
    )
    assert "VARIANT_TAG=\"verified_replay_semantic_maxent_verified_support\"" in v7_body


def test_verified_support_flag_is_default_off_and_forwarded() -> None:
    run_text = RUN_EXPERIMENT.read_text(encoding="utf-8")
    train_text = TRAIN.read_text(encoding="utf-8")

    assert (
        "export OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_"
        "VERIFIED_SUPPORT_ADVANTAGE=0"
    ) in run_text
    assert "--semantic-shannon-success-conditioned-verified-support-advantage" in train_text
    assert "--no-semantic-shannon-success-conditioned-verified-support-advantage" in train_text
    assert "semantic_estimator=mode_selected" in train_text



def test_discovery_variant_keeps_replaydr_uniform_and_proposals_outside_ppo() -> None:
    body = _case_body(
        "verified_replay_semantic_maxent_verified_support_discovery"
    )
    values = _exports(body)

    assert values[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE"
    ] == "1"
    assert values[
        "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK"
    ] == "1"
    assert values["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
    assert values["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "verified_likelihood_per_rollout"
    )
    assert values["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"
    assert values[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT"
    ] == "1"
    assert values[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS"
    ] == "0"
    assert values["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS"] == "0"
    assert values[
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER"
    ] == "1.0"
    assert "Proposal rows never enter PPO or on-policy frequency counts" in body


def test_replay_support_coupling_flag_is_default_off_and_forwarded() -> None:
    run_text = RUN_EXPERIMENT.read_text(encoding="utf-8")
    train_text = TRAIN.read_text(encoding="utf-8")

    assert (
        "export OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_"
        "INCLUDE_REPLAY_BANK=0"
    ) in run_text
    assert "--semantic-shannon-verified-support-include-replay-bank" in train_text
    assert "--no-semantic-shannon-verified-support-include-replay-bank" in train_text
    assert "verified_support_include_replay_bank=" in train_text
