from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import materialize_e117_evaluation_reserves as reserve  # noqa: E402


RESERVE_ROOT = ROOT / "var/data/e117_evaluation_reserve_v1"
IDENTITY_PATH = RESERVE_ROOT / "identity.json"


def _rows(block: str, domain: str) -> list[dict[str, object]]:
    dataset = load_from_disk(str(RESERVE_ROOT / block / domain / "eval"))
    assert {str(name): len(split) for name, split in dataset.items()} == {
        "multi_answer": 128
    }
    return [dict(row) for row in dataset["multi_answer"]]


def test_reserve_manifest_binds_protocol_code_tests_and_data():
    identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    assert identity["schema"] == "e117_evaluation_reserve_v1"
    assert identity["blocks"] == ["development", "confirmation"]
    assert identity["rows_per_domain_per_block"] == 128
    assert identity["protocol_sha256"] == reserve._sha256_file(reserve.PROTOCOL)
    assert identity["materializer_sha256"] == reserve._sha256_file(
        Path(reserve.__file__)
    )
    assert identity["tests_sha256"] == reserve._sha256_file(Path(__file__))
    assert identity["data_tree_sha256"] == reserve._reserve_data_tree_sha256(
        RESERVE_ROOT
    )
    assert identity["stage1_launch_authorized"] is False
    assert identity["stage1_outcomes_exist"] is False
    assert identity["model_outcomes_inspected"] is False
    assert identity["response_data_read"] is False
    assert identity["pointmaze"] == "excluded"


def test_every_reserve_is_exact_unique_and_three_way_disjoint():
    identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    for domain in reserve.DOMAIN_ORDER:
        source_rows = reserve._load_source_rows(domain)
        development = _rows("development", domain)
        confirmation = _rows("confirmation", domain)
        source_ids = reserve._identities(domain, source_rows)
        development_ids = reserve._identities(domain, development)
        confirmation_ids = reserve._identities(domain, confirmation)

        assert len(development) == len(development_ids) == 128
        assert len(confirmation) == len(confirmation_ids) == 128
        assert not source_ids & development_ids
        assert not source_ids & confirmation_ids
        assert not development_ids & confirmation_ids

        record = identity["domains"][domain]
        assert record["overlap_counts"] == {
            "development_vs_confirmation": 0,
            "historical_vs_confirmation": 0,
            "historical_vs_development": 0,
        }
        assert record["development"]["rows_sha256"] == reserve._rows_sha256(
            development
        )
        assert record["confirmation"]["rows_sha256"] == reserve._rows_sha256(
            confirmation
        )
        assert record["development"]["identity_sha256"] == reserve._identity_sha256(
            development_ids
        )
        assert record["confirmation"]["identity_sha256"] == reserve._identity_sha256(
            confirmation_ids
        )
        reserve._validate_reserved_rows(
            domain, development, split_tag="e117_stage1_development"
        )
        reserve._validate_reserved_rows(
            domain, confirmation, split_tag="e117_confirmation"
        )


def test_fixed_seed_and_use_boundaries_are_explicit():
    identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    assert identity["development_is_only_stage1_evaluation_block"] is True
    assert identity["confirmation_reserved_before_development_outcomes"] is True
    assert identity["confirmation_sealed_from_development_analysis"] is True
    assert identity["historical_training_banks_retained"] is True
    for block in ("development", "confirmation"):
        for domain in reserve.DOMAIN_ORDER:
            assert identity["domains"][domain][block]["seed"] == reserve.SEEDS[block][
                domain
            ]


def test_recovered_countdown_and_graph_source_contracts_remain_exact():
    reserve._verify_recovered_source_contracts()


def test_materializer_refuses_to_overwrite_the_frozen_reserve():
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reserve.materialize(RESERVE_ROOT)
