from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "var/data/pantry_plan_modebench_v1"


@pytest.mark.skipif(
    not (DATA_ROOT / "identity.json").is_file(),
    reason="frozen PantryPlan data are not materialized",
)
def test_frozen_pantry_plan_identity_and_prompt_boundary():
    datasets = pytest.importorskip("datasets")
    identity = json.loads((DATA_ROOT / "identity.json").read_text())
    train = datasets.load_from_disk(str(DATA_ROOT / "train"))["train"].to_list()
    evaluation = datasets.load_from_disk(str(DATA_ROOT / "eval"))[
        "multi_answer"
    ].to_list()

    assert identity["train_rows"] == len(train) == 96
    assert identity["eval_rows"] == len(evaluation) == 32
    assert identity["train_eval_overlap_count"] == 0
    assert identity["minimum_exact_support_count"] >= 8
    assert identity["maximum_exact_support_count"] <= 64
    assert set(identity["family_counts"]["train"].values()) == {24}
    assert set(identity["family_counts"]["eval"].values()) == {8}

    train_fingerprints = {row["instance_fingerprint"] for row in train}
    eval_fingerprints = {row["instance_fingerprint"] for row in evaluation}
    assert len(train_fingerprints) == 96
    assert len(eval_fingerprints) == 32
    assert not train_fingerprints & eval_fingerprints
    for row in train + evaluation:
        assert "certified_mode_count" not in row["problem"]
        assert "certified_support_sha256" not in row["problem"]
        spec = json.loads(row["answer"])
        assert row["answer_mode_count"] == spec["certified_mode_count"]
        assert len(spec["certified_support_sha256"]) == 64
        int(spec["certified_support_sha256"], 16)

    manifest_digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    assert len(manifest_digest) == 64
