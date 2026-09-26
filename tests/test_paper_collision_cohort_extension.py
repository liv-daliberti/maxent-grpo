"""A census extension must never silently reuse changed response admission."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
from exp_scaling import extend_paper_collision_samples as extension
from exp_scaling import load_paper_collision_samples as samples


def checkpoint(value=1):
    return {"draws": [{"metrics": {"value": value}, "metadata": {"seed": 100},
                       "origins": [{"path": "source", "line": 1}]}]}


def cell(checkpoints):
    return {"level": "level1", "scale": "qwen3b", "domain": "mathir", "method": "maxrl", "seed": 72,
            "terminal_admitted": False, "complete_checkpoints": checkpoints}


def test_new_admission_of_an_existing_checkpoint_needs_no_new_samples():
    old = cell({"0": checkpoint(), "3072": checkpoint()})
    new = deepcopy(old); new["terminal_admitted"] = True
    plan = extension.change_plan({"cells": [old]}, {"cells": [new]})
    assert len(plan["reused"]) == 2
    assert len(plan["terminal_admission_changes"]) == 1
    assert not plan["added"] and not plan["changed_complete"]


@pytest.mark.parametrize("field", ["metrics", "metadata", "origins"])
def test_any_changed_raw_admission_prevents_cached_reuse(field):
    old = cell({"3072": checkpoint()}); new = deepcopy(old)
    new["complete_checkpoints"]["3072"]["draws"][0][field] = {"changed": True}
    plan = extension.change_plan({"cells": [old]}, {"cells": [new]})
    assert len(plan["changed_complete"]) == 1
    assert not plan["reused"]


def test_withdrawn_conflicted_initial_is_not_retained():
    old = cell({"0": checkpoint()}); new = cell({"3072": checkpoint()})
    plan = extension.change_plan({"cells": [old]}, {"cells": [new]})
    assert plan["removed"][0]["step"] == 0
    assert plan["added"][0]["step"] == 3072


def normalized_checkpoint():
    draws = []
    for d in range(4):
        rows = [{"prompt_id": str(i), "prompt_index": i, "verified_keys": ["valid"] + [None] * 7,
                 "request_seeds_by_option": [100 + d], "option_ids": [None] * 8} for i in range(128)]
        draws.append({"draw_index": d, "metadata": {"seed": 100 + d}, "prompts": rows,
                      "observed_metrics": {"any_correct_at_k": 1.0, "distinct_correct_modes_at_k": 1.0, "mean_at_k": .125},
                      "origins": [], "raw_payload_sha256": "fixture"})
    return {"draws": draws, "sampling_certificate": {"fixture": True}}


def test_compaction_keeps_original_k8_groups_and_accepts_neutral_parent_metadata():
    compact, streams = extension.compact_checkpoint(normalized_checkpoint())
    assert len(compact["prompts"]) == 128
    assert compact["prompts"]["0"]["draws"] == [["valid"] + [None] * 7] * 4
    assert streams[0]["nominal_child_seeds"] == list(range(100, 108))
    assert "collision" not in compact


def test_unexpected_request_seed_branch_is_explicitly_rejected():
    cp = normalized_checkpoint(); cp["draws"][0]["prompts"][0]["request_seeds_by_option"] = [999]
    with pytest.raises(ValueError, match="request-seed branch"):
        extension.compact_checkpoint(cp)


def test_original_loader_default_uses_preserved_initial_census():
    assert samples.DEFAULT_SNAPSHOT.name == "training_curve_snapshot_initial.json"
    manifest = samples.load_primary_manifest()
    assert next(r["n"] for r in manifest["cohort_counts"] if r["level"] == "level1" and r["objective"] == "maxrl") == 67
