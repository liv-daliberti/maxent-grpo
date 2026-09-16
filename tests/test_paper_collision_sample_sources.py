"""Raw-response admission must be stricter than equality of aggregate metrics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
from exp_scaling import load_paper_collision_samples as samples


def draw(step=0, index=0):
    return {"step": step, "draw_index": index, "evaluation_kind": samples.EVALUATION_KIND,
            "sample_count": 8, "temperature": 1.0, "top_p": 1.0, "benchmark": "multi_answer",
            "seed": 100 + index, "metrics": {"any_correct_at_k": 1.0},
            "prompts": [{"prompt_index": i, "prompt": f"task {i}",
                         "reference": json.dumps({"instance_id": str(i), "num_modes": 2}),
                         "answer_keys": ["valid", "failed-but-keyed"] + [None] * 6,
                         "rewards": [1.0] + [0.0] * 7, "request_seeds_by_option": [100 + index],
                         "option_ids": [None] * 8, "metrics": {}} for i in range(128)]}


def source(tmp_path, rows, suffix="source"):
    path = tmp_path / f"{suffix}.jsonl"
    raw = b"".join((json.dumps(r) + "\n").encode() for r in rows)
    path.write_bytes(raw)
    return path, {"path": str(path), "read_bytes": len(raw), "sha256_read_prefix": hashlib.sha256(raw).hexdigest()}


def test_failed_keys_are_retained_raw_but_never_verified():
    result = samples.normalize_draw(draw())
    prompt = result["prompts"][0]
    assert prompt["answer_keys"][1] == "failed-but-keyed"
    assert prompt["verified_keys"] == ["valid"] + [None] * 7
    assert prompt["request_seeds_by_option"] == [100]


@pytest.mark.parametrize("fault", ["positive_without_key", "wrong_k", "duplicate_prompt", "nan_reward", "changed_p"])
def test_invalid_response_grid_fails_closed(fault):
    row = draw()
    if fault == "positive_without_key":
        row["prompts"][0]["answer_keys"][0] = None
    elif fault == "wrong_k":
        row["prompts"][0]["rewards"].pop()
    elif fault == "duplicate_prompt":
        row["prompts"][1] = deepcopy(row["prompts"][0])
    elif fault == "nan_reward":
        row["prompts"][0]["rewards"][1] = float("nan")
    else:
        row["top_p"] = 0.9
    with pytest.raises(samples.SourceIntegrityError):
        samples.normalize_draw(row)


def test_frozen_prefix_ignores_appends_but_rejects_old_byte_changes(tmp_path):
    path, binding = source(tmp_path, [draw()])
    wanted = [{"path": str(path), "line": 1}]
    with path.open("ab") as f:
        f.write(b'{"later":true}\n')
    rows, checks = samples.read_frozen_sources([binding], wanted)
    assert rows[str(path), 1]["step"] == 0
    assert checks[0]["appended_bytes_ignored"] > 0
    raw = path.read_bytes().replace(b'"task 0"', b'"task X"', 1)
    path.write_bytes(raw)
    with pytest.raises(samples.SourceIntegrityError, match="hash differs"):
        samples.read_frozen_sources([binding], wanted)


def test_origins_cannot_escape_the_admitted_source_set(tmp_path):
    path, binding = source(tmp_path, [draw()])
    with pytest.raises(samples.SourceIntegrityError, match="outside frozen source"):
        samples.read_frozen_sources([binding], [{"path": str(tmp_path / "unregistered.jsonl"), "line": 1}])
    with pytest.raises(samples.SourceIntegrityError, match="outside frozen prefix"):
        samples.read_frozen_sources([binding], [{"path": str(path), "line": 2}])


def test_same_rows_and_four_separate_draw_seeds_are_certified():
    checkpoint = samples.assemble_checkpoint([samples.normalize_draw(draw(index=i)) for i in range(4)], step=0)
    certificate = checkpoint["sampling_certificate"]
    assert certificate["same_prompt_identities"]
    assert certificate["distinct_integer_draw_seeds"]
    assert certificate["draw_seeds"] == [100, 101, 102, 103]
    assert "not an independent proof" in certificate["scope"]


def test_changed_prompt_surface_prevents_pooling_even_if_reference_matches():
    rows = [draw(index=i) for i in range(4)]
    rows[1]["prompts"][0]["prompt"] = "different guidance"
    with pytest.raises(samples.SourceIntegrityError, match="identity or ordering differs"):
        samples.assemble_checkpoint([samples.normalize_draw(r) for r in rows], step=0)


def test_metric_equal_duplicate_with_different_responses_is_not_silently_chosen(tmp_path):
    rows = [draw(index=i) for i in range(4)]
    duplicate = deepcopy(rows[0])
    duplicate["prompts"][0]["answer_keys"][0] = "another-valid-mode"
    path, binding = source(tmp_path, rows + [duplicate])
    admitted = [{"draw_index": i, "metrics": row["metrics"],
                 "metadata": samples.normalize_draw(row)["metadata"],
                 "origins": [{"path": str(path), "line": i + 1}]} for i, row in enumerate(rows)]
    admitted[0]["origins"].append({"path": str(path), "line": 5})
    cell = {"level": "level1", "scale": "qwen05b", "domain": "graph_coloring", "method": "drgrpo",
            "seed": 43, "terminal_admitted": True, "terminal_matches_census": True,
            "complete_checkpoints": {"0": {"draws": admitted}}, "source_files": [binding]}
    loaded = samples._load_cell(cell, True, (0,))
    assert loaded["terminal_admitted"] is True
    assert loaded["in_terminal_paired_cohort"] is True
    assert loaded["checkpoints"]["0"] is None
    assert "conflicting raw response" in loaded["sample_issues"][0]["reason"]


def test_frozen_primary_cohorts_remain_seventyfour_and_sixtyseven():
    manifest = samples.load_primary_manifest()
    counts = {(r["level"], r["objective"]): r["n"] for r in manifest["cohort_counts"]}
    assert counts["level1", "drgrpo"] == 74
    assert counts["level1", "maxrl"] == 67
    assert len(manifest["snapshot"]["cells"]) == 400
