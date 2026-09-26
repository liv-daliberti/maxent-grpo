import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v4_protocol_freezes_full_source_py3_and_3072_records():
    text = (ROOT / "paper/preregistration/constructive_code_executable_slate_v4_20260730.md").read_text()
    for required in ("30187935", "31 held-out", "exactly 64", "3,072 execution records", "reports any overlap"):
        assert required in text


def test_v4_wrappers_preserve_v3_tasks_but_allow_v1_overlap():
    materialize = (ROOT / "ops/materialize_constructive_code_v4.py").read_text()
    replay = (ROOT / "ops/replay_constructive_code_v4.py").read_text()
    assert "V4_TASKS = v3.V3_TASKS" in materialize
    assert "base.REPLAYS_PER_LABEL = 64" in materialize
    assert "base.EXCLUDE_V1_HASHES = False" in materialize
    assert "base.REQUIRED_PER_LABEL = 64" in replay
    assert "base.REQUIRE_V1_DISJOINT = False" in replay


def test_full_source_selector_can_reuse_v1_python3_hashes():
    module = _module(ROOT / "ops/materialize_constructive_code_v2.py", "materialize_cc_v2_test")
    ledger_path = ROOT / "var/data/constructive_code_review_slate_v1/359_b/python_replays.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    submissions = [{"language": row["language"], "code": row["code"]} for row in rows]
    selected, counts = module.select_heldout_python3(submissions, set(), limit=50)
    assert len(selected) == 50
    assert counts["selected"] == 50


def test_v4_launcher_binds_failed_v3_and_exact_execution_count():
    launcher = (ROOT / "ops/exp_scaling/launch_constructive_code_v4_gate.sh").read_text()
    batch = (ROOT / "ops/slurm/constructive_code_v4_gate.slurm").read_text()
    assert "30187935" in launcher
    assert '"expected_submission_suite_replays":3072' in launcher
    assert "materialize_constructive_code_v4.py" in batch
    assert "replay_constructive_code_v4.py" in batch
