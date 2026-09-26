import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v5_protocol_freezes_source_repair_and_2304_records():
    text = (ROOT / "paper/preregistration/constructive_code_executable_slate_v5_20260730.md").read_text()
    for required in (
        "30201021",
        "52 held-out",
        "before any candidate program",
        "exactly 48",
        "2,304 submission-suite execution records",
        "No task, split, source revision",
    ):
        assert required in text


def test_v5_wrappers_preserve_v4_tasks_and_freeze_48_per_label():
    materialize = (ROOT / "ops/materialize_constructive_code_v5.py").read_text()
    replay = (ROOT / "ops/replay_constructive_code_v5.py").read_text()
    assert "V5_TASKS = v4.V4_TASKS" in materialize
    assert "base.REPLAYS_PER_LABEL = 48" in materialize
    assert "base.EXCLUDE_V1_HASHES = False" in materialize
    assert "base.REQUIRED_PER_LABEL = 48" in replay
    assert "audit.REQUIRED_PER_LABEL = 48" in replay
    assert 'schema_version="constructive-code-v5-gate-audit-v1"' in replay


def test_full_source_selector_can_freeze_48_unique_python3_programs():
    module = _module(ROOT / "ops/materialize_constructive_code_v2.py", "materialize_cc_v2_v5_test")
    ledger_path = ROOT / "var/data/constructive_code_review_slate_v1/359_b/python_replays.jsonl"
    rows = [__import__("json").loads(line) for line in ledger_path.read_text().splitlines()]
    submissions = [{"language": row["language"], "code": row["code"]} for row in rows]
    selected, counts = module.select_heldout_python3(submissions, set(), limit=48)
    assert len(selected) == 48
    assert counts["selected"] == 48


def test_v5_launcher_binds_failed_v4_and_exact_execution_count():
    launcher = (ROOT / "ops/exp_scaling/launch_constructive_code_v5_gate.sh").read_text()
    batch = (ROOT / "ops/slurm/constructive_code_v5_gate.slurm").read_text()
    assert "30201021" in launcher
    assert "only 52 held-out Python-3 programs; 64 required" in launcher
    assert '"expected_submission_suite_replays":2304' in launcher
    assert "materialize_constructive_code_v5.py" in batch
    assert "replay_constructive_code_v5.py" in batch
