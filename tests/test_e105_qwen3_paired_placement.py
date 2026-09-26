from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"
SCRIPT = EXP / "apply_e105_qwen3_paired_a6000_placement_amendment.py"
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e105_qwen3_paired_a6000_placement_amendment_20260817.md"
)


def load(name: str, path: Path):
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_amendment_is_prospective_paired_outcome_blind_and_static_only():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "before E105 submission" in text
    assert "both its control and replay jobs or neither" in text
    assert "complete E104+E106 mechanism gate" in text
    assert "did not satisfy its separate one-update replay-application" in text
    assert "inherits its" in text and "matched replay comparator's placement" in text
    assert "E109 amendment supersedes only the Python-comparator" in text
    assert "four historical E80-R1 Python jobs" in text
    assert "PointMaze remains excluded" in text


def test_candidate_set_moves_eight_historical_pairs_and_assigns_two_python_pairs():
    amendment = load("e105_q3_placement_candidates", SCRIPT)
    ledger = amendment.load(amendment.E80_LEDGER)
    pairs = amendment.candidate_pairs(ledger)

    assert set(pairs) == set(amendment.HISTORICAL_CANDIDATE_CELLS)
    assert len(pairs) == 8
    assert set(amendment.PROSPECTIVE_PYTHON_CELLS) == {
        ("python_factors", 73),
        ("python_factors", 74),
    }
    assert set(amendment.CANDIDATE_CELLS) == (
        set(amendment.HISTORICAL_CANDIDATE_CELLS)
        | set(amendment.PROSPECTIVE_PYTHON_CELLS)
    )
    assert all(set(pair) == {"control", "replay"} for pair in pairs.values())
    job_ids = {
        int(run["job_id"])
        for pair in pairs.values()
        for run in pair.values()
    }
    assert len(job_ids) == 16
    assert all(domain != "python_factors" for domain, _seed in pairs)
    assert all("pointmaze" not in domain for domain, _seed in pairs)


def test_original_and_amended_guards_preserve_scientific_environment():
    amendment = load("e105_q3_placement_guards", SCRIPT)
    ledger = amendment.load(amendment.E80_LEDGER)
    pair = amendment.candidate_pairs(ledger)[("pantry_plan", 73)]

    for run in pair.values():
        scientific = " ".join(amendment.scientific_needles(ledger, run))
        original = " ".join(
            (
                scientific,
                "JobState=PENDING",
                "RunTime=00:00:00",
                f"Partition={amendment.ORIGINAL_PARTITION}",
                f"ReqNodeList={amendment.ORIGINAL_NODE_LIST}",
                f"TresPerNode={amendment.ORIGINAL_GRES}",
            )
        )
        amended = " ".join(
            (
                scientific,
                "JobState=PENDING",
                "RunTime=00:00:00",
                f"Partition={amendment.TARGET_PARTITION}",
                f"ReqNodeList={amendment.TARGET_NODE_LIST}",
                f"TresPerNode={amendment.TARGET_GRES}",
            )
        )

        amendment.validate_original(ledger, run, original)
        amendment.validate_amended(ledger, run, amended)
        assert amendment.pending_at_zero(original)
        running = amended.replace("JobState=PENDING", "JobState=RUNNING").replace(
            "RunTime=00:00:00", "RunTime=00:07:00"
        )
        amendment.validate_recovered_amended(ledger, run, running)
        assert not amendment.pending_at_zero(running)

    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1" in (
        amendment.scientific_needles(ledger, pair["control"])
    )
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0" in (
        amendment.scientific_needles(ledger, pair["replay"])
    )


def test_only_matching_e105_qwen3_cell_changes_hardware():
    amendment = load("e105_q3_placement_routing_amendment", SCRIPT)
    launcher = amendment.e105
    root = launcher.repo_root()
    snapshot = (root / launcher.e106.SNAPSHOT).resolve()
    moved_cell = ("python_factors", 73)
    moved_run = next(
        run
        for run in launcher.references(root, "qwen3b")
        if (str(run["domain"]), int(run["seed"])) == moved_cell
    )
    stationary_run = next(
        run
        for run in launcher.references(root, "qwen3b")
        if (str(run["domain"]), int(run["seed"])) == ("python_factors", 72)
    )

    moved_env, _ = launcher.build_env(root, "qwen3b", moved_run, snapshot)
    moved = launcher.sbatch_command(
        root, "qwen3b", moved_run, moved_env, {moved_cell}
    )
    stationary_env, _ = launcher.build_env(
        root, "qwen3b", stationary_run, snapshot
    )
    stationary = launcher.sbatch_command(
        root, "qwen3b", stationary_run, stationary_env, {moved_cell}
    )

    assert "--partition=lowprio" in moved
    assert f"--nodelist={launcher.QWEN3_A6000_NODE_LIST}" in moved
    assert "--gres=gpu:a6000:1" in moved
    assert "--time=3-00:00:00" in moved
    assert "--partition=mltheory" in stationary
    assert "--nodelist=node302" in stationary
    assert "--gres=gpu:a100:1" in stationary
    assert moved_env == launcher.build_env(root, "qwen3b", moved_run, snapshot)[0]


def test_application_requires_the_complete_method_gate_before_any_update():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "e105.check_gate(ROOT)" in source
    assert source.index("e105.check_gate(ROOT)") < source.index("execute(command)")
    assert "if E105_LEDGER.exists()" in source
    assert '"outcome_metrics_inspected": False' in source
    assert '"scientific_configuration_changed": False' in source
    assert '"paired_a6000_cells"' in source
    assert "for domain, seed in HISTORICAL_CANDIDATE_CELLS" in source
    assert '"recovered_after_interrupted_apply": bool(recovered)' in source
    assert '"recovered_moved_pairs"' in source
    assert "e80.atomic_json(OUT, payload)" in source
    assert "e80.e81.atomic_json" not in source
