from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from oat_drgrpo.math_grader import validated_modebench_outcome_key
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "paper/preregistration/e59_mathir_global_verified_replay_05b.md"
)
SMOKE = (
    ROOT
    / "ops/exp_scaling/launch_e59_mathir_global_replay_smoke.sh"
)
MATCHED = ROOT / "ops/exp_scaling/launch_e59_mathir_matched.sh"
AUDITOR = (
    ROOT
    / "ops/exp_scaling/audit_e59_mathir_global_replay_smoke.py"
)
DATA = ROOT / "var/data/mathir_action_menu_v1"
REFRESH = ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py"
PLOT = ROOT / "ops/exp_scaling/plot_divergence.py"
MONITOR = ROOT / "ops/exp_scaling/monitor_campaign.py"


def test_e59_data_is_finite_executable_and_disjoint():
    identity = json.loads((DATA / "identity.json").read_text(encoding="utf-8"))
    train = load_from_disk(str(DATA / "train"))["train"]
    evaluation = load_from_disk(str(DATA / "eval"))["multi_answer"]

    assert identity["schema"] == "mathir_action_menu_v1"
    assert identity["support"] == "finite_exhaustively_enumerated"
    assert identity["mode_counts"] == [5]
    assert identity["training_bank_initialization"] == "empty"
    assert identity["gold_mode_catalogue_supplied_to_policy"] is False
    assert len(train) == 384
    assert len(evaluation) == 128
    assert set(train["answer_mode_count"]) == {5}
    assert set(evaluation["answer_mode_count"]) == {5}

    def row_identity(row):
        spec = json.loads(row["answer"])
        return spec["family"], tuple(sorted(spec["bindings"].items()))

    assert not (
        {row_identity(row) for row in train}
        & {row_identity(row) for row in evaluation}
    )
    specimen = dict(evaluation[0])
    spec = json.loads(specimen["answer"])
    assert len(enumerate_mathir_action_menu_keys(spec)) == 5


def test_e59_validator_not_labels_is_the_bank_boundary():
    evaluation = load_from_disk(str(DATA / "eval"))["multi_answer"]
    row = dict(evaluation[0])
    spec = json.loads(row["answer"])
    keys = enumerate_mathir_action_menu_keys(spec)

    assert len(keys) == 5
    assert validated_modebench_outcome_key(
        r"\boxed{I used route S2 and x=7}",
        row["answer"],
    ) is None
    assert "strategy" not in spec
    assert "routes" not in spec


def test_e59_is_a_smoke_gated_matched_extension_of_exact_e58_variant():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    smoke = SMOKE.read_text(encoding="utf-8")
    matched = MATCHED.read_text(encoding="utf-8")
    auditor = AUDITOR.read_text(encoding="utf-8")

    for literal in (
        "separately identified fourth-domain extension of E58",
        "Every row has exactly five",
        "does not relabel that probe as passing",
        "`verified_first_global_replay_canonical`",
        "exactly one model-discovered verified prompt bank",
        "ordinary Dr.GRPO",
    ):
        assert literal in protocol
    for surface in (smoke, matched):
        assert "OAT_ZERO_COMPARATIVE_TASK=math" in surface
        assert "OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome" in surface
        assert (
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1"
            in surface
        )
        assert "OAT_ZERO_MAXENT_ALPHA=0" in surface
        assert "OAT_ZERO_GENERATE_MAX_LENGTH=64" in surface
    assert "OAT_ZERO_ONLY_ARMS=grpo,verified_first_global_replay_canonical" in matched
    assert 'audit.get("status") != "pass"' in matched
    assert "pre-discovery" in auditor
    assert "checkpoint global replay scheduler mismatch" in auditor


def test_e59_is_added_to_the_shared_live_modebench_display():
    refresh = REFRESH.read_text(encoding="utf-8")
    plot = PLOT.read_text(encoding="utf-8")
    monitor = MONITOR.read_text(encoding="utf-8")
    prefix = "mie59_mathir_global_verified_replay_05b_50ep"
    identity = "e59_mathir_global_verified_replay_matched_identity.json"

    for surface in (refresh, plot, monitor):
        assert prefix in surface
        assert identity in surface
    assert "if E59_MATHIR_IDENTITY.is_file()" in refresh
    assert "+ E59_MATHIR_CELLS" in refresh
    assert "Executable MathIR — E59 action-menu eval " in plot
    assert "(not MATH-500; live frontier)" in plot
    assert "e59_mathir_global_verified_replay_05b_live" in plot
    assert "exact rational equation-state trajectory" in plot
    assert '"Executable MathIR (action-menu eval; not MATH-500)"' in monitor
    assert "MathIR keys come only from the exact rational equation-state" in monitor


def test_e59_is_a_separate_non_math500_standard_modebench_extension():
    standard_plot = (
        ROOT / "ops/plot_modebench_paper.py"
    ).read_text(encoding="utf-8")
    live_results = (
        ROOT / "ops/slurm/e59_mathir_live_results.slurm"
    ).read_text(encoding="utf-8")

    assert "Executable MathIR (held-out action menu; not MATH-500)" in standard_plot
    assert '"evaluation_dataset": "mathir_action_menu_v1/multi_answer"' in standard_plot
    assert '"is_math500": False' in standard_plot
    assert "verified_first_global_replay_canonical" in standard_plot
    assert "summary[\"extensions\"]" in standard_plot
    assert "ops/plot_modebench_paper.py" in live_results
