"""The cohort registry is the single source of truth for what exists.

Four times a cohort was launched and then failed to appear where it belonged --
a semantic arm missing from Figure 4, another from the campaign table, a third
from the 3B row, a fourth from the Falcon row -- because the plotter and the
status table each carried a hand-maintained list. These tests make that class of
omission a test failure instead of a silently missing line.
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load(name: str, path: Path):
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # frozen dataclasses resolve their own module out of sys.modules, so it has
    # to be registered before the body executes
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry():
    return load("cohorts_under_test", EXP / "cohorts.py")


# --------------------------------------------------------------------------
# the guard that actually prevents the bug
# --------------------------------------------------------------------------


def test_every_released_ledger_is_registered_or_explicitly_excluded(registry):
    missing = registry.unregistered_released_ledgers()
    assert not missing, (
        "these released cohorts are not in the registry, so they are invisible "
        f"to the figure and the campaign table: {missing}. Add them to "
        "REGISTRY, or to EXCLUDED with the reason."
    )


def test_excluded_ledgers_carry_a_reason(registry):
    for name, reason in registry.EXCLUDED.items():
        assert reason.strip(), f"{name} is excluded without a stated reason"


def test_live_rlep_monitor_uses_e98r1_and_retains_e98_failure(registry):
    live = registry.by_tag("e98r1")

    assert live.ledger == "e98r1_sparse_rlep_dr_05b_jobs.json"
    assert "sparse RLEP-Dr" in live.label
    reason = registry.EXCLUDED["e98_rlep_dr_05b_jobs.json"]
    assert "failed" in reason
    assert "E98-R1" in reason


# --------------------------------------------------------------------------
# a registered cohort has to be usable
# --------------------------------------------------------------------------


def test_registry_entries_are_internally_coherent(registry):
    tags = [c.tag for c in registry.REGISTRY]
    assert len(tags) == len(set(tags)), "duplicate cohort tag"
    ledgers = [c.ledger for c in registry.REGISTRY]
    assert len(ledgers) == len(set(ledgers)), "duplicate ledger"
    for c in registry.REGISTRY:
        assert c.kind in {"paired", "semantic", "replay_dose", "point_maze", "repair"}
        if c.family is not None:
            assert c.family in registry.FAMILIES, f"{c.tag}: unknown family"


def test_semantic_arms_declare_a_family_arm_and_comparator(registry):
    # The comparator is load-bearing: an arm added on top of replay must be read
    # against replay, one added on top of control against control. A missing or
    # wrong comparator reports two interventions as one.
    for c in registry.REGISTRY:
        if c.kind not in registry.OVERLAY_KINDS:
            continue
        assert c.family, f"{c.tag}: overlay arm without a family row"
        assert c.arm, f"{c.tag}: overlay arm without an arm key"
        assert c.comparator in {
            "replay",
            "control",
        }, f"{c.tag}: comparator must be replay or control, got {c.comparator!r}"


def test_a_cohort_left_off_the_figure_has_to_say_why(registry):
    for c in registry.REGISTRY:
        if c.kind in registry.OVERLAY_KINDS and not c.plotted:
            assert c.not_plotted_because.strip(), (
                f"{c.tag} is excluded from the figure without a stated reason; "
                "silence here is how an arm goes missing by accident"
            )


def test_every_plotted_arm_has_a_drawing_style(registry):
    # Read the style keys from source: the plotters import matplotlib, which the
    # test interpreter does not carry.
    import ast

    source = (EXP / "plot_e78_figure4_preview.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    styles: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ARM_STYLE" for t in node.targets
        ):
            for key in node.value.keys:
                if isinstance(key, ast.Constant):
                    styles.add(str(key.value))
    assert styles, "could not read ARM_STYLE from the preview plotter"
    for c in registry.REGISTRY:
        if c.kind in registry.OVERLAY_KINDS and c.plotted:
            assert (
                c.arm in styles
            ), f"{c.tag} is marked plotted but {c.arm!r} has no ARM_STYLE entry"


# --------------------------------------------------------------------------
# the consumers must actually read the registry
# --------------------------------------------------------------------------


def test_campaign_stats_is_derived_from_the_registry(registry):
    stats = load("campaign_stats_under_test", EXP / "campaign_stats.py")
    assert len(stats.COHORTS) == len(registry.REGISTRY)
    assert {row[1] for row in stats.COHORTS} == {c.ledger for c in registry.REGISTRY}
    point = {row[1] for row in stats.COHORTS if row[2] == "point"}
    # Reader selection follows resolved_reader(), not kind: a semantic arm on an
    # interactive domain has a semantic role and a point-shaped ledger, and
    # asserting against kind here is what let that cohort report zero steps.
    assert point == {
        c.ledger for c in registry.REGISTRY if c.resolved_reader() == "point"
    }


def test_e104_e106_are_visible_as_the_effective_fifteen_cell_gate(registry):
    e104 = registry.by_tag("e104")
    e106 = registry.by_tag("e106")

    assert e104.resolved_reader() == "smoke"
    assert e104.excluded_domains == ("python_factors",)
    assert e106.resolved_reader() == "smoke"
    assert not e106.excluded_domains
    assert e104.ledger == ("e104_group_centered_semantic_repair_three_scale_jobs.json")
    assert e106.ledger == ("e106_python_lambda_normalization_three_scale_jobs.json")


def test_e105_is_registered_before_or_after_release(registry):
    e105 = registry.by_tag("e105")
    assert e105.ledger == (
        "e105_group_centered_semantic_repair_full_three_scale_jobs.json"
    )
    assert e105.resolved_reader() == "static"
    assert e105.path().name == e105.ledger


def test_e109_repaired_python_comparator_remains_registered_after_release(registry):
    e109 = registry.by_tag("e109")
    assert e109.ledger == ("e109_repaired_python_replay_comparators_jobs.json")
    assert e109.resolved_reader() == "static"
    assert e109.path().name == e109.ledger


def test_e117r1_effective_preflight_is_registered_and_original_is_excluded(
    registry,
):
    effective = registry.by_tag("e117r1")
    assert effective.ledger == ("e117r1_same_plumbing_component_preflight_jobs.json")
    assert effective.resolved_reader() == "smoke"
    assert "e117_same_plumbing_component_preflight_jobs.json" in (registry.EXCLUDED)
    assert (
        "zero runtime"
        in registry.EXCLUDED["e117_same_plumbing_component_preflight_jobs.json"]
    )


def test_e117r2_repaired_preflight_is_registered_and_monitorable(
    registry,
    monkeypatch,
):
    repaired = registry.by_tag("e117r2")
    assert repaired.ledger == ("e117r2_same_plumbing_component_preflight_jobs.json")
    assert repaired.resolved_reader() == "smoke"

    stats = load("campaign_stats_e117r2", EXP / "campaign_stats.py")
    ledger = json.loads(repaired.path().read_text(encoding="utf-8"))
    job_ids = [int(run["job_id"]) for run in ledger["runs"]]
    assert ledger["released"] is True
    recovery = ledger.get("signal53_batch_wave_recovery")
    if recovery is None:
        assert job_ids == list(range(30970803, 30970815))
    else:
        assert job_ids[0] == 30970803
        assert job_ids[1:] == recovery["replacement_job_ids"]
        assert list(map(int, recovery["mapping"])) == list(range(30970804, 30970815))
        assert list(recovery["mapping"].values()) == job_ids[1:]

    monkeypatch.setattr(
        stats.shared,
        "scheduler_states",
        lambda queried: {job_id: "PENDING" for job_id in queried},
    )
    monkeypatch.setattr(stats.shared, "run_step", lambda _run_dir: 0)
    monkeypatch.setattr(
        stats.shared,
        "is_complete",
        lambda _run_dir, _step, _target: False,
    )

    row = stats.cohort_row(
        repaired.label,
        repaired.path(),
        repaired.resolved_reader(),
    )
    assert row is not None
    assert row["cells"] == 12
    assert row["terminal"] == 0
    assert row["running"] == 0
    assert row["pending"] == 12
    assert row["failed"] == 0
    assert row["realized"] == 0
    assert row["total"] == 768
    assert row["scale_breakdown"]["qwen05b"]["cells"] == 9
    assert row["scale_breakdown"]["falcon1b"]["cells"] == 3


def test_e111_verified_support_gate_is_registered_before_release(registry):
    e111 = registry.by_tag("e111")
    assert e111.ledger == ("e111_verified_support_discovery_mechanism_gate_jobs.json")
    assert e111.resolved_reader() == "smoke"
    assert e111.path().name == e111.ledger


def test_e112_corrected_full_evaluation_is_registered_before_release(registry):
    e112 = registry.by_tag("e112")
    assert e112.ledger == ("e112_verified_support_discovery_full_three_scale_jobs.json")
    assert e112.resolved_reader() == "static"
    assert "failed sampler-contract" in e112.label
    assert e112.path().name == e112.ledger

    replacement = registry.by_tag("e112r1")
    assert replacement.ledger == (
        "e112r1_verified_support_discovery_full_three_scale_jobs.json"
    )
    assert replacement.resolved_reader() == "static"
    assert "E112-R1" in replacement.label
    assert replacement.path().name == replacement.ledger


def test_e113_dapo_is_registered_for_campaign_monitoring(registry):
    e113 = registry.by_tag("e113")
    assert e113.ledger == "e113_dapo_direct_baseline_jobs.json"
    assert e113.resolved_reader() == "static"
    assert e113.family is None
    assert "DAPO" in e113.label
    assert e113.path().name == e113.ledger


def test_e113r1_recovery_smokes_are_registered_without_science_claim(registry):
    e113r1 = registry.by_tag("e113r1")
    assert e113r1.ledger == "e113r1_dapo_recovery_smoke_jobs.json"
    assert e113r1.resolved_reader() == "static"
    assert e113r1.plotted is False
    assert "zero scientific cells" in e113r1.not_plotted_because


def test_e113r3_full_relaunch_is_registered_as_a_50_cell_gate(registry):
    e113r3 = registry.by_tag("e113r3")
    assert e113r3.ledger == "e113r3_dapo_full_relaunch_jobs.json"
    assert e113r3.resolved_reader() == "static"
    assert e113r3.plotted is False
    assert "50-cell" in e113r3.not_plotted_because


def test_e113r4_official_verl_dapo_is_registered_as_50_science_cells(registry):
    e113r4 = registry.by_tag("e113r4")
    assert e113r4.ledger == "e113r4_official_verl_dapo_jobs.json"
    assert e113r4.resolved_reader() == "static"
    assert e113r4.plotted is False
    assert "unmodified upstream" in e113r4.not_plotted_because


def test_closed_e113r2_gate_is_not_active(registry):
    assert "e113r2" not in {cohort.tag for cohort in registry.REGISTRY}
    assert (
        ROOT / "paper/preregistration/e113r2_dapo_full_relaunch_20260819.md"
    ).is_file()


def test_e113r1m1_memory_recovery_is_registered_without_science(registry):
    repair = registry.by_tag("e113r1m1")
    assert repair.ledger == "e113r1m1_qwen_memory_recovery_jobs.json"
    assert repair.plotted is False
    assert "zero scientific cells" in repair.not_plotted_because


def test_e113_launch_gates_are_visible_but_excluded_from_cells(
    tmp_path,
    monkeypatch,
):
    stats = load("campaign_stats_e113_gates", EXP / "campaign_stats.py")
    ledger = tmp_path / "e113.json"
    ledger.write_text(
        json.dumps(
            {
                "smokes": {
                    "qwen05b": {
                        "job_id": 101,
                        "run_dir": "/runs/qwen-smoke",
                        "max_train": 32,
                    },
                    "falcon1b": {
                        "job_id": 102,
                        "run_dir": "/runs/falcon-smoke",
                        "max_train": 32,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stats.shared,
        "scheduler_states",
        lambda _job_ids: {101: "FAILED", 102: "PENDING"},
    )
    monkeypatch.setattr(
        stats.shared,
        "run_step",
        lambda _run_dir: 0,
    )
    monkeypatch.setattr(
        stats.shared,
        "is_complete",
        lambda _run_dir, step, target: step >= target,
    )

    gates = stats.load_launch_gates(ledger)
    assert gates == {"total": 2, "terminal": 0, "running": 0, "pending": 1, "failed": 1}
    row = {
        "label": "E113 DAPO",
        "cells": 50,
        "terminal": 0,
        "running": 0,
        "pending": 50,
        "realized": 0,
        "total": 153_600,
        "depth": 0.0,
        "passes": 8,
        "launch_gates": gates,
    }
    retired = {
        "label": "E105 retired",
        "cells": 75,
        "terminal": 24,
        "running": 0,
        "pending": 0,
        "realized": 73_799,
        "total": 230_400,
        "depth": 2.56,
        "passes": 8,
        "retired": True,
    }
    console = stats.render([row, retired], markdown=False)
    markdown = stats.render([row, retired], markdown=True)
    assert "E113 DAPO" in console
    assert "GATE  E113 DAPO: 0/2 terminal; 0 running; 1 pending; 1 failed" in console
    assert "Launch smoke gates" in markdown
    assert "RETIRED  " not in console
    assert "Retired campaigns" not in markdown
    assert "TOTAL" in console and "    50" in console

    completed_gate = {
        **row,
        "launch_gates": {
            "total": 2,
            "terminal": 2,
            "running": 0,
            "pending": 0,
            "failed": 0,
        },
    }
    completed_console = stats.render([completed_gate], markdown=False)
    completed_markdown = stats.render([completed_gate], markdown=True)
    assert "launch smoke gates" not in completed_console
    assert "GATE  E113 DAPO" not in completed_console
    assert "Launch smoke gates" not in completed_markdown
    assert "E113 DAPO" in completed_console
    assert "E113 DAPO" in completed_markdown

    history = stats.render(
        [row, retired],
        markdown=False,
        include_history=True,
    )
    assert "GATE  E113 DAPO: 0/2 terminal; 0 running; 1 pending; 1 failed" in history
    assert "RETIRED  E105 retired: 24/75 terminal" in history


def test_smoke_only_recovery_ledger_is_a_zero_cell_monitor_row(
    tmp_path,
    monkeypatch,
):
    stats = load("campaign_stats_e113r1_smoke_only", EXP / "campaign_stats.py")
    ledger = tmp_path / "e113r1.json"
    ledger.write_text(
        json.dumps(
            {
                "runs": [],
                "scientific_cells": 0,
                "target_steps": 32,
                "passes": 1,
                "smokes": {
                    "qwen05b": {
                        "job_id": 201,
                        "run_dir": "/runs/qwen-r1",
                        "max_train": 32,
                    },
                    "falcon1b": {
                        "job_id": 202,
                        "run_dir": "/runs/falcon-r1",
                        "max_train": 32,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stats.shared,
        "scheduler_states",
        lambda _job_ids: {201: "RUNNING", 202: "PENDING"},
    )
    monkeypatch.setattr(stats.shared, "run_step", lambda _run_dir: 0)
    monkeypatch.setattr(
        stats.shared,
        "is_complete",
        lambda _run_dir, step, target: step >= target,
    )

    row = stats.cohort_row("E113-R1 DAPO recovery", ledger, "static")
    assert row is not None
    assert row["cells"] == 0
    assert row["total"] == 0
    assert row["launch_gates"] == {
        "total": 2,
        "terminal": 0,
        "running": 1,
        "pending": 1,
        "failed": 0,
    }
    assert "E113-R1 DAPO recovery" not in stats.render([row], markdown=False)
    assert "E113-R1 DAPO recovery" in stats.render(
        [row],
        markdown=False,
        include_history=True,
    )


def test_r4_gate_cancellations_are_awaiting_replacement_not_failures(
    tmp_path,
    monkeypatch,
):
    stats = load("campaign_stats_e113r4_gate_recovery", EXP / "campaign_stats.py")
    ledger = tmp_path / "e113r4.json"
    frozen_runs = [{"job_id": 300 + index, "state": "CANCELLED"} for index in range(50)]
    ledger.write_text(
        json.dumps(
            {
                "target_steps": 24,
                "passes": 1,
                "runs": frozen_runs,
                "vllm_scheduler_recovery": {
                    "science_replacements_submitted": False,
                    "superseded_runs_pending_replacement": frozen_runs,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stats.shared,
        "load_snapshot",
        lambda _ledger: (_ for _ in ()).throw(
            AssertionError("zero-runtime superseded jobs must not be read")
        ),
    )

    row = stats.cohort_row("E113-R4 DAPO", ledger, "static")
    assert row is not None
    assert row["cells"] == 50
    assert row["terminal"] == 0
    assert row["running"] == 0
    assert row["pending"] == 50
    assert row["failed"] == 0
    assert row["realized"] == 0
    assert row["total"] == 1_200


def test_scientific_failures_are_not_collapsed_into_pending(tmp_path, monkeypatch):
    stats = load("campaign_stats_science_failure", EXP / "campaign_stats.py")
    ledger = tmp_path / "science.json"
    ledger.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        stats.shared,
        "load_snapshot",
        lambda _ledger: {
            "rows": [
                {"state": "FAILED", "step": 0},
                {"state": "PENDING", "step": 0},
            ],
            "target": 32,
            "steps_per_pass": 32,
            "passes": 1,
        },
    )
    monkeypatch.setattr(stats, "load_launch_gates", lambda _ledger: None)
    row = stats.cohort_row("DAPO science", ledger, "static")
    assert row is not None
    assert row["cells"] == 2
    assert row["pending"] == 1
    assert row["failed"] == 1
    console = stats.render([row], markdown=False)
    markdown = stats.render([row], markdown=True)
    assert " fail " in console
    assert "| failed |" in markdown


def test_transient_scheduler_states_remain_visible_as_running(tmp_path, monkeypatch):
    stats = load("campaign_stats_transient_states", EXP / "campaign_stats.py")
    ledger = tmp_path / "science.json"
    ledger.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        stats.shared,
        "load_snapshot",
        lambda _ledger: {
            "rows": [
                {"state": "CONFIGURING", "step": 0},
                {"state": "COMPLETING", "step": 3},
            ],
            "target": 32,
            "steps_per_pass": 32,
            "passes": 1,
        },
    )
    monkeypatch.setattr(stats, "load_launch_gates", lambda _ledger: None)
    row = stats.cohort_row("DAPO science", ledger, "static")
    assert row is not None
    assert row["cells"] == 2
    assert row["running"] == 2
    assert row["pending"] == 0
    assert row["failed"] == 0


def test_incomplete_multiscale_progress_is_stratified_and_rendered(
    tmp_path,
    monkeypatch,
):
    stats = load("campaign_stats_scale_breakdown", EXP / "campaign_stats.py")
    ledger = tmp_path / "multiscale.json"
    runs = [
        {"job_id": job_id, "scale": scale}
        for job_id, scale in (
            (1, "qwen05b"),
            (2, "qwen05b"),
            (3, "qwen3b"),
            (4, "qwen3b"),
            (5, "falcon1b"),
            (6, "falcon1b"),
        )
    ]
    ledger.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    monkeypatch.setattr(
        stats.shared,
        "load_snapshot",
        lambda _ledger: {
            "rows": [
                {"job_id": 1, "state": "COMPLETED", "step": 10},
                {"job_id": 2, "state": "RUNNING", "step": 6},
                {"job_id": 3, "state": "PENDING", "step": 0},
                {"job_id": 4, "state": "PENDING", "step": 0},
                {"job_id": 5, "state": "COMPLETED", "step": 10},
                {"job_id": 6, "state": "COMPLETED", "step": 10},
            ],
            "target": 10,
            "steps_per_pass": 5,
            "passes": 2,
        },
    )
    monkeypatch.setattr(stats, "load_launch_gates", lambda _ledger: None)

    row = stats.cohort_row("multiscale", ledger, "static")
    assert row is not None
    assert row["terminal"] == 3
    assert row["running"] == 1
    assert row["pending"] == 2
    assert row["scale_breakdown"] == {
        "falcon1b": {
            "cells": 2,
            "terminal": 2,
            "running": 0,
            "pending": 0,
            "failed": 0,
            "realized": 20,
            "total": 20,
            "depth": 2.0,
            "passes": 2,
        },
        "qwen05b": {
            "cells": 2,
            "terminal": 1,
            "running": 1,
            "pending": 0,
            "failed": 0,
            "realized": 16,
            "total": 20,
            "depth": 1.6,
            "passes": 2,
        },
        "qwen3b": {
            "cells": 2,
            "terminal": 0,
            "running": 0,
            "pending": 2,
            "failed": 0,
            "realized": 0,
            "total": 20,
            "depth": 0.0,
            "passes": 2,
        },
    }
    for markdown in (False, True):
        rendered = stats.render([row], markdown=markdown)
        assert "scale coverage" in rendered.lower()
        assert "qwen3b 0/2 terminal" in rendered
        assert "falcon1b 2/2 terminal" not in rendered
        history = stats.render([row], markdown=markdown, include_history=True)
        assert "falcon1b 2/2 terminal" in history

    # A held pending job does not revive a stopped cohort's scale details.
    # Historical coverage and the underlying cohort totals remain available.
    for tag in ("e117s1", "e113r4"):
        stopped = dict(row, label=stats.registry.by_tag(tag).label, running=0)
        for markdown in (False, True):
            rendered = stats.render([stopped], markdown=markdown)
            assert stopped["label"] in rendered
            assert "scale coverage" not in rendered.lower()
            history = stats.render([stopped], markdown=markdown, include_history=True)
            assert "scale coverage" in history.lower()
            resumed = stats.render([dict(stopped, running=1)], markdown=markdown)
            assert "scale coverage" in resumed.lower()


def test_full_r3_release_retires_only_an_exact_50_cell_successor(tmp_path):
    stats = load("campaign_stats_e113r3_retirement", EXP / "campaign_stats.py")
    ledger = tmp_path / "e113r3.json"
    runs = [
        {"model_family": family, "domain": domain, "seed": seed}
        for family, seeds in {
            "qwen05b": range(43, 48),
            "falcon1b": range(55, 60),
        }.items()
        for domain in (
            "graph_coloring",
            "countdown",
            "python_factors",
            "mathir",
            "pantry_plan",
        )
        for seed in seeds
    ]
    ledger.write_text(
        json.dumps(
            {
                "schema": "e113r3_dapo_full_relaunch_jobs_v1",
                "released": True,
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    assert stats.e113_r3_release(ledger) is not None
    runs.pop()
    ledger.write_text(
        json.dumps(
            {
                "schema": "e113r3_dapo_full_relaunch_jobs_v1",
                "released": True,
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    assert stats.e113_r3_release(ledger) is None


def test_smoke_reader_materializes_effective_e104_plus_e106_rows(
    registry,
    monkeypatch,
):
    stats = load("campaign_stats_smoke_reader", EXP / "campaign_stats.py")
    monkeypatch.setattr(
        stats.shared,
        "scheduler_states",
        lambda job_ids: {job_id: "PENDING" for job_id in job_ids},
    )
    monkeypatch.setattr(stats.shared, "run_step", lambda _run_dir: 0)
    monkeypatch.setattr(
        stats.shared,
        "is_complete",
        lambda _run_dir, _step, _target: False,
    )

    e104 = registry.by_tag("e104")
    e106 = registry.by_tag("e106")
    non_python = stats.load_smoke_snapshot(
        e104.path(),
        excluded_domains=e104.excluded_domains,
    )
    repaired_python = stats.load_smoke_snapshot(e106.path())

    assert len(non_python["rows"]) == 12
    assert {row["domain"] for row in non_python["rows"]} == {
        "graph_coloring",
        "countdown",
        "mathir",
        "pantry_plan",
    }
    assert len(repaired_python["rows"]) == 3
    assert {row["domain"] for row in repaired_python["rows"]} == {"python_factors"}
    assert non_python["target"] == repaired_python["target"] == 64


def test_e108_smoke_reader_reports_all_eight_passes(registry, monkeypatch):
    stats = load("campaign_stats_e108_smoke_reader", EXP / "campaign_stats.py")
    e108 = registry.by_tag("e108")
    ledger = json.loads(e108.path().read_text(encoding="utf-8"))
    target = int(ledger["target_steps"])

    monkeypatch.setattr(stats.shared, "scheduler_states", lambda job_ids: {})
    monkeypatch.setattr(stats.shared, "run_step", lambda _run_dir: target)
    monkeypatch.setattr(
        stats.shared,
        "is_complete",
        lambda _run_dir, _step, _target: True,
    )

    snapshot = stats.load_smoke_snapshot(e108.path())
    assert snapshot["target"] == 64
    assert snapshot["passes"] == 8
    assert snapshot["steps_per_pass"] == 8
    row = stats.cohort_row(e108.label, e108.path(), "smoke")
    assert row is not None
    assert row["depth"] == 8.0


def test_the_plotter_attaches_arms_from_the_registry_not_a_local_list():
    source = (EXP / "plot_figure4_with_falcon_preview.py").read_text(encoding="utf-8")
    assert "registry.semantic_arms(" in source, (
        "the plotter must drive attachment from the registry; a hand-written "
        "call site per family is what caused four missing arms"
    )
    # no stray per-cohort ledger constants feeding attachment by hand
    assert "E87_LEDGER" not in source
    assert source.count("_attach_semantic(") == 2  # the definition and the loop


def test_semantic_arms_lookup_partitions_by_family(registry):
    seen = []
    for family in registry.FAMILIES:
        arms = registry.semantic_arms(family)
        for arm in arms:
            assert arm.family == family
            assert arm.plotted
        seen.extend(a.tag for a in arms)
    plotted = {
        c.tag
        for c in registry.REGISTRY
        if c.kind in registry.OVERLAY_KINDS and c.plotted
    }
    assert set(seen) == plotted, "a plotted arm belongs to no family row"


def test_repair_lookup_resolves_the_cohorts_it_supersedes(registry):
    for tag in ("e81", "e82", "e83"):
        repair = registry.repair_for(tag)
        assert repair is not None and repair.kind == "repair", (
            f"{tag}'s PantryPlan cells are superseded but no repair cohort "
            "resolves for it, so the figure would redraw the inert runs"
        )
    assert registry.repair_for("e78") is None


# --------------------------------------------------------------------------
# Label honesty
#
# E88 and E89 shipped as "adaptive MaxEnt" and "adaptive rho=.015" while their
# objective is adaptive_open_set_semantic_maxent_ON_VERIFIED_REPLAY. In the same
# table, "MaxEnt only" (E83/E86) means *without* replay, so those labels read as
# the no-replay adaptive arm -- the opposite of what ran. The registry already
# encodes the fact needed to catch this: an arm added on top of replay is read
# against replay.


def test_arms_built_on_replay_say_so_in_their_label(registry):
    offenders = [
        c.label
        for c in registry.REGISTRY
        if c.comparator == "replay" and "replay" not in c.label.lower()
    ]
    assert not offenders, (
        "these cohorts apply verified replay but their campaign label does not "
        f"say so, which reads as the without-replay arm: {offenders}"
    )


# --------------------------------------------------------------------------
# Reader selection
#
# `kind` says what a cohort is for the figure; it does not say how its ledger
# records progress. A semantic arm on an interactive domain has a semantic role
# but writes an append-only metrics file rather than an oat run directory, and
# reading it with the static reader silently reported a running cohort as zero
# steps. The ledger's own shape is the ground truth, so assert against it.


def test_reader_matches_how_each_ledger_records_progress(registry):
    import json
    from pathlib import Path

    artifacts = Path(__file__).resolve().parents[1] / "var/artifacts"
    wrong = []
    for cohort in registry.REGISTRY:
        path = artifacts / cohort.ledger
        if not path.is_file():
            continue
        try:
            runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or []
        except ValueError:
            continue
        if not runs:
            continue
        records_metrics_file = all("metrics_path" in run for run in runs)
        reader = cohort.resolved_reader()
        if records_metrics_file and reader != "point":
            wrong.append(f"{cohort.tag}: runs carry metrics_path but reader={reader!r}")
        if not records_metrics_file and reader == "point":
            wrong.append(f"{cohort.tag}: runs carry no metrics_path but reader='point'")
    assert not wrong, (
        "these cohorts would be read with the wrong reader and report zero "
        f"progress while running: {wrong}"
    )


def test_reader_defaults_from_kind_when_unset(registry):
    for cohort in registry.REGISTRY:
        if cohort.reader is None:
            expected = "point" if cohort.kind == "point_maze" else "static"
            assert cohort.resolved_reader() == expected
