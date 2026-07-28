"""Narrative and evidence contracts for the ModeBench-centered manuscript."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/main.tex"
RESULT = ROOT / "paper/results/modebench_long_horizon_interim.json"
PYTHON_IDENTITY = (
    ROOT / "paper/results/python_factor_modebench_v1_identity.json"
)


def test_main_paper_centers_modebench_without_internal_experiment_ids():
    text = MAIN.read_text(encoding="utf-8")
    main_text = text.split(r"\bibliographystyle", maxsplit=1)[0]

    assert "ModeBench: Executable Outcome Discovery" in main_text
    assert r"\section{ModeBench}" in main_text
    assert r"\section{Online Verified Maximum Entropy}" in main_text
    assert "exactly 50" in main_text
    assert "no upper projection" in main_text
    assert "passive observer" in main_text
    assert re.search(r"\bE[0-9]+[A-Za-z-]*\b", text) is None


def test_paper_has_exactly_three_executable_modebench_domains():
    text = MAIN.read_text(encoding="utf-8")
    prose = " ".join(text.split())

    for required in (
        r"\subsection{Graph coloring}",
        r"\subsection{Countdown}",
        r"\subsection{Python factor functions}",
        "The three ModeBench environments",
        "Correctness and mode identity come from the same executed object",
    ):
        assert required in prose
    assert "16--3,600 modes per prompt" in prose
    assert "isolated external interpreter" in prose
    assert "separately frozen matched extension" in prose


def test_paper_uses_the_frozen_interim_result_without_calling_it_terminal():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    text = MAIN.read_text(encoding="utf-8")
    prose = " ".join(text.split())

    assert result["status"] == "FROZEN_INTERIM_PAIRED_COMMON_HORIZON"
    assert result["seeds"] == [43, 44, 45]
    assert result["domains"]["Graph coloring"]["paired_common_horizon_passes"] == 8.25
    assert result["domains"]["Countdown"]["paired_common_horizon_passes"] == 2.75
    assert ".343 to .635" in prose
    assert ".056 to .160" in prose
    assert "not terminal comparisons" in prose
    assert "study is still in progress" in prose


def test_python_dataset_identity_matches_the_manuscript_audit():
    identity = json.loads(PYTHON_IDENTITY.read_text(encoding="utf-8"))
    text = MAIN.read_text(encoding="utf-8")

    assert identity == {
        "case_count": 4,
        "eval_rows": 128,
        "eval_rows_sha256": (
            "be0f621c5a0ae84ca45ef4ab866ae12f4c64472eeed1a0ae4666918ab794d183"
        ),
        "external_verifier": "isolated_python_jsonl_worker",
        "max_value": 96,
        "maximum_exact_mode_count": 3600,
        "minimum_exact_mode_count": 16,
        "schema": "python_factor_modebench_v1",
        "seed": 5100,
        "support": "finite_exact",
        "train_rows": 384,
        "train_rows_sha256": (
            "bcfa9accfa3b5c7dd312c85683157af070e7f2741385e9b5a882fd57da037cf7"
        ),
    }
    assert "bcfa9accfa3b5c7d...a882fd57da037cf7" in text
    assert "be0f621c5a0ae84c...4666918ab794d183" in text


def test_design_history_records_core_decisions_not_scheduler_chronology():
    text = MAIN.read_text(encoding="utf-8")
    main_text = text.split(r"\bibliographystyle", maxsplit=1)[0].lower()

    for required in (
        "sequence entropy increased length and eos avoidance",
        "candidate-local normalization favored short samples",
        "novel wrong answers received diversity pressure",
        "control \\(h/\\log k\\)",
        "remove the upper projection",
    ):
        assert required in main_text
    assert "job " not in main_text
    assert "scheduler" not in main_text
    assert "queued" not in main_text
