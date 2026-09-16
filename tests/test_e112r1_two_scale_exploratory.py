from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper/results/e112r1_two_scale_exploratory_results.json"
FIGURE = ROOT / "paper/figures/verified_support_discovery_two_scale_effects.json"
MAIN = ROOT / "paper/main.tex"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_e112_two_scale_result_has_exact_integrity_valid_membership():
    result = _load(RESULT)
    design = result["design"]

    assert result["schema"] == "e112r1_two_scale_exploratory_results_v1"
    assert result["pointmaze"] == "excluded"
    assert design["frozen_cells"] == 50
    assert design["cells"] == 49
    assert design["qwen3b_cells_included"] == 0
    assert design["integrity_exclusions"] == [
        {
            "scale": "falcon1b",
            "domain": "countdown",
            "seed": 59,
            "comparator_job_id": 30269051,
            "reason": (
                "conflicting duplicate sampled rows in the registered job"
            ),
            "source_log": (
                "var/data/xdr_falcon3_1b_instruct_verified_first_replay_"
                "rehearsal_only_e79_falcon_aligned_countdown_replay_s59/"
                "debug_job30269051/eval_mode_coverage_draws.jsonl"
            ),
            "source_log_sha256": (
                "2f6e2ea3419c617dcedafd0ce2f7da7734ead2ad19c73056c0ae579252d64465"
            ),
            "outcome_value_selected": False,
        }
    ]

    for domain, family in result["families"]["qwen05b"].items():
        assert domain in design["domains"]
        assert family["paired_seeds"] == [43, 44, 45, 46, 47]
        assert all(summary["n"] == 5 for summary in family["paired_effects"].values())
    for domain, family in result["families"]["falcon1b"].items():
        expected = [55, 56, 57, 58] if domain == "countdown" else [55, 56, 57, 58, 59]
        assert family["paired_seeds"] == expected
        assert all(
            summary["n"] == len(expected)
            for summary in family["paired_effects"].values()
        )


def test_e112_disclosure_and_provenance_are_fail_closed():
    result = _load(RESULT)
    disclosure = result["disclosure"]

    assert disclosure["confirmatory"] is False
    assert disclosure["continuous_outcome_blindness"] is False
    assert disclosure["prior_private_looks_terminal_cells"] == [14, 33, 50]
    assert disclosure["campaign_mutation_allowed_from_outcomes"] is False
    assert disclosure["integrity_valid_pairs"] == 49
    assert disclosure["excluded_frozen_pairs"] == 1
    assert "conflicting duplicate" in disclosure["trajectory_auc"]

    for path_key, digest_key in (
        ("analysis_builder", "analysis_builder_sha256"),
        ("registered_full_builder", "registered_full_builder_sha256"),
        ("shared_endpoint_builder", "shared_endpoint_builder_sha256"),
        ("analysis_plotter", "analysis_plotter_sha256"),
        ("response_free_identity_erratum", "response_free_identity_erratum_sha256"),
        ("integrity_amendment", "integrity_amendment_sha256"),
    ):
        path = Path(result["analysis_provenance"][path_key])
        assert path.is_file()
        assert _sha256(path) == result["analysis_provenance"][digest_key]


def test_e112_figure_and_paper_match_result():
    result = _load(RESULT)
    figure = _load(FIGURE)
    main = MAIN.read_text(encoding="utf-8")

    assert figure["input_sha256"] == _sha256(RESULT)
    assert len(figure["cells"]) == 10
    assert sum(cell["n"] for cell in figure["cells"]) == 49
    assert [
        (cell["scale"], cell["domain"], cell["n"])
        for cell in figure["cells"]
        if cell["n"] != 5
    ] == [("falcon1b", "countdown", 4)]
    assert figure["estimand_scope"] == {
        "kind": "bundled verified-support proposal-pressure-replay contrast",
        "component_isolated": False,
        "models": ["Qwen2.5-0.5B", "Falcon3-1B"],
        "pair_count": 49,
        "qwen3b_analyzed": False,
    }
    assert "fig:verified-support-two-scale-effects" in main
    assert "49 matched endpoints" in main
    assert "Qwen2.5-3B discovery effects" in " ".join(main.split())
