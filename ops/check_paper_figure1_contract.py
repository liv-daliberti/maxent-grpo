#!/usr/bin/env python3
"""Fail closed if any manuscript figure regresses from its accepted contract."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from functools import lru_cache
import re
import statistics
from pathlib import Path
import subprocess
from check_terminal_comparison_figures import check as check_terminal_figures
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ops/plot_paper_collapse_toy.py"
AUDIT = ROOT / "var/artifacts/paper_graph_collapse_toy.json"
MAXRL_SOURCE = ROOT / "ops/exp_scaling/plot_paper_e118_qwen05b_factorial.py"
MAXRL_AUDIT = ROOT / "paper/figures/replay_maxrl_qwen05b.json"
MANUSCRIPT = ROOT / "paper/main.tex"
ICLR_STYLE = ROOT / "paper/iclr2027_conference.sty"
ICLR_BIB_STYLE = ROOT / "paper/iclr2027_conference.bst"
ICLR_FANCYHDR = ROOT / "paper/fancyhdr.sty"
ICLR_NATBIB = ROOT / "paper/natbib.sty"
ICLR_STYLE_SHA256 = "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6"
ICLR_BIB_STYLE_SHA256 = "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5"
ICLR_FANCYHDR_SHA256 = "b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea"
ICLR_NATBIB_SHA256 = "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d"
EXAMPLES_SOURCE = ROOT / "ops/plot_paper_modebench_examples.py"
EXAMPLES_PDF = ROOT / "paper/figures/modebench_examples.pdf"
MECHANISM_SOURCE = ROOT / "ops/plot_paper_verified_replay_mechanism.py"
MECHANISM_PDF = ROOT / "paper/figures/verified_replay_mechanism.pdf"
MAIN_PDF = ROOT / "paper/main.pdf"
INTERIM_TABLE = ROOT / "paper/results/figure4_interim_20260806_table.json"
INTERIM_FIGURE = ROOT / "paper/figures/figure4_interim_20260806.json"
COMPARISON_DIR = ROOT / "paper/figures/comparisons"
TERMINAL_COMPARISON_FIGURES = {
    "core_retention_falcon1b_part1": {
        "label": "fig:falcon-terminal-trajectories-a",
        "comparison": "core_retention",
        "scale": "falcon1b",
        "domains": ["graph_coloring", "countdown", "python_factors"],
        "methods": {"drgrpo", "replay_grpo"},
        "seeds": [55, 56, 57, 58, 59],
        "semantic_arms": set(),
    },
    "core_retention_falcon1b_part2": {
        "label": "fig:falcon-terminal-trajectories-b",
        "comparison": "core_retention",
        "scale": "falcon1b",
        "domains": ["mathir", "pantry_plan"],
        "methods": {"drgrpo", "replay_grpo"},
        "seeds": [55, 56, 57, 58, 59],
        "semantic_arms": set(),
    },
    "fixed_semantic_factorial_cross_scale": {
        "label": "fig:maxent-factorial-trajectories",
        "comparison": "fixed_semantic_factorial",
        "methods": {
            "drgrpo", "semantic_maxent", "replay_grpo",
            "replay_semantic_maxent",
        },
        "forbidden_stems": (
            "fixed_semantic_factorial_qwen05b_part1",
            "fixed_semantic_factorial_qwen05b_part2",
        ),
        "rows": [
            {
                "row": "A",
                "scale": "qwen05b",
                "domains": [
                    "graph_coloring", "countdown", "python_factors",
                    "mathir", "pantry_plan",
                ],
                "missing_domains": [],
                "seeds": [43, 44, 45, 46, 47],
                "semantic_arms": {"semantic", "semantic_only"},
            },
            {
                "row": "B",
                "scale": "falcon1b",
                "domains": ["graph_coloring", "mathir"],
                "missing_domains": [
                    "countdown", "python_factors", "pantry_plan",
                ],
                "seeds": [55, 56, 57, 58, 59],
                "semantic_arms": {"semantic", "semantic_only"},
            },
        ],
    },
    "adaptive_semantic_replay_qwen05b_part1": {
        "label": "fig:adaptive-semantic-qwen-terminal",
        "comparison": "adaptive_semantic_replay",
        "scale": "qwen05b",
        "domains": ["graph_coloring", "countdown", "mathir"],
        "methods": {
            "drgrpo", "replay_grpo", "adaptive_semantic_replay",
        },
        "seeds": [43, 44, 45, 46, 47],
        "semantic_arms": {"adaptive_semantic"},
    },
    "adaptive_semantic_replay_falcon1b_part1": {
        "label": "fig:adaptive-semantic-falcon-terminal",
        "comparison": "adaptive_semantic_replay",
        "scale": "falcon1b",
        "domains": ["python_factors"],
        "methods": {
            "drgrpo", "replay_grpo", "adaptive_semantic_replay",
        },
        "seeds": [55, 56, 57, 58, 59],
        "semantic_arms": {"adaptive_semantic"},
    },
}
CROSS_SCALE_ENDPOINT = ROOT / "paper/figures/cross_scale_terminal_endpoint_effects"
DIRECT_COMPARATOR_ENDPOINT = (
    ROOT / "paper/figures/direct_comparator_endpoint_effects"
)
QWEN3B_ENDPOINT_PROGRESS = (
    ROOT / "paper/figures/qwen3b_exact_endpoint_progress"
)
SUSTAINED_AUC_EFFECTS = (
    ROOT / "paper/figures/sustained_auc_effects_qwen05b"
)
HISTORICAL_DECODING_FRONTIER = (
    ROOT / "paper/figures/e72_decoding_frontier"
)
TERMINAL_FRONTIER = ROOT / "paper/figures/terminal_pass8_distinct8_frontier"
PLAIN_GRPO_RESULTS = (
    ROOT / "paper/results/e95_falcon_plain_grpo_reportable.json"
)
QWEN3B_FIXED_SEMANTIC = (
    ROOT / "paper/results/e87_qwen3b_fixed_semantic_seed70.json"
)
ADAPTIVE_FRONTIER = (
    ROOT / "paper/figures/xmode_adaptive_cross_scale_distinct_at_k.json"
)
ADAPTIVE_PASS_FRONTIER = (
    ROOT / "paper/figures/xmode_adaptive_cross_scale_pass_at_k"
)
RLEP_DIRECT_STRIP = (
    ROOT / "paper/figures/direct_comparator_endpoint_effects.json"
)
OPEN_BANK_BUNDLE = ROOT / "paper/figures/open_bank_bundle_endpoint_effects"
E102_AUDIT = (
    ROOT / "var/artifacts/e102_full_open_bank_maxent_replay_05b_audit_latest.json"
)
REPLAY_MECHANISM_TELEMETRY = (
    ROOT / "paper/figures/replay_mechanism_telemetry_qwen05b"
)
BANK_OCCUPANCY_OUTCOMES = (
    ROOT / "paper/figures/bank_occupancy_retained_breadth"
)
E78_TERMINAL_RESULTS = ROOT / "paper/results/e78_terminal_05b.json"
ADAPTIVE_DOSE_GATE = ROOT / "paper/figures/adaptive_semantic_gate_e88"
ADAPTIVE_MECHANISM_OUTCOMES = (
    ROOT / "paper/figures/adaptive_mechanism_outcomes_qwen05b"
)
REPLAY_DOSE_PROGRESS = {
    "replay_dose_qwen05b_progress_part1": {
        "graph_coloring": [43, 44, 45, 46, 47],
        "countdown": [43, 44, 45, 46, 47],
        "python_factors": [43, 44, 45, 46, 47],
    },
    "replay_dose_qwen05b_progress_part2": {
        "mathir": [43, 44, 45, 46, 47],
        "pantry_plan": [43, 44, 45, 46, 47],
    },
}
UCPO_INTERIM = ROOT / "paper/results/ucpo_interim_05b.json"
UCPO_INTERIM_BODY = ROOT / "paper/results/ucpo_interim_05b_table_body.tex"
UCPO_CURVES = ROOT / "paper/figures/direct_comparator_endpoint_effects.json"
UCPO_CURVES_PDF = ROOT / "paper/figures/ucpo_interim_learning_curves.pdf"
UCPO_CURVES_PNG = ROOT / "paper/figures/ucpo_interim_learning_curves.png"
ALIGNED_MODEL_ORDER = ["qwen05b", "falcon1b", "qwen3b"]
ALIGNED_DOMAIN_ORDER = [
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
]
FIXED_AVAILABLE = {
    "graph_coloring": {
        "qwen05b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "falcon1b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "replay_semantic_maxent": 1},
    },
    "countdown": {
        "qwen05b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "falcon1b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "replay_semantic_maxent": 1},
    },
    "python_factors": {
        "qwen05b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "falcon1b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "replay_semantic_maxent": 1},
    },
    "mathir": {
        "qwen05b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "falcon1b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "replay_semantic_maxent": 1},
    },
    "pantry_plan": {
        "qwen05b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "falcon1b": {"drgrpo": 5, "semantic_maxent": 5, "replay_grpo": 5, "replay_semantic_maxent": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "replay_semantic_maxent": 1},
    },
}
CORE_AVAILABLE = {
    "graph_coloring": {
        "qwen05b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "falcon1b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "qwen3b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
    },
    "countdown": {
        "qwen05b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "falcon1b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "qwen3b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
    },
    "python_factors": {
        "qwen05b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "falcon1b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "qwen3b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
    },
    "mathir": {
        "qwen05b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "falcon1b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "qwen3b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
    },
    "pantry_plan": {
        "qwen05b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "falcon1b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
        "qwen3b": {"grpo": 5, "drgrpo": 5, "replay_grpo": 5},
    },
}
ADAPTIVE_AVAILABLE = {
    "graph_coloring": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 1},
    },
    "countdown": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 1},
    },
    "python_factors": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 1},
    },
    "mathir": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 1},
    },
    "pantry_plan": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 5},
        "qwen3b": {"drgrpo": 5, "replay_grpo": 5, "adaptive_semantic_replay": 1},
    },
}
REPLAY_DOSE_AVAILABLE = {
    domain: {"qwen05b": {method: count for method in (
        "drgrpo", "replay_grpo", "adaptive_replay_grpo"
    )}}
    for domain, count in {
        "graph_coloring": 5, "countdown": 5, "python_factors": 5,
        "mathir": 5, "pantry_plan": 5,
    }.items()
}
DIRECT_AVAILABLE = {
    "graph_coloring": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
    },
    "countdown": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
    },
    "python_factors": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 2},
    },
    "mathir": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
    },
    "pantry_plan": {
        "qwen05b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
        "falcon1b": {"drgrpo": 5, "replay_grpo": 5, "ucpo": 5, "rlep_dr": 5},
    },
}
ALIGNED_DOMAIN_STRIPS = {
    COMPARISON_DIR / "fixed_semantic_factorial_cross_scale_strip": {
        "label": "fig:maxent-factorial-trajectories",
        "comparison": "fixed_semantic_factorial",
        "layout": "three physical model rows by five static-domain columns",
        "methods": {"drgrpo", "semantic_maxent", "replay_grpo", "replay_semantic_maxent"},
        "available": FIXED_AVAILABLE,
    },
    COMPARISON_DIR / "core_retention_falcon1b_static_strip": {
        "label": "fig:cross-model-terminal-trajectories", "comparison": "core_retention",
        "layout": "three physical model rows by five static-domain columns",
        "methods": {"grpo", "drgrpo", "replay_grpo"}, "available": CORE_AVAILABLE,
    },
    COMPARISON_DIR / "core_retention_falcon1b_pass8_static_strip": {
        "label": "fig:cross-model-pass8-trajectories", "comparison": "core_retention",
        "layout": "three physical model rows by five static-domain columns",
        "methods": {"grpo", "drgrpo", "replay_grpo"}, "available": CORE_AVAILABLE,
    },
    COMPARISON_DIR / "core_retention_falcon1b_mean8_static_strip": {
        "label": "fig:cross-model-mean8-trajectories", "comparison": "core_retention",
        "layout": "three physical model rows by five static-domain columns",
        "methods": {"grpo", "drgrpo", "replay_grpo"}, "available": CORE_AVAILABLE,
    },
    COMPARISON_DIR / "adaptive_semantic_replay_cross_scale_strip": {
        "label": "fig:adaptive-semantic-terminal", "comparison": "adaptive_semantic_replay",
        "layout": "three physical model rows by five static-domain columns",
        "methods": {"drgrpo", "replay_grpo", "adaptive_semantic_replay"},
        "available": ADAPTIVE_AVAILABLE,
    },
    COMPARISON_DIR / "replay_dose_qwen05b_progress_static_strip": {
        "label": "fig:replay-dose-progress", "comparison": "replay_dose",
        "layout": "three physical model rows by five static-domain columns",
        "methods": {"drgrpo", "replay_grpo", "adaptive_replay_grpo"},
        "available": REPLAY_DOSE_AVAILABLE,
    },
    ROOT / "paper/figures/direct_baseline_learning_curves_static_strip": {
        "label": "fig:direct-baseline-curves", "comparison": "direct_baselines",
        "layout": "three physical model rows by five static-domain columns",
        "methods": {"drgrpo", "replay_grpo", "ucpo", "rlep_dr"},
        "available": DIRECT_AVAILABLE,
    },
}
OBSOLETE_SPLIT_FIGURE_INCLUDES = (
    "figures/comparisons/core_retention_falcon1b_part1.pdf",
    "figures/comparisons/core_retention_falcon1b_part2.pdf",
    "figures/comparisons/fixed_semantic_factorial_cross_scale.pdf",
    "figures/comparisons/adaptive_semantic_replay_qwen05b_part1.pdf",
    "figures/comparisons/adaptive_semantic_replay_falcon1b_part1.pdf",
    "figures/comparisons/replay_dose_qwen05b_progress_20260813_part1.pdf",
    "figures/comparisons/replay_dose_qwen05b_progress_20260813_part2.pdf",
    "figures/ucpo_interim_learning_curves.pdf",
)
MAXENT_FACTORIAL = ROOT / "paper/results/maxent_factorial_05b.json"
MAXENT_EFFECT_BODY = ROOT / "paper/results/maxent_factorial_05b_table_body.tex"
PROGRAM_STATUS = ROOT / "paper/results/paper_program_status.json"
PROGRAM_STATUS_BODY = ROOT / "paper/results/paper_program_status_table_body.tex"
DAPO_PROGRESS = ROOT / "paper/results/e113r4_dapo_progress.json"
MAXENT_UNCERTAINTY_BODY = (
    ROOT / "paper/results/maxent_factorial_05b_uncertainty_table_body.tex"
)
HEADLINE_SOURCE = ROOT / "ops/plot_paper_modecollapse.py"
APPENDIX_SOURCE = ROOT / "ops/exp_scaling/plot_e70_clean_05b_wide_live.py"
HEADLINE_PDF = ROOT / "paper/figures/modecollapse_training.pdf"
APPENDIX_PDF = (
    ROOT
    / "paper/figures/e68_e58_vs_grpo_05b_12ep_terminal_provenance.pdf"
)
APPENDIX_PROVENANCE = (
    ROOT / "var/artifacts/clean_05b_eight_environment_figure_provenance.json"
)
E70_TERMINAL_AUDIT = ROOT / "var/artifacts/e70_clean_stage_a_05b_audit_latest.json"
PANTRY_TERMINAL_AUDIT = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json"

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Paper figure contract failed: {message}")


def check_additional_evidence_figures(manuscript: str) -> None:
    compiled_manuscript = re.sub(
        r"\\iffalse.*?\\fi", "", manuscript, flags=re.DOTALL
    )
    require(
        "PointMaze" not in compiled_manuscript
        and "interactive_domain_boundary" not in compiled_manuscript
        and "interactive-domain" not in compiled_manuscript,
        "PointMaze is outside the active manuscript scope",
    )
    for stem, label in (
        (CROSS_SCALE_ENDPOINT, "fig:cross-scale-terminal-effects"),
        (DIRECT_COMPARATOR_ENDPOINT, "fig:direct-comparator-effects"),
        (ADAPTIVE_PASS_FRONTIER, "fig:inference-pass-frontier"),
        (QWEN3B_ENDPOINT_PROGRESS, "fig:qwen3b-exact-progress"),
        (SUSTAINED_AUC_EFFECTS, "fig:sustained-auc-effects"),
        (
            HISTORICAL_DECODING_FRONTIER,
            "fig:historical-decoding-frontier",
        ),
        (TERMINAL_FRONTIER, "fig:terminal-frontier"),
        (REPLAY_MECHANISM_TELEMETRY, "fig:replay-mechanism-telemetry"),
        (BANK_OCCUPANCY_OUTCOMES, "fig:bank-occupancy-outcomes"),
        (ADAPTIVE_DOSE_GATE, "fig:adaptive-dose-gate"),
        (ADAPTIVE_MECHANISM_OUTCOMES, "fig:adaptive-mechanism-outcomes"),
        (OPEN_BANK_BUNDLE, "fig:open-bank-bundle-effects"),
    ):
        for suffix in ("json", "pdf", "png"):
            path = stem.with_suffix(f".{suffix}")
            require(
                path.is_file() and path.stat().st_size > 0,
                f"evidence figure {path.name} is missing or empty",
            )
        require(
            compiled_manuscript.count(f"figures/{stem.name}.pdf") <= 1,
            f"evidence figure {stem.name} is compiled more than once",
        )
        require(
            compiled_manuscript.count(rf"\label{{{label}}}") <= 1,
            f"evidence figure {stem.name} reuses its manuscript label",
        )


    endpoint = json.loads(
        CROSS_SCALE_ENDPOINT.with_suffix(".json").read_text(encoding="utf-8")
    )
    rows = endpoint.get("rows", [])
    require(
        endpoint.get("schema") == "paper-cross-scale-endpoint-effects-v2"
        and endpoint.get("status")
        == "all available terminal pass-8 seeds with exact n"
        and endpoint.get("model_rows")
        == ["Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B"]
        and endpoint.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and endpoint.get("methods")
        == ["GRPO", "matched Dr.GRPO", "Re:Dr.GRPO"]
        and len(rows) == 15,
        "cross-scale endpoint figure is not the 15-cell exact-n snapshot",
    )
    require(
        [row.get("model") for row in rows]
        == ["Qwen2.5-0.5B"] * 5
        + ["Falcon3-1B"] * 5
        + ["Qwen2.5-3B"] * 5,
        "cross-scale endpoint model rows drifted",
    )
    require(
        all(
            1 <= len(row.get("seeds", [])) <= 5
            and row.get("n") == len(row["seeds"])
            and set(row.get("per_seed_effects", {}))
            == {str(seed) for seed in row["seeds"]}
            for row in rows
        ),
        "cross-scale endpoint figure lacks exact per-row seed effects",
    )
    plain_grpo = json.loads(
        PLAIN_GRPO_RESULTS.read_text(encoding="utf-8")
    )
    require(
        plain_grpo.get("schema")
        == "paper-e95-falcon-plain-grpo-available-v2"
        and plain_grpo.get("status")
        == "all available sampled checkpoints; no minimum seed count"
        and plain_grpo.get("model") == "Falcon3-1B"
        and plain_grpo.get("scale") == "falcon1b"
        and plain_grpo.get("registered_seeds") == [55, 56, 57, 58, 59]
        and list(plain_grpo.get("domains", {})) == ALIGNED_DOMAIN_ORDER,
        "plain-GRPO summary drifted from the all-available E95 contract",
    )
    require(
        all(
            "8.0" in record.get("summary_by_pass", {})
            and 1 <= record["summary_by_pass"]["8.0"]["n"] <= 5
            and record["summary_by_pass"]["8.0"]["n"]
            == len(record["summary_by_pass"]["8.0"]["seeds"])
            for record in plain_grpo["domains"].values()
        ),
        "plain-GRPO summary omits an available pass-8 domain",
    )
    grpo_rows = [
        row for row in rows if row.get("grpo_per_seed_effects")
    ]
    require(
        [(row.get("model"), row.get("domain")) for row in grpo_rows]
        == [("Falcon3-1B", domain) for domain in ALIGNED_DOMAIN_ORDER]
        and all(
            set(row["grpo_summaries"])
            == {"pass8", "adjusted_breadth8"}
            for row in grpo_rows
        )
        and endpoint.get("plain_grpo_source_sha256")
        == file_sha256(PLAIN_GRPO_RESULTS),
        "endpoint forest does not contain every available Falcon GRPO contrast",
    )
    for row in grpo_rows:
        pass8 = plain_grpo["domains"][row["domain"]]["summary_by_pass"]["8.0"]
        expected = {str(seed) for seed in pass8["seeds"]} & {
            str(seed) for seed in row["seeds"]
        }
        require(
            set(row["grpo_per_seed_effects"]) == expected,
            f"{row['domain']}: GRPO endpoint seed set drifted",
        )

    direct = json.loads(
        DIRECT_COMPARATOR_ENDPOINT.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    direct_cells = direct.get("cells", [])
    require(
        direct.get("schema") == "paper-direct-comparator-endpoint-effects-v2"
        and direct.get("status")
        == (
            "mixed terminal balanced blocks and exact terminal prefixes; "
            "only n=5 receives a mean and paired 95% Student-t interval"
        )
        and direct.get("model_rows")
        == ["Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B"]
        and direct.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and direct.get("comparators") == ["grpo", "ucpo", "rlep_dr"]
        and len(direct_cells) == 15
        and len(direct.get("input_sha256", {})) > 0,
        "direct-comparator endpoint forest metadata drifted",
    )
    direct_by_cell = {
        (cell.get("model"), cell.get("domain")): cell
        for cell in direct_cells
    }
    require(
        set(direct_by_cell)
        == {
            (model, domain)
            for model in direct["model_rows"]
            for domain in direct["domain_order"]
        }
        and all(
            set(cell.get("methods", {})) <= set(direct["comparators"])
            and all(
                record.get("n") == len(record.get("seeds", []))
                and set(record.get("per_seed", {}))
                == {str(seed) for seed in record.get("seeds", [])}
                and record.get("evidence")
                == (
                    "balanced_five_seed_terminal"
                    if record.get("n") == 5
                    else "exact_terminal_prefix"
                )
                and ("summaries" in record) == (record.get("n") == 5)
                and (
                    set(record.get("summaries", {}))
                    == {"pass8", "adjusted_breadth8"}
                    if record.get("n") == 5
                    else True
                )
                for record in cell.get("methods", {}).values()
            )
            for cell in direct_cells
        ),
        "direct-comparator forest lacks exact method-specific paired evidence",
    )
    require(
        direct_by_cell[("Qwen2.5-0.5B", "graph_coloring")]["methods"]
        ["ucpo"]["n"]
        == 5
        and direct_by_cell[("Qwen2.5-0.5B", "countdown")]["methods"]
        ["ucpo"]["seeds"]
        == [43, 44, 45, 46, 47]
        and direct_by_cell[("Qwen2.5-0.5B", "countdown")]["methods"]
        ["rlep_dr"]["seeds"]
        == [43, 44, 45, 46, 47]
        and direct_by_cell[("Qwen2.5-3B", "countdown")]["methods"]
        ["grpo"]["seeds"]
        == [70, 71, 72, 73, 74]
        and all(
            direct_by_cell[("Falcon3-1B", domain)]["methods"]["grpo"]["n"]
            == 5
            for domain in ALIGNED_DOMAIN_ORDER
        )
        and direct_by_cell[("Qwen2.5-0.5B", "graph_coloring")]["methods"]
        ["rlep_dr"]["n"]
        == 5
        and direct_by_cell[("Qwen2.5-0.5B", "python_factors")]["methods"]
        ["rlep_dr"]["n"]
        == 5
        and all(
            direct_by_cell[("Falcon3-1B", domain)]["methods"]["ucpo"]["n"]
            == 5
            for domain in ALIGNED_DOMAIN_ORDER
        )
        and all(
            direct_by_cell[("Falcon3-1B", domain)]["methods"]["rlep_dr"]["n"]
            == 5
            for domain in ("graph_coloring", "countdown", "mathir", "pantry_plan")
        )
        and direct_by_cell[("Falcon3-1B", "python_factors")]["methods"]
        ["rlep_dr"]["seeds"]
        == [55, 57],
        "direct-comparator forest lost a completed block or sparse RLEP prefix",
    )

    open_bank = json.loads(
        OPEN_BANK_BUNDLE.with_suffix(".json").read_text(encoding="utf-8")
    )
    open_bank_cells = open_bank.get("cells", [])
    audit = json.loads(E102_AUDIT.read_text(encoding="utf-8"))
    require(
        open_bank.get("schema") == "paper-open-bank-bundle-endpoint-effects-v1"
        and open_bank.get("status")
        == "terminal preregistered follow-up; 25/25 cells passed mechanism audit"
        and open_bank.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and open_bank.get("seeds") == [43, 44, 45, 46, 47]
        and [cell.get("domain") for cell in open_bank_cells]
        == ALIGNED_DOMAIN_ORDER
        and all(
            cell.get("n") == 5
            and cell.get("seeds") == [43, 44, 45, 46, 47]
            and set(cell.get("per_seed", {}))
            == {"43", "44", "45", "46", "47"}
            and set(cell.get("effect_summaries", {}))
            == {"replay", "control"}
            for cell in open_bank_cells
        )
        and open_bank.get("mechanism_audit", {}).get("violations") == []
        and audit.get("schema") == "e102-full-open-bank-campaign-audit-v1"
        and audit.get("released") is True
        and audit.get("terminal") is True
        and audit.get("passed_so_far") is True
        and audit.get("violations") == [],
        "E102 terminal bundle or mechanism-audit contract drifted",
    )
    require(
        open_bank.get("mechanism_audit", {}).get("domain_gate")
        == audit.get("domain_gate")
        and open_bank.get("mechanism_audit", {}).get("compute_accounting")
        == audit.get("compute_accounting")
        and open_bank.get("reporting_gap")
        == (
            "not retained by the E102 actor path; this preregistered reporting "
            "quantity is unavailable and is not imputed"
        ),
        "E102 mechanism counts or reporting boundary drifted",
    )
    if f"figures/{OPEN_BANK_BUNDLE.name}.pdf" in compiled_manuscript:
        require(
            "Realized proposal prompt and" in compiled_manuscript
            and "does not identify which of its three additions causes an effect"
            in compiled_manuscript,
            "the open-bank bundle is promoted without its reporting boundary",
        )
    for cell in open_bank_cells:
        for comparator in ("replay", "control"):
            for metric in ("pass8", "adjusted_breadth8"):
                values = [
                    cell["per_seed"][str(seed)]["effects"][comparator][metric]
                    for seed in cell["seeds"]
                ]
                summary = cell["effect_summaries"][comparator][metric]
                mean = statistics.fmean(values)
                half = (
                    2.7764451051977987
                    * statistics.stdev(values)
                    / math.sqrt(5)
                )
                require(
                    math.isclose(
                        summary["mean"], mean, rel_tol=0.0, abs_tol=1e-12
                    )
                    and all(
                        math.isclose(
                            left, right, rel_tol=0.0, abs_tol=1e-12
                        )
                        for left, right in zip(
                            summary["student_t_95"],
                            [mean - half, mean + half],
                        )
                    ),
                    f"E102 interval drifted for {cell['domain']}/{comparator}/{metric}",
                )

    qwen3b = json.loads(
        QWEN3B_ENDPOINT_PROGRESS.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    qwen3b_methods = [
        "drgrpo",
        "grpo",
        "replay_grpo",
        "replay_semantic_maxent",
        "adaptive_semantic_replay",
    ]
    qwen3b_cells = qwen3b.get("cells", [])
    require(
        qwen3b.get("schema") == "paper-qwen3b-exact-endpoint-progress-v1"
        and qwen3b.get("status")
        == "all exact terminal Qwen2.5-3B endpoints at the freeze"
        and qwen3b.get("model") == "Qwen2.5-3B"
        and qwen3b.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and qwen3b.get("methods") == qwen3b_methods
        and qwen3b.get("aggregation")
        == "none; no means, ranges, intervals, or pooling"
        and qwen3b.get("metrics") == {"x": "pass@8", "y": "distinct@8"}
        and [cell.get("domain") for cell in qwen3b_cells]
        == ALIGNED_DOMAIN_ORDER,
        "Qwen2.5-3B exact-progress metadata drifted",
    )
    expected_qwen3b_seeds = {
        "graph_coloring": {
            "drgrpo": [70, 71, 72, 73, 74],
            "grpo": [70, 71, 72, 73, 74],
            "replay_grpo": [70, 71, 72, 73, 74],
            "replay_semantic_maxent": [70],
            "adaptive_semantic_replay": [70],
        },
        "countdown": {
            "drgrpo": [70, 71, 72, 73, 74],
            "grpo": [70, 71, 72, 73, 74],
            "replay_grpo": [70, 71, 72, 73, 74],
            "replay_semantic_maxent": [70],
            "adaptive_semantic_replay": [70],
        },
        "python_factors": {
            "drgrpo": [70, 71, 72, 73, 74],
            "grpo": [70, 71, 72, 73, 74],
            "replay_grpo": [70, 71, 72, 73, 74],
            "replay_semantic_maxent": [70],
            "adaptive_semantic_replay": [70],
        },
        "mathir": {
            "drgrpo": [70, 71, 72, 73, 74],
            "grpo": [70, 71, 72, 73, 74],
            "replay_grpo": [70, 71, 72, 73, 74],
            "replay_semantic_maxent": [70],
            "adaptive_semantic_replay": [70],
        },
        "pantry_plan": {
            "drgrpo": [70, 71, 72, 73, 74],
            "grpo": [70, 71, 72, 73, 74],
            "replay_grpo": [70, 71, 72, 73, 74],
            "replay_semantic_maxent": [70],
            "adaptive_semantic_replay": [70],
        },
    }
    for cell in qwen3b_cells:
        expected = expected_qwen3b_seeds[cell["domain"]]
        require(
            set(cell.get("methods", {})) == set(expected),
            f"Qwen2.5-3B exact-progress methods drifted for {cell['domain']}",
        )
        for method, seeds in expected.items():
            record = cell["methods"][method]
            require(
                record.get("n") == len(seeds)
                and record.get("seeds") == seeds
                and record.get("evidence")
                == "exact_terminal_seeds_no_aggregate"
                and set(record.get("per_seed", {}))
                == {str(seed) for seed in seeds}
                and all(
                    set(point) == {"pass8", "distinct8", "job_id"}
                    for point in record["per_seed"].values()
                )
                and not (
                    {"summary", "mean", "range", "interval", "ci"}
                    & set(record)
                ),
                f"Qwen2.5-3B exact-progress evidence drifted for "
                f"{cell['domain']}/{method}",
            )
    require(
        qwen3b.get("input_sha256")
        and all(
            (ROOT / relative).is_file()
            and metadata.get("byte_length") == (ROOT / relative).stat().st_size
            and metadata.get("sha256") == file_sha256(ROOT / relative)
            for relative, metadata in qwen3b["input_sha256"].items()
        ),
        "Qwen2.5-3B exact-progress source ledger drifted",
    )

    sustained = json.loads(
        SUSTAINED_AUC_EFFECTS.with_suffix(".json").read_text(encoding="utf-8")
    )
    sustained_source = ROOT / str(sustained.get("source_json", ""))
    sustained_metrics = {
        "normalized_auc_pass8",
        "normalized_auc_distinct8",
        "normalized_auc_adjusted_breadth8",
    }
    require(
        sustained.get("schema") == "paper-sustained-auc-effects-qwen05b-v1"
        and sustained.get("status")
        == "balanced five-seed full-trajectory effect forest"
        and sustained.get("model") == "Qwen2.5-0.5B-Instruct"
        and sustained.get("comparison")
        == "Re:Dr.GRPO minus matched Dr.GRPO"
        and sustained.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and sustained.get("paired_seeds") == [43, 44, 45, 46, 47]
        and set(sustained.get("metrics", {})) == sustained_metrics
        and [cell.get("domain") for cell in sustained.get("cells", [])]
        == ALIGNED_DOMAIN_ORDER
        and sustained_source == ROOT / "paper/results/e78_terminal_05b.json"
        and sustained_source.is_file()
        and sustained.get("source_sha256") == file_sha256(sustained_source),
        "sustained-effect AUC forest metadata or source hash drifted",
    )
    sustained_frozen = json.loads(sustained_source.read_text(encoding="utf-8"))
    for cell in sustained["cells"]:
        effects = cell.get("effects", {})
        seeds = cell.get("seeds", [])
        require(
            cell.get("n") == 5
            and seeds == [43, 44, 45, 46, 47]
            and cell.get("evidence") == "balanced_five_seed_full_trajectory"
            and set(effects) == sustained_metrics
            and all(
                set(record.get("per_seed", {}))
                == {str(seed) for seed in seeds}
                and len(record.get("student_t_95", [])) == 2
                for record in effects.values()
            ),
            f"sustained-effect evidence drifted for {cell['domain']}",
        )
        source_effects = sustained_frozen["domains"][cell["domain"]][
            "paired_effects"
        ]
        for seed in seeds:
            key = str(seed)
            pass_value = effects["normalized_auc_pass8"]["per_seed"][key]
            distinct_value = effects["normalized_auc_distinct8"]["per_seed"][key]
            adjusted_value = effects[
                "normalized_auc_adjusted_breadth8"
            ]["per_seed"][key]
            require(
                math.isclose(
                    pass_value,
                    source_effects["normalized_auc_pass8"]["per_seed"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    distinct_value,
                    source_effects["normalized_auc_distinct8"]["per_seed"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    adjusted_value,
                    distinct_value - pass_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"sustained-effect seed value drifted for {cell['domain']}/{seed}",
            )
        for record in effects.values():
            values = list(record["per_seed"].values())
            mean = statistics.fmean(values)
            half_width = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
            require(
                math.isclose(record["mean"], mean, rel_tol=0.0, abs_tol=1e-12)
                and all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                    for left, right in zip(
                        record["student_t_95"],
                        [mean - half_width, mean + half_width],
                    )
                ),
                f"sustained-effect interval drifted for {cell['domain']}",
            )

    decoding = json.loads(
        HISTORICAL_DECODING_FRONTIER.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    decoding_source_path = ROOT / str(decoding.get("summary", ""))
    require(
        decoding.get("schema") == "paper-historical-decoding-frontier-v2"
        and decoding.get("status")
        == "complete five-seed historical control; not x-Mode evidence"
        and decoding.get("model") == "Qwen2.5-0.5B-Instruct"
        and decoding.get("checkpoint") == "terminal pass 12"
        and decoding.get("paired_seeds") == [43, 44, 45, 46, 47]
        and decoding.get("temperature_sweep")
        == [0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
        and decoding.get("metrics") == {"x": "mean@8", "y": "distinct@8"}
        and decoding.get("arms")
        == {
            "drgrpo": "matched Dr.GRPO",
            "xgrpo": "historical multi-component treatment",
        }
        and decoding.get("cells_measured") == 660
        and decoding.get("reproduction_gate_passed") is True
        and decoding.get("reproduction_gate_checked") == 240
        and decoding.get("reproduction_gate_failed") == 0
        and decoding_source_path
        == ROOT / "var/artifacts/e72_decoding_frontier_summary.json"
        and decoding_source_path.is_file(),
        "historical decoding frontier metadata or evidence boundary drifted",
    )
    require(
        decoding.get("input_sha256")
        and all(
            (ROOT / relative).is_file()
            and record.get("path") == relative
            and record.get("byte_length") == (ROOT / relative).stat().st_size
            and record.get("sha256") == file_sha256(ROOT / relative)
            for relative, record in decoding["input_sha256"].items()
        ),
        "historical decoding frontier source hash drifted",
    )
    decoding_source = json.loads(decoding_source_path.read_text(encoding="utf-8"))
    require(
        decoding_source.get("schema") == "e72_decoding_frontier_summary_v1"
        and decoding_source.get("cells_measured") == 660
        and decoding_source.get("reproduction_gate", {}).get("passed") is True
        and decoding_source.get("reproduction_gate", {}).get("checked") == 240
        and decoding_source.get("reproduction_gate", {}).get("failed") == 0,
        "frozen E72 decoding summary no longer passes its reproduction gate",
    )
    displayed = decoding.get("displayed", {})
    series = displayed.get("series", [])
    require(
        displayed.get("domains") == ALIGNED_DOMAIN_ORDER
        and len(series) == 10
        and {
            (record.get("domain"), record.get("arm")) for record in series
        }
        == {
            (domain, arm)
            for domain in ALIGNED_DOMAIN_ORDER
            for arm in ("drgrpo", "xgrpo")
        },
        "historical decoding frontier lost a domain or arm",
    )
    source_points = {
        (point["domain"], point["arm"], point["temperature"]): point
        for point in decoding_source["frontier"]["points"]
        if point.get("k") == 8 and point.get("arm") in {"drgrpo", "xgrpo"}
    }
    for record in series:
        points = record.get("points", [])
        require(
            [point.get("temperature") for point in points]
            == [0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
            and all(
                point.get("n") == 5
                and point.get("seeds") == [43, 44, 45, 46, 47]
                for point in points
            ),
            f"historical decoding series denominator drifted for "
            f"{record.get('domain')}/{record.get('arm')}",
        )
        for point in points:
            source_point = source_points[
                (record["domain"], record["arm"], point["temperature"])
            ]
            require(
                math.isclose(
                    point["mean_at_8"], source_point["mean_at_k"],
                    rel_tol=0.0, abs_tol=1e-12,
                )
                and math.isclose(
                    point["distinct_at_8"], source_point["distinct_at_k"],
                    rel_tol=0.0, abs_tol=1e-12,
                )
                and point.get("per_seed") == source_point.get("per_seed"),
                f"historical decoding point drifted for "
                f"{record['domain']}/{record['arm']}/T={point['temperature']}",
            )
    expected_repair = [
        row
        for row in decoding_source["frontier"]["temperature_repair"]
        if row["arm"] == "drgrpo"
    ]
    expected_budget = [
        row for row in decoding_source["frontier"]["budget"]
        if row["arm"] in {"drgrpo", "xgrpo"}
    ]
    expected_nucleus = [
        row for row in decoding_source["frontier"]["nucleus"]
        if row["arm"] in {"drgrpo", "xgrpo"}
    ]
    require(
        decoding.get("temperature_repair") == expected_repair
        and decoding.get("sample_budget") == expected_budget
        and decoding.get("nucleus") == expected_nucleus
        and len(expected_repair) == 5
        and max(row["repair_index_rho"] for row in expected_repair) < 0.73
        and len(expected_budget) == 10
        and len(expected_nucleus) == 20,
        "historical decoding secondary controls drifted",
    )
    if f"figures/{HISTORICAL_DECODING_FRONTIER.name}.pdf" in compiled_manuscript:
        for token in (
            "superseded 12-pass multi-component treatment",
            "it is not an estimate of",
            "checkpoint-support control only",
            "all 240 temperature-one reproduction",
        ):
            require(
                token in compiled_manuscript,
                f"historical decoding scope is missing {token!r}",
            )

    pass_frontier = json.loads(
        ADAPTIVE_PASS_FRONTIER.with_suffix(".json").read_text(encoding="utf-8")
    )
    source_frontier = json.loads(ADAPTIVE_FRONTIER.read_text(encoding="utf-8"))
    require(
        pass_frontier.get("schema") == "paper-xmode-pass-at-k-companion-v1"
        and pass_frontier.get("metric") == "pass@K"
        and pass_frontier.get("source_json")
        == "paper/figures/xmode_adaptive_cross_scale_distinct_at_k.json"
        and pass_frontier.get("source_sha256") == file_sha256(ADAPTIVE_FRONTIER)
        and pass_frontier.get("row_order")
        == ["qwen05b", "falcon1b", "qwen3b"]
        and pass_frontier.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and pass_frontier.get("ks") == [1, 2, 4, 8, 16, 32]
        and pass_frontier.get("methods")
        == ["drgrpo", "adaptive_semantic_replay"],
        "pass@K companion metadata drifted from the frozen breadth record",
    )
    pass_cells = pass_frontier.get("cells", {})
    require(
        set(pass_cells) == {"qwen05b", "falcon1b", "qwen3b"}
        and set(pass_cells["qwen05b"]) == set(ALIGNED_DOMAIN_ORDER)
        and set(pass_cells["falcon1b"]) == set(ALIGNED_DOMAIN_ORDER)
        and set(pass_cells["qwen3b"]) == set(ALIGNED_DOMAIN_ORDER),
        "pass@K companion does not preserve the fully populated invariant grid",
    )
    for scale in ("qwen05b", "falcon1b", "qwen3b"):
        for domain in ALIGNED_DOMAIN_ORDER:
            cell = pass_cells[scale][domain]
            source_cell = source_frontier["cells"][scale][domain]
            require(
                cell.get("n") == len(cell.get("seeds", []))
                and cell.get("seeds") == source_cell.get("seeds")
                and cell.get("checkpoint_step")
                == source_cell.get("checkpoint_step")
                and cell.get("checkpoint_training_pass")
                == source_cell.get("checkpoint_training_pass")
                and cell.get("terminal_checkpoint")
                == source_cell.get("terminal_checkpoint")
                and set(cell.get("methods", {}))
                == {"drgrpo", "adaptive_semantic_replay"},
                f"pass@K companion cell metadata drifted for {scale}/{domain}",
            )
            for method, source_method in (
                ("drgrpo", "control"),
                ("adaptive_semantic_replay", "xmode"),
            ):
                by_k = cell["methods"][method].get("by_k", {})
                require(
                    set(by_k) == {"1", "2", "4", "8", "16", "32"}
                    and all(
                        by_k[str(k)]
                        == source_cell["summaries"][source_method][str(k)]["pass8"]
                        and set(by_k[str(k)].get("per_seed", {}))
                        == {str(seed) for seed in cell["seeds"]}
                        for k in (1, 2, 4, 8, 16, 32)
                    ),
                    f"pass@K companion values drifted for {scale}/{domain}/{method}",
                )

    frontier = json.loads(
        TERMINAL_FRONTIER.with_suffix(".json").read_text(encoding="utf-8")
    )
    frontier_cells = frontier.get("cells", [])
    frontier_method_keys = [
        "drgrpo",
        "grpo",
        "replay_grpo",
        "ucpo",
        "rlep_dr",
    ]
    require(
        frontier.get("schema")
        == "paper-terminal-accuracy-breadth-frontier-v5"
        and frontier.get("status")
        == (
            "current direct-method pass-8 endpoints with exact n; retired "
            "semantic treatments and endpoint-incomplete DAPO are excluded"
        )
        and frontier.get("x_metric") == "pass@8"
        and frontier.get("y_metric") == "distinct@8"
        and frontier.get("methods")
        == [
            "matched Dr.GRPO",
            "GRPO",
            "Re:Dr.GRPO",
            "UCPO",
            "RLEP-Dr",
        ]
        and frontier.get("method_keys") == frontier_method_keys
        and frontier.get("model_rows")
        == ["Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B"]
        and frontier.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and frontier.get("plain_grpo_source_sha256")
        == file_sha256(PLAIN_GRPO_RESULTS)
        and frontier.get("direct_comparator_source_sha256")
        == file_sha256(DIRECT_COMPARATOR_ENDPOINT.with_suffix(".json"))
        and set(frontier.get("excluded", {}))
        == {"dapo", "retired_semantic_maxent"}
        and "no standardized" in frontier["excluded"]["dapo"]
        and "historical appendix"
        in frontier["excluded"]["retired_semantic_maxent"]
        and len(frontier_cells) == 15,
        "terminal correctness-breadth frontier drifted",
    )
    complete_frontier_methods = set(frontier_method_keys)
    expected_frontier_methods = {
        (model, domain): complete_frontier_methods
        for model in ("Qwen2.5-0.5B", "Falcon3-1B")
        for domain in ALIGNED_DOMAIN_ORDER
    }
    expected_frontier_methods.update({
        ("Qwen2.5-3B", domain): {"drgrpo", "grpo", "replay_grpo"}
        for domain in ALIGNED_DOMAIN_ORDER
    })
    require(
        all(
            set(cell.get("methods", {}))
            == expected_frontier_methods[(cell["model"], cell["domain"])]
            and set(cell.get("methods", {}))
            | set(cell.get("blank_methods", {}))
            == set(frontier_method_keys)

            and all(
                record["n"]
                == (
                    2
                    if cell["model"] == "Falcon3-1B"
                    and cell["domain"] == "python_factors"
                    and method == "rlep_dr"
                    else 5
                )
                and record["n"] == len(record["seeds"])
                and set(record["per_seed"])
                == {str(seed) for seed in record["seeds"]}
                and all(
                    set(point) == {"pass8", "distinct8"}
                    for point in record["per_seed"].values()
                )
                and "summary" in record
                and record["evidence"] == "all_available_terminal_seeds"
                for method, record in cell["methods"].items()
            )
            for cell in frontier_cells
        ),
        "frontier lacks exact method-specific pass-8 endpoint evidence",
    )

    gate = json.loads(
        ADAPTIVE_DOSE_GATE.with_suffix(".json").read_text(encoding="utf-8")
    )
    require(
        gate.get("schema") == "paper-adaptive-dose-gate-figure-v1"
        and gate.get("status") == "closed registered mechanism-gate evidence"
        and gate.get("target_ratio") == 0.05
        and gate.get("maximum_bound_hit_fraction") == 0.20
        and len(gate.get("records", [])) == 9,
        "adaptive-dose gate figure drifted from the E88 closure record",
    )

    telemetry = json.loads(
        REPLAY_MECHANISM_TELEMETRY.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    require(
        telemetry.get("schema") == "paper-replay-mechanism-telemetry-v1"
        and telemetry.get("status") == "terminal observational mechanism telemetry"
        and telemetry.get("replay_runs") == 25
        and telemetry.get("control_runs") == 25
        and telemetry.get("replay_updates") == 76800
        and telemetry.get("capacity") == 16
        and telemetry.get("capacity_hit_updates") == 477,
        "replay mechanism telemetry drifted from the terminal E78 logs",
    )
    require(
        list(telemetry.get("domains", {}))
        == ["graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"]
        and all(
            row.get("seeds") == [43, 44, 45, 46, 47]
            and all(
                seed_row.get("updates") == 3072
                for seed_row in row.get("replay_seed_summaries", {}).values()
            )
            for row in telemetry.get("domains", {}).values()
        ),
        "replay mechanism telemetry is not the complete five-domain seed block",
    )
    require(
        any("survival" in item for item in telemetry.get("limitations", [])),
        "replay telemetry must disclose that identity-level survival is unavailable",
    )

    occupancy_outcomes = json.loads(
        BANK_OCCUPANCY_OUTCOMES.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    require(
        occupancy_outcomes.get("schema")
        == "paper-bank-occupancy-retained-breadth-v2"
        and occupancy_outcomes.get("status")
        == (
            "terminal observational cross-measure alignment; "
            "no pooled or causal estimate"
        )
        and occupancy_outcomes.get("model") == "Qwen2.5-0.5B-Instruct"
        and occupancy_outcomes.get("comparison")
        == "Re:Dr.GRPO minus matched Dr.GRPO"
        and occupancy_outcomes.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and occupancy_outcomes.get("registered_seeds") == [43, 44, 45, 46, 47]
        and occupancy_outcomes.get("capacity") == 16
        and set(occupancy_outcomes.get("panels", {}))
        == {"trajectory", "terminal", "capacity"}
        and occupancy_outcomes.get("aggregation")
        == (
            "five seed points and one descriptive within-domain mean; "
            "no regression, correlation, interval, or cross-domain pooling"
        )
        and len(occupancy_outcomes.get("cells", [])) == 5,
        "bank occupancy/outcome figure metadata or claim boundary drifted",
    )
    require(
        occupancy_outcomes.get("input_sha256")
        and set(occupancy_outcomes["input_sha256"])
        == {
            "paper/figures/replay_mechanism_telemetry_qwen05b.json",
            "paper/results/e78_terminal_05b.json",
        }
        and all(
            (ROOT / relative).is_file()
            and record.get("path") == relative
            and record.get("byte_length") == (ROOT / relative).stat().st_size
            and record.get("sha256") == file_sha256(ROOT / relative)
            for relative, record in occupancy_outcomes["input_sha256"].items()
        ),
        "bank occupancy/outcome source hash drifted",
    )
    terminal_results = json.loads(E78_TERMINAL_RESULTS.read_text(encoding="utf-8"))
    require(
        terminal_results.get("schema") == "e78_terminal_paper_results_v1"
        and terminal_results.get("design", {}).get("domains")
        == ALIGNED_DOMAIN_ORDER
        and terminal_results.get("design", {}).get("paired_seeds")
        == [43, 44, 45, 46, 47],
        "terminal result source for occupancy/outcome join drifted",
    )
    joined_cells = occupancy_outcomes["cells"]
    require(
        [cell.get("domain") for cell in joined_cells] == ALIGNED_DOMAIN_ORDER
        and all(
            cell.get("n") == 5
            and cell.get("seeds") == [43, 44, 45, 46, 47]
            and set(cell.get("per_seed", {}))
            == {"43", "44", "45", "46", "47"}
            for cell in joined_cells
        ),
        "bank occupancy/outcome join lost a domain or paired seed",
    )
    joined_fields = {
        "mean_bank_occupancy",
        "capacity_hit_updates",
        "capacity_hit_fraction",
        "normalized_auc_pass8_effect",
        "normalized_auc_distinct8_effect",
        "normalized_auc_adjusted_breadth_effect",
        "late_bank_occupancy",
        "terminal_pass8_effect",
        "terminal_distinct8_effect",
        "terminal_adjusted_breadth_effect",
    }
    for cell in joined_cells:
        domain = cell["domain"]
        telemetry_cell = telemetry["domains"][domain]
        effects = terminal_results["domains"][domain]["paired_effects"]
        for seed in (43, 44, 45, 46, 47):
            key = str(seed)
            record = cell["per_seed"][key]
            require(
                set(record) == joined_fields
                and math.isclose(
                    record["mean_bank_occupancy"],
                    telemetry_cell["replay_seed_summaries"][key][
                        "mean_available_modes"
                    ],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    record["late_bank_occupancy"],
                    telemetry_cell["late_occupancy"]["seed_values"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and record["capacity_hit_updates"]
                == telemetry_cell["replay_seed_summaries"][key][
                    "capacity_hit_updates"
                ]
                and math.isclose(
                    record["capacity_hit_fraction"],
                    telemetry_cell["replay_seed_summaries"][key][
                        "capacity_hit_fraction"
                    ],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    record["normalized_auc_pass8_effect"],
                    effects["normalized_auc_pass8"]["per_seed"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    record["normalized_auc_distinct8_effect"],
                    effects["normalized_auc_distinct8"]["per_seed"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    record["normalized_auc_adjusted_breadth_effect"],
                    record["normalized_auc_distinct8_effect"]
                    - record["normalized_auc_pass8_effect"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    record["terminal_pass8_effect"],
                    effects["terminal_pass8"]["per_seed"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    record["terminal_distinct8_effect"],
                    effects["terminal_distinct8"]["per_seed"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    record["terminal_adjusted_breadth_effect"],
                    effects["terminal_excess_modes8"]["per_seed"][key],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"bank occupancy/outcome value drifted for {domain}/{seed}",
            )
        require(
            set(cell.get("means", {})) == joined_fields
            and all(
                math.isclose(
                    cell["means"][field],
                    statistics.fmean(
                        cell["per_seed"][str(seed)][field]
                        for seed in (43, 44, 45, 46, 47)
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field in joined_fields
            ),
            f"bank occupancy/outcome domain mean drifted for {domain}",
        )
    require(
        sum(
            record["capacity_hit_updates"]
            for cell in joined_cells
            for record in cell["per_seed"].values()
        )
        == 477
        and all(
            record["capacity_hit_updates"] == 0
            for cell in joined_cells
            if cell["domain"] != "pantry_plan"
            for record in cell["per_seed"].values()
        ),
        "bank occupancy/outcome capacity-contact evidence drifted",
    )
    require(
        any(
            "cannot be reconstructed" in item
            for item in occupancy_outcomes.get("limitations", [])
        ),
        "bank occupancy/outcome limitations left the artifact",
    )
    if f"figures/{BANK_OCCUPANCY_OUTCOMES.name}.pdf" in compiled_manuscript:
        require(
            "no regression, correlation, interval, or" in compiled_manuscript
            and re.search(
                r"not an estimated\s+mediator or replay dose", compiled_manuscript
            ),
            "bank occupancy/outcome limitations are not explicit in the paper",
        )

    adaptive_outcomes = json.loads(
        ADAPTIVE_MECHANISM_OUTCOMES.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    adaptive_fields = [
        "final_realized_ratio",
        "final_coefficient",
        "delta_pass8",
        "delta_adjusted_breadth8",
    ]
    expected_adaptive_seeds = {
        "graph_coloring": [43, 44, 45, 46, 47],
        "countdown": [43, 44, 45, 46, 47],
        "python_factors": [43, 44, 45, 46, 47],
        "mathir": [43, 44, 45, 46, 47],
        "pantry_plan": [43, 44, 45, 46, 47],
    }
    require(
        adaptive_outcomes.get("schema")
        == "paper-adaptive-mechanism-outcomes-qwen05b-v4"
        and adaptive_outcomes.get("status")
        == (
            "all terminal E89 reachable-controller seeds; balanced summaries "
            "only for complete five-seed domains"
        )
        and adaptive_outcomes.get("model") == "Qwen2.5-0.5B-Instruct"
        and adaptive_outcomes.get("target_ratio") == 0.015
        and adaptive_outcomes.get("target_steps") == 3072
        and adaptive_outcomes.get("domain_order") == ALIGNED_DOMAIN_ORDER
        and adaptive_outcomes.get("registered_seeds") == [43, 44, 45, 46, 47]
        and adaptive_outcomes.get("fields") == adaptive_fields
        and [cell.get("domain") for cell in adaptive_outcomes.get("cells", [])]
        == ALIGNED_DOMAIN_ORDER,
        "adaptive mechanism-to-outcome metadata drifted",
    )
    e89_gate = adaptive_outcomes.get("e89_mechanism_gate", {})
    require(
        e89_gate.get("status") == "failed"
        and e89_gate.get("decisive_registered_criterion") == 2
        and e89_gate.get("maximum_bound_hit_fraction") == 0.2
        and [
            (item.get("domain"), item.get("seed"))
            for item in e89_gate.get("bound_hit_violations", [])
        ] == [("python_factors", 46)]
        and {
            (item.get("domain"), item.get("seed"))
            for item in e89_gate.get("not_adapted_cells", [])
        } == {("python_factors", 46), ("python_factors", 47)}
        and [
            (item.get("domain"), item.get("seed"))
            for item in e89_gate.get("final_band_misses", [])
        ] == [("python_factors", 47)],
        "completed E89 cohort no longer records its preregistered gate failure",
    )
    failed_context = adaptive_outcomes.get("failed_rho05_context", {})
    require(
        failed_context.get("evidence") == "mechanism_only; no outcome comparison"
        and failed_context.get("status") == gate["status"]
        and failed_context.get("outcome") == gate["outcome"]
        and failed_context.get("target_ratio") == gate["target_ratio"]
        and failed_context.get("maximum_bound_hit_fraction")
        == gate["maximum_bound_hit_fraction"]
        and failed_context.get("records") == gate["records"]
        and failed_context.get("source_json")
        == "paper/figures/adaptive_semantic_gate_e88.json",
        "failed adaptive cohort is not retained as exact mechanism-only context",
    )
    for cell in adaptive_outcomes["cells"]:
        seeds = expected_adaptive_seeds[cell["domain"]]
        require(
            cell.get("n") == len(seeds)
            and cell.get("seeds") == seeds
            and cell.get("evidence")
            == (
                "balanced_five_seed_terminal"
                if len(seeds) == 5
                else "exact_terminal_prefix"
            )
            and set(cell.get("per_seed", {})) == {str(seed) for seed in seeds}
            and ("summaries" in cell) == (len(seeds) == 5),
            f"adaptive mechanism-to-outcome seed block drifted for "
            f"{cell['domain']}",
        )
        for seed in seeds:
            record = cell["per_seed"][str(seed)]
            mechanism = record.get("mechanism", {})
            fixed = record.get("fixed_endpoint", {})
            adaptive = record.get("adaptive_endpoint", {})
            effects = record.get("effects", {})
            require(
                set(mechanism)
                == {
                    "final_realized_ratio",
                    "final_coefficient",
                    "bound_hit_fraction",
                    "frozen_fraction",
                    "observations",
                    "updates_applied",
                    "global_step",
                    "source_line",
                    "final_within_factor_two",
                }
                and mechanism["global_step"] >= 3072
                and set(fixed) == {"pass8", "distinct8"}
                and set(adaptive) == {"pass8", "distinct8"}
                and set(effects)
                == {
                    "delta_pass8",
                    "delta_distinct8",
                    "delta_adjusted_breadth8",
                }
                and math.isclose(
                    effects["delta_pass8"],
                    adaptive["pass8"] - fixed["pass8"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    effects["delta_distinct8"],
                    adaptive["distinct8"] - fixed["distinct8"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    effects["delta_adjusted_breadth8"],
                    effects["delta_distinct8"] - effects["delta_pass8"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"adaptive mechanism-to-outcome value drifted for "
                f"{cell['domain']}/{seed}",
            )
        if len(seeds) == 5:
            require(
                set(cell["summaries"]) == set(adaptive_fields),
                f"adaptive mechanism summary fields drifted for {cell['domain']}",
            )
            for field, summary in cell["summaries"].items():
                values = [
                    (
                        cell["per_seed"][str(seed)]["mechanism"][field]
                        if field.startswith("final_")
                        else cell["per_seed"][str(seed)]["effects"][field]
                    )
                    for seed in seeds
                ]
                mean = statistics.fmean(values)
                half = (
                    2.7764451051977987
                    * statistics.stdev(values)
                    / math.sqrt(5)
                )
                require(
                    math.isclose(
                        summary["mean"], mean, rel_tol=0.0, abs_tol=1e-12
                    )
                    and all(
                        math.isclose(
                            left, right, rel_tol=0.0, abs_tol=1e-12
                        )
                        for left, right in zip(
                            summary["student_t_95"],
                            [mean - half, mean + half],
                        )
                    ),
                    f"adaptive mechanism interval drifted for "
                    f"{cell['domain']}/{field}",
                )
    require(
        adaptive_outcomes.get("input_sha256")
        and all(
            (ROOT / relative).is_file()
            and metadata.get("byte_length") == (ROOT / relative).stat().st_size
            and metadata.get("sha256") == file_sha256(ROOT / relative)
            for relative, metadata in adaptive_outcomes["input_sha256"].items()
        ),
        "adaptive mechanism-to-outcome source hash drifted",
    )

    for stem, expected_domains in REPLAY_DOSE_PROGRESS.items():
        paths = {
            suffix: COMPARISON_DIR / f"{stem}.{suffix}"
            for suffix in ("json", "pdf", "png")
        }
        for path in paths.values():
            require(
                path.is_file() and path.stat().st_size > 0,
                f"replay-dose progress asset {path.name} is missing or empty",
            )
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        require(
            payload.get("schema") == "paper-comparison-progress-figure-v2"
            and payload.get("evidence") == "progress"
            and payload.get("status")
            == "frozen progress snapshot; exact available paired subsets"
            and payload.get("paired_terminal_seeds_by_domain")
            == expected_domains,
            f"replay-dose progress provenance {stem} drifted",
        )
        require(
            payload.get("domains") == list(expected_domains)
            and set(payload.get("methods", []))
            == {"drgrpo", "replay_grpo", "adaptive_replay_grpo"},
            f"replay-dose progress identity {stem} drifted",
        )
        require(
            payload.get("five_seed_estimand_complete_by_domain")
            == {
                domain: len(seeds) == 5
                for domain, seeds in expected_domains.items()
            },
            f"replay-dose completeness flags drifted for {stem}",
        )
        for domain, seeds in expected_domains.items():
            record = payload["records"][domain]
            require(
                record["paired_seeds_by_pass"]["8.0"] == seeds
                and record["semantic_seeds_by_arm"]["bank_normalized_replay"]
                == seeds,
                f"replay-dose progress {stem}/{domain} is not exactly paired",
            )
    dose_strip = COMPARISON_DIR / "replay_dose_qwen05b_progress_static_strip"
    require(
        manuscript.count(r"\label{fig:replay-dose-progress}")
        == manuscript.count(
            str(dose_strip.with_suffix(".pdf").relative_to(ROOT / "paper"))
        ),
        "the replay-dose progress figure and its label must appear together",
    )


def check_aligned_domain_strips(manuscript: str) -> None:
    """Enforce domain columns and explicit model rows for cross-scale plots."""

    for obsolete in OBSOLETE_SPLIT_FIGURE_INCLUDES:
        require(
            obsolete not in manuscript,
            f"obsolete split trajectory figure remains compiled: {obsolete}",
        )
    for stem, expected in ALIGNED_DOMAIN_STRIPS.items():
        for suffix in ("json", "pdf", "png"):
            path = stem.with_suffix(f".{suffix}")
            require(
                path.is_file() and path.stat().st_size > 0,
                f"aligned domain strip is missing or empty: {path.name}",
            )
        include = str(stem.with_suffix(".pdf").relative_to(ROOT / "paper"))
        require(
            manuscript.count(include) <= 1,
            f"aligned domain strip {stem.name} is compiled more than once",
        )
        require(
            manuscript.count(rf"\label{{{expected['label']}}}") <= 1,
            f"aligned domain strip {stem.name} reuses its label",
        )
        payload = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        require(
            payload.get("schema") == "paper-aligned-domain-strip-v2"
            and payload.get("layout") == expected["layout"]
            and payload.get("environment_columns") == ALIGNED_DOMAIN_ORDER
            and list(payload.get("panels", {})) == ALIGNED_DOMAIN_ORDER,
            f"aligned domain strip {stem.name} drifted from its row contract",
        )
        require(
            payload.get("model_rows") == ALIGNED_MODEL_ORDER,
            f"{stem.name} does not preserve Qwen-0.5B, Falcon-1B, Qwen-3B rows",
        )
        require(
            payload.get("comparison") == expected["comparison"]
            and set(payload.get("methods", [])) == expected["methods"],
            f"aligned domain strip {stem.name} changed scientific identity",
        )
        for domain in ALIGNED_DOMAIN_ORDER:
            panel = payload["panels"][domain]
            actual = {
                entry["scale"]: entry["minimum_paired_seed_count_by_method"]
                for entry in panel.get("available", [])
            }
            expected_scales = expected["available"][domain]
            cells = panel.get("cells", {})
            require(
                list(cells) == ALIGNED_MODEL_ORDER
                and all(
                    cells[scale].get("status")
                    == ("available" if scale in actual else "blank")
                    for scale in ALIGNED_MODEL_ORDER
                ),
                f"{stem.name}/{domain} does not preserve blank model cells",
            )
            require(
                set(actual) == set(expected_scales),
                f"{stem.name}/{domain} has the wrong model columns",
            )
            for scale, count in expected_scales.items():
                if isinstance(count, dict):
                    require(
                        actual[scale] == count,
                        f"{stem.name}/{domain}/{scale} has the wrong method n",
                    )
                else:
                    require(
                        set(actual[scale]) == expected["methods"]
                        and set(actual[scale].values()) == {count},
                        f"{stem.name}/{domain}/{scale} has the wrong paired n",
                    )

        for source in payload.get("sources", []):
            source_path = ROOT / source["path"]
            require(
                source_path.is_file()
                and file_sha256(source_path) == source.get("sha256"),
                f"{stem.name} source hash drifted: {source.get('path')}",
            )
        if expected["comparison"] == "direct_baselines":
            expected_source_counts = {
                (scale, domain, method): count
                for domain, scales in DIRECT_AVAILABLE.items()
                for scale, methods in scales.items()
                for method, count in methods.items()
            }
            # Qwen Pantry RLEP has five scientific seed cells, but the two
            # recovered cells retain both their pre- and post-recovery
            # trajectory files in provenance.  Those seven physical sources
            # must not be mistaken for seven independent science cells.
            expected_source_counts[("qwen05b", "pantry_plan", "rlep_dr")] = 7
            # Falcon Pantry likewise retains one pre-repair and one terminal
            # recovery trajectory for each of its five scientific seeds.
            expected_source_counts[("falcon1b", "pantry_plan", "rlep_dr")] = 10
            actual_source_counts = {
                key: sum(
                    1
                    for source in payload.get("trajectory_sources", [])
                    if (
                        source.get("scale"),
                        source.get("domain"),
                        source.get("method"),
                    ) == key
                )
                for key in expected_source_counts
            }
            require(
                payload.get("evidence")
                == (
                    "all terminal registered UCPO cells and every terminal "
                    "sparse RLEP-Dr seed; exact n is recorded per method and domain"
                )
                and actual_source_counts == expected_source_counts
                and all(
                    (ROOT / source["path"]).is_file()
                    and source.get("byte_length", 0) > 0
                    and len(source.get("sha256", "")) == 64
                    for source in payload.get("trajectory_sources", [])
                ),
                "direct-baseline strip lost terminal evidence or provenance",
            )


@lru_cache(maxsize=None)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def file_prefix_sha256(path: Path, byte_length: int) -> str:
    digest = hashlib.sha256()
    remaining = byte_length
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                return ""
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise SystemExit(f"Paper figure contract failed: missing assignment {name}")


def pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


B1B_SUMMARY = ROOT / "var/artifacts/e72_b1b_summary.json"
E73_SUMMARY = ROOT / "var/artifacts/e73_falcon_cross_family_summary_v2.json"
REHEARSAL_TABLE_ROWS = {
    "Graph coloring": ("graph_coloring", "E"),
    "Countdown": ("countdown", "E"),
    "Python factors": ("python_factors", "rehearsal better"),
    "MathIR action menu": ("mathir", "E"),
    "PantryPlan": ("pantry_plan", "balance better"),
}


def check_rehearsal_table(manuscript: str) -> None:
    """Bind the clean rehearsal/replay-plus-balance table to its artifact.

    The binding applies only while the manuscript prints that table; the
    condensed method section no longer does, and a table that is not printed
    has no cells to hold against the artifact.
    """
    if not B1B_SUMMARY.is_file():
        return
    if r"\label{tab:rehearsal-ablation}" not in manuscript:
        return
    summary = json.loads(B1B_SUMMARY.read_text())
    require(
        summary.get("all_25_cells_terminal_and_valid") is True,
        "rehearsal table is present without 25 valid terminal cells",
    )
    by_domain = {entry["domain"]: entry for entry in summary["domains"]}
    section = manuscript.split(r"\label{tab:rehearsal-ablation}", 1)[0]
    section = section.rsplit(r"\begin{tabular}", 1)[-1]

    def decimal(value: float) -> str:
        rendered = f"{value:.3f}"
        return rendered[1:] if rendered.startswith("0.") else rendered

    for title, (domain, effect) in REHEARSAL_TABLE_ROWS.items():
        entry = by_domain.get(domain)
        require(
            entry is not None and entry.get("reportable"),
            f"rehearsal table shows {title} but its domain is withheld",
        )
        rows = re.findall(
            rf"^\s*{re.escape(title)}\s*&.*?\\\\",
            section,
            flags=re.MULTILINE | re.DOTALL,
        )
        require(len(rows) == 1, f"rehearsal table is missing {title}")
        cells = [cell.strip() for cell in rows[0].split("&")]
        expected = (
            decimal(entry["means"]["drgrpo"]["distinct8"]),
            decimal(entry["means"]["b1b"]["distinct8"]),
            decimal(entry["means"]["b1a"]["distinct8"]),
            effect,
        )
        printed = (
            cells[1],
            cells[2],
            cells[3],
            cells[4].split(r"\\", 1)[0].strip(),
        )
        require(
            printed == expected,
            f"rehearsal table {title} shows {printed!r}, artifact says {expected!r}",
        )


def check_cross_family_table(manuscript: str) -> None:
    'Bind the terminal Falcon table to all fifty audited cells.'
    if not E73_SUMMARY.is_file():
        return
    summary = json.loads(E73_SUMMARY.read_text())
    require(
        summary.get("all_50_cells_terminal_and_valid") is True,
        "cross-family table is present without 50 valid terminal cells",
    )
    outcomes = summary.get("registered_directional_outcomes", {})
    require(
        outcomes.get("distinct_positive_domains") == 5
        and outcomes.get("pass8_nonnegative_domains") == 5
        and outcomes.get("distinct_paired_wins") == 25
        and outcomes.get("pass8_paired_wins") == 25,
        "cross-family directional claim does not match the terminal artifact",
    )
    if r"\label{tab:cross-family}" not in manuscript:
        return
    section = manuscript.split(r"\label{tab:cross-family}", 1)[0]
    section = section.rsplit(r"\begin{tabular}", 1)[-1]
    titles = {
        "Graph coloring": "graph_coloring",
        "Countdown": "countdown",
        "Python factors": "python_factors",
        "MathIR": "mathir",
        "PantryPlan": "pantry_plan",
    }

    def clean(cell: str) -> str:
        cell = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cell)
        return " ".join(cell.split())

    def decimal3(value: float) -> str:
        rendered = f"{value:.3f}"
        return rendered[1:] if rendered.startswith("0.") else rendered

    for title, domain in titles.items():
        entry = summary["domains"].get(domain)
        require(
            entry is not None and entry.get("complete")
            and entry.get("common_endpoint") == 4608,
            f"cross-family table shows {title} without a terminal domain",
        )
        match = re.search(
            rf"^\s*{re.escape(title)}\s*&\s*matched Dr\.GRPO\s*&"
            rf"(.*?)\\\\\s*&\s*\\historicalt\{{\}}\s*&(.*?)\\\\",
            section,
            flags=re.MULTILINE | re.DOTALL,
        )
        require(match is not None, f"cross-family table is missing {title}")
        printed = tuple(
            clean(cell)
            for group in match.groups()
            for cell in group.split("&")
        )
        expected_cells: list[str] = []
        for arm in ("drgrpo", "xgrpo"):
            means = entry["means"][arm]
            passed = means["pass_at_8"]
            expected_cells.extend(
                (
                    decimal3(means["pass_at_1"]),
                    decimal3(means["mean_at_8"]),
                    decimal3(passed),
                    f"{means['distinct_at_8']:.2f}",
                    f"{means['distinct_at_8'] / passed:.2f}" if passed else "--",
                )
            )
        expected = tuple(expected_cells)
        require(
            printed == expected,
            f"cross-family table {title} shows {printed!r}, "
            f"artifact says {expected!r}",
        )


# Sixth copy of this map in the repository, and the fifth to break when a new
# domain appeared. It is imported from one place now; the contract keeps its own
# name for the table it validates.
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parent / "exp_scaling"))
from status_e78 import DOMAIN_TITLES as INTERIM_DOMAIN_TITLES  # noqa: E402


def check_historical_interim_replay_table(manuscript: str) -> None:
    """Bind the explicitly interim main-text table to its dated snapshot."""

    require(
        r"\label{tab:interim-replay}" in manuscript,
        "dated interim table is absent from the manuscript",
    )
    require(INTERIM_TABLE.is_file(), "dated interim four-metric table JSON is missing")
    payload = json.loads(INTERIM_TABLE.read_text())
    require(
        payload.get("schema") == "figure4_interim_four_metric_table_v1",
        "dated interim table has the wrong schema",
    )
    figure_snapshot = json.loads(INTERIM_FIGURE.read_text(encoding="utf-8"))
    require(
        payload.get("figure_snapshot_generated_at")
        == figure_snapshot.get("generated_at"),
        "dated interim table snapshot identity drifted",
    )
    label_position = manuscript.index(r"\label{tab:interim-replay}")
    table_start = manuscript.rfind(r"\begin{table}", 0, label_position)
    table_end = manuscript.index(r"\end{table}", label_position)
    require(table_start >= 0, "interim table environment is missing")
    table = manuscript[table_start:table_end]
    table = table.split(r"\begin{tabular}", 1)[-1]
    blocks = {
        "Qwen2.5-0.5B": table.split(r"\qwenmark{}2.5-0.5B", 1)[1].split(
            "Falcon3-1B", 1
        )[0],
        "Falcon3-1B": table.split("Falcon3-1B", 1)[1].split(
            r"\qwenmark{}2.5-3B", 1
        )[0],
        "Qwen2.5-3B": table.split(r"\qwenmark{}2.5-3B", 1)[1],
    }

    def decimal3(value: float) -> str:
        rendered = f"{value:.3f}"
        if rendered.startswith("0."):
            return rendered[1:]
        if rendered.startswith("-0."):
            return "-" + rendered[2:]
        return rendered

    expected_rows = 0
    for family, family_record in payload["families"].items():
        for domain, domain_record in family_record["domains"].items():
            expected_rows += 1
            title = INTERIM_DOMAIN_TITLES[domain]
            match = re.search(
                rf"&\s*{re.escape(title)}\s*&(.*?)\\\\",
                blocks[family],
            )
            require(match is not None, f"interim table is missing {family}/{title}")
            raw_cells = tuple(
                " ".join(cell.split()) for cell in match.group(1).split("&")
            )
            highlighted = tuple(r"\bettercell{" in cell for cell in raw_cells)
            printed = tuple(
                re.sub(r"\\bettercell\{([^{}]+)\}", r"\1", cell)
                for cell in raw_cells
            )
            expected_cells = [
                f"{len(domain_record['paired_seeds'])}@{float(domain_record['pass']):.1f}"
            ]
            for arm in ("control", "replay"):
                means = domain_record["arms"][arm]["means"]
                for metric in ("pass1", "pass8", "mean8", "distinct8"):
                    value = means.get(metric)
                    expected_cells.append("--" if value is None else decimal3(float(value)))
            expected = tuple(expected_cells)
            require(
                not any(highlighted[:5]),
                f"interim table shades a non-treatment cell in {family}/{title}",
            )
            control_means = domain_record["arms"]["control"]["means"]
            replay_means = domain_record["arms"]["replay"]["means"]
            for metric_index, metric in enumerate(
                ("pass1", "pass8", "mean8", "distinct8")
            ):
                control_value = control_means.get(metric)
                replay_value = replay_means.get(metric)
                should_highlight = (
                    control_value is not None
                    and replay_value is not None
                    and float(replay_value) > float(control_value)
                )
                require(
                    highlighted[5 + metric_index] == should_highlight,
                    f"interim table highlight is wrong for {family}/{title}/{metric}",
                )
            require(
                printed == expected,
                f"interim table {family}/{title} shows {printed!r}, "
                f"snapshot says {expected!r}",
            )
    allowed_blank_rows = 0
    # The loop above proves every snapshot panel appears in the table. Count
    # the printed rows to prove the converse, so a panel that leaves the
    # snapshot cannot survive as a stale row. Binding this to the snapshot
    # rather than to a fixed count lets newly evaluable panels through, which
    # is the whole point of an interim table.
    printed_rows = sum(len(re.findall(r"\\\\", block)) for block in blocks.values())
    require(
        printed_rows == expected_rows + allowed_blank_rows,
        f"interim table prints {printed_rows} rows, snapshot has {expected_rows}",
    )


def check_core_terminal_table(manuscript: str) -> None:
    """Bind the current matched endpoint table to its generated JSON."""

    payload_path = ROOT / "paper/results/core_terminal_endpoints.json"
    body_path = ROOT / "paper/results/core_terminal_endpoints_table_body.tex"
    require(
        r"\label{tab:core-terminal-endpoints}" in manuscript,
        "current core terminal table is absent from the manuscript",
    )
    require(
        r"\input{results/core_terminal_endpoints_table_body.tex}" in manuscript,
        "current core terminal table does not input its generated body",
    )
    require(
        payload_path.is_file() and body_path.is_file(),
        "current core terminal table artifacts are missing",
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    require(
        payload.get("schema") == "paper-core-terminal-endpoints-v1"
        and payload.get("expected_draws") == 4
        and payload.get("domains") == ALIGNED_DOMAIN_ORDER,
        "current core terminal endpoint schema or grid drifted",
    )
    for source in payload.get("sources", {}).values():
        source_path = Path(source.get("path", ""))
        require(
            source_path.is_file()
            and source.get("sha256") == file_sha256(source_path),
            f"current core terminal source hash drifted: {source_path}",
        )

    model_labels = {
        "Qwen2.5-0.5B": r"\qwenmark{}2.5-0.5B",
        "Falcon3-1B": "Falcon3-1B",
        "Qwen2.5-3B": r"\qwenmark{}2.5-3B",
    }
    domain_labels = {
        "graph_coloring": "Graph coloring",
        "countdown": "Countdown",
        "python_factors": "Python factors",
        "mathir": "MathIR",
        "pantry_plan": "PantryPlan",
    }
    lines = []
    for model, model_record in payload["models"].items():
        first = True
        for domain in ALIGNED_DOMAIN_ORDER:
            record = model_record["domains"].get(domain)
            methods = record.get("methods", {}) if record else {}
            if not {"control", "replay"} <= set(methods):
                continue
            control = methods["control"]["per_seed"]
            replay = methods["replay"]["per_seed"]
            paired = sorted(set(control) & set(replay), key=int)
            if not paired:
                continue
            control_values = [
                statistics.fmean(control[seed][metric] for seed in paired)
                for metric in ("pass8", "mean8", "distinct8")
            ]
            replay_values = [
                statistics.fmean(replay[seed][metric] for seed in paired)
                for metric in ("pass8", "mean8", "distinct8")
            ]
            rendered = []
            for index, value in enumerate(control_values + replay_values):
                cell = f"{value:.3f}"
                if cell.startswith("0."):
                    cell = cell[1:]
                elif cell.startswith("-0."):
                    cell = "-" + cell[2:]
                if index >= 3 and value > control_values[index - 3]:
                    cell = rf"\bettercell{{{cell}}}"
                rendered.append(cell)
            model_cell = model_labels[model] if first else ""
            first = False
            lines.append(
                f"    {model_cell} & {domain_labels[domain]} & "
                f"{len(paired)}@{float(record['training_pass']):.1f} & "
                + " & ".join(rendered)
                + r" \\"
            )
        lines.append(r"    \addlinespace[2pt]")
    expected_body = "\n".join(lines[:-1]) + "\n    \\bottomrule\n"
    require(
        body_path.read_text(encoding="utf-8") == expected_body,
        "current core terminal table body drifted from its endpoint JSON",
    )
    python_methods = payload["models"]["Qwen2.5-3B"]["domains"][
        "python_factors"
    ]["methods"]
    require(
        sorted(python_methods["control"]["per_seed"], key=int)
        == ["70", "71", "72", "73", "74"]
        and sorted(python_methods["replay"]["per_seed"], key=int)
        == ["70", "71", "72", "73", "74"],
        "current Qwen2.5-3B Python independent endpoint counts drifted",
    )


def check_maxent_factorial_tables(manuscript: str) -> None:
    """Bind both semantic-MaxEnt tables to the completed 5x2x2 artifact."""

    for path in (MAXENT_FACTORIAL, MAXENT_EFFECT_BODY, MAXENT_UNCERTAINTY_BODY):
        require(path.is_file(), f"MaxEnt paper artifact is missing: {path.name}")
    payload = json.loads(MAXENT_FACTORIAL.read_text(encoding="utf-8"))
    require(
        payload.get("schema") == "maxent_factorial_05b_paper_results_v1",
        "fixed-MaxEnt factorial has the wrong schema",
    )
    design = payload.get("design", {})
    require(
        design.get("paired_seeds") == [43, 44, 45, 46, 47]
        and design.get("arms")
        == ["control", "replay", "semantic_only", "semantic"]
        and design.get("evaluation_draws") == 4
        and design.get("target_steps") == 3072,
        "fixed-MaxEnt factorial does not contain the completed registered design",
    )
    titles = {
        "graph_coloring": "Graph coloring",
        "countdown": "Countdown",
        "python_factors": "Python factors",
        "mathir": "MathIR",
        "pantry_plan": "PantryPlan",
    }
    require(
        set(payload.get("domains", {})) == set(titles),
        "fixed-MaxEnt factorial has the wrong domain set",
    )

    def signed(value: float) -> str:
        return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")

    def ci_cell(summary: dict) -> str:
        low, high = summary["student_t_95"]
        return rf"${signed(summary['mean'])}\;[{signed(low)},{signed(high)}]$"

    effect_rows = []
    uncertainty_rows = []
    for domain, title in titles.items():
        contrasts = payload["domains"][domain].get("contrasts", {})
        require(
            set(contrasts)
            == {
                "maxent_without_replay",
                "maxent_with_replay",
                "factorial_interaction",
            },
            f"{domain}: incomplete fixed-MaxEnt contrast set",
        )
        for contrast in contrasts.values():
            for metric in ("pass8", "distinct8", "adjusted_breadth"):
                require(
                    set(contrast[metric]["per_seed"])
                    == {"43", "44", "45", "46", "47"},
                    f"{domain}/{metric}: fixed-MaxEnt seed set is incomplete",
                )
        no_replay = contrasts["maxent_without_replay"]
        with_replay = contrasts["maxent_with_replay"]
        interaction = contrasts["factorial_interaction"]
        effect_rows.append(
            "    "
            + title
            + " & "
            + " & ".join(
                signed(summary["mean"])
                for summary in (
                    no_replay["pass8"],
                    no_replay["distinct8"],
                    no_replay["adjusted_breadth"],
                    with_replay["pass8"],
                    with_replay["distinct8"],
                    with_replay["adjusted_breadth"],
                    interaction["distinct8"],
                )
            )
            + r" \\"
        )
        uncertainty_rows.append(
            "    "
            + title
            + " & "
            + " & ".join(
                ci_cell(summary)
                for summary in (
                    no_replay["distinct8"],
                    with_replay["distinct8"],
                    interaction["distinct8"],
                )
            )
            + r" \\"
        )
    require(
        MAXENT_EFFECT_BODY.read_text(encoding="utf-8")
        == "\n".join([*effect_rows, r"    \bottomrule"]) + "\n",
        "fixed-MaxEnt effect table body drifted from its JSON artifact",
    )
    require(
        MAXENT_UNCERTAINTY_BODY.read_text(encoding="utf-8")
        == "\n".join([*uncertainty_rows, r"    \bottomrule"]) + "\n",
        "fixed-MaxEnt uncertainty table body drifted from its JSON artifact",
    )
    maxent_tokens = (
        r"\label{tab:maxent-factorial-effects}",
        r"\label{tab:maxent-factorial-uncertainty}",
        r"\input{results/maxent_factorial_05b_table_body.tex}",
        r"\input{results/maxent_factorial_05b_uncertainty_table_body.tex}",
    )
    if any(token in manuscript for token in maxent_tokens):
        for token in maxent_tokens:
            require(token in manuscript, f"manuscript is missing MaxEnt token {token!r}")


def check_ucpo_interim_table(manuscript: str) -> None:
    """Bind completed direct-comparator evidence to its forest and strip."""

    payload = json.loads(
        DIRECT_COMPARATOR_ENDPOINT.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    cells = {
        (cell["model"], cell["domain"]): cell
        for cell in payload.get("cells", [])
    }
    ucpo = [
        (model, domain, cell["methods"]["ucpo"])
        for (model, domain), cell in cells.items()
        if "ucpo" in cell.get("methods", {})
    ]
    grpo = [
        (model, domain, cell["methods"]["grpo"])
        for (model, domain), cell in cells.items()
        if "grpo" in cell.get("methods", {})
    ]
    rlep = [
        (model, domain, cell["methods"]["rlep_dr"])
        for (model, domain), cell in cells.items()
        if "rlep_dr" in cell.get("methods", {})
    ]
    require(
        payload.get("schema")
        == "paper-direct-comparator-endpoint-effects-v2"
        and len(ucpo) == 10
        and sum(record["n"] == 5 for _model, _domain, record in ucpo) == 10
        and [(model, domain, record["seeds"]) for model, domain, record in ucpo if record["n"] < 5]
        == []
        and sum(record["n"] == 5 for _model, _domain, record in grpo) == 15
        and [
            (model, domain, record["seeds"])
            for model, domain, record in grpo
            if record["n"] < 5
        ] == []
        and sum(record["n"] == 5 for _model, _domain, record in rlep) == 9
        and [
            (model, domain, record["seeds"])
            for model, domain, record in rlep
            if record["n"] < 5
        ] == [
            ("Falcon3-1B", "python_factors", [55, 57]),
        ],
        "completed UCPO/RLEP direct-comparator coverage drifted",
    )
    for _model, _domain, record in [*grpo, *ucpo, *rlep]:
        require(
            all(
                set(seed_record) >= {"baseline", "comparator", "effect"}
                and math.isclose(
                    seed_record["effect"]["adjusted_breadth8"],
                    (
                        seed_record["comparator"]["distinct8"]
                        - seed_record["comparator"]["pass8"]
                        - seed_record["baseline"]["distinct8"]
                        + seed_record["baseline"]["pass8"]
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for seed_record in record["per_seed"].values()
            ),
            "direct-comparator per-seed endpoint algebra drifted",
        )
    for token in (
        r"\label{fig:direct-comparator-effects}",
        r"{figures/direct_comparator_endpoint_effects.pdf}",
        r"\label{fig:direct-baseline-curves}",
        r"{figures/direct_baseline_learning_curves_static_strip.pdf}",
        "none of the five Qwen2.5-3B GRPO blocks supports a positive matched effect",
        r"Complete Qwen Countdown UCPO has a mean \texttt{pass@8} effect of $+.133$",
        "Qwen MathIR RLEP-Dr has a positive correctness effect, $+.126$",
        "plus Falcon Python at $n=2$",
        "local rather than uniform",
    ):
        require(
            " ".join(token.split()) in " ".join(manuscript.split()),
            f"direct-comparator boundary missing {token!r}",
        )

def check_program_status(manuscript: str) -> None:
    for path in (PROGRAM_STATUS, PROGRAM_STATUS_BODY):
        require(
            path.is_file() and path.stat().st_size > 0,
            f"paper program artifact missing: {path.name}",
        )
    payload = json.loads(PROGRAM_STATUS.read_text(encoding="utf-8"))
    require(
        payload.get("schema") == "modebench-paper-program-status-v1"
        and payload.get("shape") == {
            "methods": 10,
            "scales": 3,
            "domains": 5,
            "seeds_per_scale": 5,
            "cells": 750,
        },
        "paper program registry has the wrong schema or matrix shape",
    )
    rows = payload.get("methods", [])
    require(
        len(rows) == 10 and len({row.get("key") for row in rows}) == 10,
        "paper program registry must contain ten unique methods",
    )
    require(
        all(
            row.get("target") == 75
            and row.get("registered", -1) + row.get("missing", -1) == 75
            and 0 <= row.get("terminal", -1) <= row.get("registered", -1)
            for row in rows
        ),
        "paper program method counts do not partition their 75-cell targets",
    )
    require(
        payload.get("registered") == sum(row["registered"] for row in rows)
        and payload.get("terminal") == sum(row["terminal"] for row in rows)
        and payload.get("missing") == sum(row["missing"] for row in rows)
        and payload["registered"] + payload["missing"] == 750,
        "paper program summary does not match its method rows",
    )
    by_key = {row["key"]: row for row in rows}
    require(
        by_key["adaptive_semantic_maxent"]["registered"] == 0
        and by_key["adaptive_semantic_maxent"]["missing"] == 75,
        "adaptive Semantic MaxEnt without replay must remain explicitly absent",
    )
    grpo = by_key["grpo"]
    require(
        grpo["registered"] == 75
        and grpo["terminal"] == 75
        and grpo["active"] == 0,
        "GRPO registry must retain its registered cells",
    )

    rlep = by_key["rlep_dr"]
    ucpo = by_key["ucpo"]
    require(
        ucpo["registered"] == 75
        and ucpo["terminal"] == 50
        and ucpo["active"] == 25
        and
        rlep["registered"] == 75
        and rlep["terminal"] == 47
        and rlep["active"] == 25
        and rlep["blocked"] == 3
        and rlep["failed"] == 0
        and sum(
            rlep[field] for field in
            ("terminal", "active", "blocked", "partial", "failed", "inactive")
        ) == 75,
        "UCPO/RLEP program rows drifted from their current terminal coverage",
    )
    body = PROGRAM_STATUS_BODY.read_text(encoding="utf-8")
    require(body.count(r" \\") == 10, "program table must contain ten method rows")
    for row in rows:
        require(
            rf"\textbf{{{row['label']}}}" in body,
            f"program table is missing {row['label']}",
        )
    for token in (
        r"\label{app:program-registry}",
        r"\label{tab:program-status}",
        r"\input{results/paper_program_status_table_body.tex}",
        "not one completed factorial",
        "Adaptive Re:Dr.GRPO",
        "Adaptive Semantic MaxEnt without replay",
        "Figures expose every available sampled checkpoint",
        "no five-seed display gate",
    ):
        require(token in manuscript, f"program registry boundary missing {token!r}")


def check_scientific_method_scope(manuscript: str) -> None:
    collapsed = " ".join(manuscript.split())
    for token in (
        r"\label{app:program-registry}",
        r"\subsection{Comparison registry}",
        "Level 1 covers five domains at Qwen2.5-0.5B, Falcon3-1B, and Qwen2.5-3B",
        "remain direct comparators rather than components of the primary factorial",
        "Seed prefixes are labeled with exact $n$ and are never promoted to "
        "five-seed estimates",
    ):
        require(
            token in collapsed,
            f"scientific method-scope boundary missing {token!r}",
        )
    for forbidden in (
        r"\label{tab:program-status}",
        r"\input{results/paper_program_status_table_body.tex}",
        "Snapshot boundary",
        "Runs continue after this cutoff",
    ):
        require(
            forbidden not in manuscript,
            f"live program-status framing returned: {forbidden!r}",
        )


def check_dapo_progress(manuscript: str) -> None:
    if "DAPO" not in manuscript:
        return
    require(
        DAPO_PROGRESS.is_file() and DAPO_PROGRESS.stat().st_size > 0,
        "official DAPO progress artifact is missing",
    )
    payload = json.loads(DAPO_PROGRESS.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    require(
        payload.get("schema") == "paper-e113r4-official-dapo-progress-v1"
        and payload.get("registered_science_cells") == 50
        and payload.get("terminal_science_cells") == 4
        and payload.get("nonterminal_science_cells") == 46
        and payload.get("running_science_cells") == 0
        and payload.get("pending_science_cells") == 46
        and payload.get("failed_science_cells") == 0
        and payload.get("other_nonterminal_science_cells") == 0,
        "official DAPO progress partition drifted from the paper snapshot",
    )
    expected = {
        ("qwen05b", "graph_coloring", 43): 0.2734375,
        ("qwen05b", "graph_coloring", 44): 0.265625,
        ("qwen05b", "graph_coloring", 45): 0.2734375,
        ("qwen05b", "graph_coloring", 46): 0.265625,
    }
    require(
        len(records) == 4
        and {
            (row.get("family"), row.get("domain"), row.get("seed")):
            row.get("final_upstream_validation_acc_at_1")
            for row in records
        }
        == expected
        and all(
            row.get("accepted_training_steps") == 24
            and row.get("standardized_pass_at_8_available") is False
            and row.get("standardized_breadth_at_8_available") is False
            and row.get("evidence_class")
            == "terminal_upstream_acc_at_1_diagnostic"
            for row in records
        )
        and "mean" not in payload
        and "interval" not in payload,
        "DAPO diagnostic was aggregated, relabelled, or changed cells",
    )
    manuscript_flat = " ".join(manuscript.split())
    for token in (
        "available artifacts contain four Qwen2.5-0.5B Graph runs",
        r"standardized eight-sample \texttt{pass@8}",
        "DAPO therefore contributes no point, aggregate, interval",
    ):
        require(
            token in manuscript_flat,
            f"DAPO evidence boundary missing {token!r}",
        )



def main() -> None:
    source = SOURCE.read_text()
    manuscript = MANUSCRIPT.read_text()
    audit = json.loads(AUDIT.read_text())
    maxrl_source = MAXRL_SOURCE.read_text()
    maxrl_audit = json.loads(MAXRL_AUDIT.read_text())
    dated_snapshot = (
        r"\label{fig:interim-replay}" in manuscript
        and r"\textbf{Snapshot boundary.}" in manuscript
    )
    main_page_limit = 10 if dated_snapshot else 9
    page_contract = "snapshot limit 10" if dated_snapshot else "final limit 9"
    example_source = EXAMPLES_SOURCE.read_text()
    require("excess@" not in manuscript, "excess@K remains in manuscript")
    require(r"\paragraph{" not in manuscript, "compact bold headings regressed")
    require(ICLR_STYLE.is_file(), "official ICLR 2027 style file is missing")
    require(ICLR_BIB_STYLE.is_file(), "official ICLR 2027 bibliography style is missing")
    require(ICLR_FANCYHDR.is_file(), "official ICLR 2027 fancyhdr dependency is missing")
    require(ICLR_NATBIB.is_file(), "official ICLR 2027 natbib dependency is missing")
    require(
        hashlib.sha256(ICLR_STYLE.read_bytes()).hexdigest() == ICLR_STYLE_SHA256,
        "ICLR 2027 style file differs from the official archive",
    )
    require(
        hashlib.sha256(ICLR_BIB_STYLE.read_bytes()).hexdigest()
        == ICLR_BIB_STYLE_SHA256,
        "ICLR 2027 bibliography style differs from the official archive",
    )
    require(
        hashlib.sha256(ICLR_FANCYHDR.read_bytes()).hexdigest()
        == ICLR_FANCYHDR_SHA256,
        "ICLR 2027 fancyhdr dependency differs from the official archive",
    )
    require(
        hashlib.sha256(ICLR_NATBIB.read_bytes()).hexdigest() == ICLR_NATBIB_SHA256,
        "ICLR 2027 natbib dependency differs from the official archive",
    )
    require(
        r"\usepackage{iclr2027_conference,times}" in manuscript,
        "paper is not using the ICLR 2027 template",
    )
    require(
        r"\bibliographystyle{iclr2027_conference}" in manuscript,
        "paper is not using the ICLR 2027 bibliography style",
    )
    require("iclr2026_conference" not in manuscript, "ICLR 2026 template remains")
    require(
        r"\setcitestyle{numbers" not in manuscript,
        "numeric citations override ICLR 2027's required author-year natbib style",
    )
    require(r"\cite{" not in manuscript, "plain cite command remains; use citep or citet")
    # Float and caption spacing are set once in the preamble and are what keeps
    # main content inside the nine-page limit checked below; the template's own
    # defaults cost two pages. Body-text spacing and one-off \vspace nudges stay
    # forbidden, since those are the overrides that hide overflow locally.
    for token in (
        r"\setlength{\parskip}",
        r"\vspace{-",
    ):
        require(token not in manuscript, f"non-template spacing override remains: {token}")
    compiled_manuscript = re.sub(
        r"\\iffalse.*?\\fi", "", manuscript, flags=re.DOTALL
    )
    table_blocks = re.findall(
        r"\\begin\{table\}.*?\\end\{table\}",
        compiled_manuscript,
        flags=re.DOTALL,
    )
    require(table_blocks, "manuscript contains no compiled tables")
    for index, block in enumerate(table_blocks, start=1):
        caption_position = block.find(r"\caption{")
        content_positions = tuple(
            position
            for token in (r"\begin{tabular}", r"\begin{tabularx}", r"\resizebox")
            if (position := block.find(token)) >= 0
        )
        require(caption_position >= 0, f"compiled table {index} has no caption")
        require(content_positions, f"compiled table {index} has no tabular content")
        require(
            caption_position < min(content_positions),
            f"compiled table {index} title is below the table; ICLR 2027 requires it above",
        )
    figure_blocks = re.findall(
        r"\\begin\{figure\}.*?\\end\{figure\}",
        compiled_manuscript,
        flags=re.DOTALL,
    )
    for index, block in enumerate(figure_blocks, start=1):
        require(
            block.find(r"\includegraphics") < block.find(r"\caption{"),
            f"compiled figure {index} title is not below the figure",
        )
    require("neurips_2025" not in manuscript, "NeurIPS template remains in manuscript")
    abstract = manuscript.split(r"\begin{abstract}", 1)[1].split(
        r"\end{abstract}", 1
    )[0]
    title_match = re.search(r"\\title\{([^{}]*)\}", manuscript, flags=re.DOTALL)
    require(title_match is not None, "manuscript title is missing or unexpectedly nested")
    title_block = title_match.group(1)
    require(
        title_block.count(r"\\") <= 1,
        "title exceeds the requested two-line maximum",
    )
    require(
        title_block.count(r"\\") == 1,
        "page-1 title should preserve the reviewed two-line composition",
    )
    abstract_plain = re.sub(r"\\[A-Za-z]+", "", abstract)
    abstract_words = re.findall(r"[A-Za-z0-9@.+-]+", abstract_plain)
    require(len(abstract_words) <= 211, f"abstract has {len(abstract_words)} words")
    introduction = manuscript.split(r"\section{Introduction}", 1)[1].split(
        r"\section{Related Work}", 1
    )[0]
    introduction_flat = " ".join(introduction.split())
    introduction_plain = re.sub(
        r"\\begin\{figure\}.*?\\end\{figure\}",
        "",
        introduction,
        flags=re.DOTALL,
    )
    introduction_words = re.findall(
        r"\b[A-Za-z][A-Za-z0-9@.+-]*\b", introduction_plain
    )
    require(
        len(introduction_words) <= 545,
        f"Introduction exceeds the 545-word narrative budget: {len(introduction_words)}",
    )
    for token in (
        "RLVR) scales outcome supervision",
        "A policy can therefore become more accurate as its support over correct behavior contracts",
        "This limitation weakens inference-time scaling",
        "additional samples reproduce the familiar",
        "RLVR lacks the execution-grounded identity needed to support both",
        "We make three contributions",
        r"\textbf{ModeBench measures verified solution support.}",
        r"\textbf{Re:MaxRL explores and preserves.}",
        r"\textbf{Matched evidence isolates the value of memory.}",
        "Re:MaxRL exceeds MaxRL in mean",
        "mnih2015human",
        "lin1992selfimproving",
        "ecoffet2021return",
        "mouret2015illuminating",
        "tajwar2026maxrl",
        r"\ref{fig:story}",
    ):
        require(token in introduction_flat, f"Introduction structure missing {token!r}")
    for forbidden in (r"V(y,s_x)", r"D_K(x)", r"P_K(x)=", "Summary of contributions"):
        require(
            forbidden not in introduction,
            f"Introduction contains stale or overly formal token {forbidden!r}",
        )
    for forbidden in (
        "flattens what it means to be right",
        "plurality matters everywhere",
        "correct-answer effect",
        "open-set executable task set",
        "semantic identity binds correctness",
        "common retention mechanism",
    ):
        require(
            forbidden not in introduction,
            f"Introduction retains superseded wording {forbidden!r}",
        )
    # Anchored on the section, not on a "Conclusion." run-in: discussion,
    # limitations, and the closing argument are now bolded paragraphs of one
    # Conclusion section, which spends two section rules on text instead.
    require(
        r"\section{Conclusion}" in manuscript,
        "manuscript lost its Conclusion section",
    )
    conclusion = manuscript.split(r"\section{Conclusion}", 1)[1].split(
        "bibliographystyle", 1
    )[0]
    conclusion_flat = " ".join(conclusion.split())
    for token in (
        "Binary reward records success but not which successful execution occurred",
        "preserve discoveries that fresh sampling can erase",
        "all 75 completed Level-1 comparisons",
        "small synthetic tasks",
        "Scale and Level-2 results remain incomplete",
        # The Conclusion must keep saying what verified breadth is *not*: this
        # replaced a standardized-DAPO sentence the manuscript no longer makes,
        # since DAPO is no longer discussed anywhere in the paper.
        "verified breadth is neither downstream utility nor",
        "More verified modes are not inherently better",
    ):
        require(token in conclusion_flat, f"Conclusion scope missing {token!r}")
    for token in (
        "capacity 16 can also truncate the protected support",
        "The numerical dose controls finite-time strength, not which distribution the categorical objective targets",
        "rather than a stronger neural-network claim",
    ):
        require(token in " ".join(manuscript.split()), f"Appendix claim boundary missing {token!r}")
    required_statement_headings = (
        r"\subsubsection*{AI Use Statement}",
        r"\subsubsection*{Ethics Statement}",
        r"\subsubsection*{Reproducibility Statement}",
    )
    for heading in required_statement_headings:
        require(
            manuscript.count(heading) == 1,
            f"required pre-reference heading {heading!r} must appear exactly once",
        )
    statement_positions = [manuscript.index(heading) for heading in required_statement_headings]
    require(
        statement_positions == sorted(statement_positions)
        and statement_positions[-1] < manuscript.index(r"\bibliographystyle{"),
        "AI Use, Ethics, and Reproducibility statements must appear in that order before references",
    )
    ai_statement = manuscript.split(required_statement_headings[0], 1)[1].split(
        required_statement_headings[1], 1
    )[0]
    ai_statement_flat = " ".join(ai_statement.split())
    for token in (
        "interpret recorded experimental results",
        "write auxiliary result-reporting code",
        "generate experimental observations",
        "reviewed all AI-assisted prose, code, and claims",
        "take responsibility for the final content",
    ):
        require(token in ai_statement_flat, f"AI Use Statement is missing {token!r}")
    require(
        r"\section{LLM Usage Disclosure}" not in manuscript,
        "obsolete appendix LLM disclosure remains",
    )
    related = manuscript.split(r"\section{Related Work}", 1)[1].split(
        r"\section{Correctness and Verified Support}", 1
    )[0]
    related_paragraphs = [
        paragraph
        for paragraph in related.split("\n\n")
        if paragraph.strip() and not paragraph.lstrip().startswith(r"\label{")
    ]
    require(
        len(related_paragraphs) == 3,
        "Related Work must contain exactly three thematic paragraphs",
    )
    require(
        all(p.lstrip().startswith(r"\noindent\textbf{") for p in related_paragraphs),
        "Every Related Work paragraph must begin with a bold thematic lead-in",
    )
    require(
        len(re.findall(r"\b[A-Za-z][A-Za-z0-9@-]*\b", related)) <= 290,
        "Related Work exceeds its 290-word source budget",
    )
    for token in (
        r"\textbf{Verifiable reasoning and support collapse.}",
        r"\textbf{Sampling-aware exploration.}",
        r"\textbf{Replay and behavioral archives.}",
        "yue2025rlvrlimit", "kirk2024understanding",
        "lochab2026ucpo", "zhang2025rlep",
        "tajwar2026maxrl", "MaxRL jointly", "ArgMaxRL",
        "exact binary special", "one exemplar per canonical",
        "retention an explicit",
    ):
        require(token in related, f"Related Work coverage missing {token!r}")

    # The section is called Experimental Design; the three experiments, the
    # shared protocol, and the comparator table all live under it.
    require(
        r"\section{Experimental Design}" in manuscript,
        "manuscript lost its Experimental Design section",
    )
    experiments = manuscript.split(r"\section{Experimental Design}", 1)[1].split(
        r"\section{Results}", 1
    )[0]
    experiments_flat = " ".join(experiments.split())
    for token in (
        r"\subsection{Experiment 1: retention and direct alternatives}",
        r"\label{tab:direct-comparators}",
        "the control traverses the same serialized bank path with exactly zero replay gradient",
        "Paired methods share model revision, initialization, prompt order",
        "group size 16",
        "384 training prompts",
        "eight passes (3,072 updates)",
        "Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B",
        "75 terminal comparisons",
        "Smaller intersections report exact",
        "We do not substitute shallower checkpoints, impute final results, or pool models or levels",
        "without canonical balancing",
    ):
        require(
            " ".join(token.split()) in experiments_flat,
            f"experiment design missing {token!r}",
        )
    # Exact model revisions and seed ranges are stated once, in the appendix run
    # contract the design section points at, rather than repeated in the body.
    for token in (
        "Qwen2.5-0.5B-Instruct", "Falcon3-1B-Instruct", "Qwen2.5-3B-Instruct",
        "43--47", "55--59", "70--74",
    ):
        require(
            " ".join(token.split()) in " ".join(manuscript.split()),
            f"run identity missing {token!r}",
        )

    results = manuscript.split(r"\section{Results}", 1)[1].split(
        r"\section{Conclusion}", 1
    )[0]
    results_flat = " ".join(results.split())
    # Results are reported experiment by experiment, in the order the
    # Experimental Design section registers them: retention and direct
    # alternatives, then memory beyond MaxRL, then the Level-2 transfer.
    ordered_result_sections = (
        r"\subsection{Experiment 1: canonical replay retains verified support across scale}",
        r"\subsection{Experiment 2: replay adds value beyond MaxRL}",
        r"\subsection{Experiment 3: Level 2 is harder; the first transfer block is complete}",
    )
    for heading in ordered_result_sections:
        require(heading in results, f"Results lost its section {heading!r}")
    result_positions = [
        results.index(heading)
        for heading in ordered_result_sections
        if heading in results
    ]
    require(
        result_positions == sorted(result_positions),
        "Results must follow retention and direct alternatives, memory beyond "
        "MaxRL, then the Level-2 transfer",
    )
    for token in (
        r"\label{fig:cross-scale-terminal-effects}",
        r"\label{fig:maxrl-factorial}",
        r"\label{fig:level2-admission}",
        "all 75 Level-1 model--domain--seed pairs",
        "Other alternatives produce domain-local effects",
        r"Re:MaxRL exceeds MaxRL in mean \texttt{pass@8} and \texttt{distinct@8} in every domain",
        r"The mean pass-rate effects range from \(+.084\) to \(+.509\)",
        "use PantryPlan as an enumerable stress test of support concentration under RL",
    ):
        require(
            token in results_flat,
            f"result narrative or promoted figure missing {token!r}",
        )
    main_body = manuscript.split(r"\appendix", 1)[0]
    appendix_labels = (
        "app:theory", "app:python", "app:prompts", "app:data", "app:algorithm",
        "app:reproducibility",
    )
    for label in appendix_labels:
        require(
            rf"\ref{{{label}}}" in main_body,
            f"appendix {label!r} is not tied to the main body",
        )
    # The reproduction contract is reached from the evidence policy and from
    # every claim it fails closed on, so it carries no separate link paragraph;
    # the other five appendices each open with exactly one.
    linked_appendix_labels = tuple(
        label for label in appendix_labels if label != "app:reproducibility"
    )
    require(
        manuscript.count(r"\noindent\emph{Main-body link.}")
        == len(linked_appendix_labels),
        "each linked appendix must begin with one explicit main-body link",
    )
    check_rehearsal_table(manuscript)
    check_aligned_domain_strips(manuscript)
    check_additional_evidence_figures(manuscript)
    check_core_terminal_table(manuscript)
    check_ucpo_interim_table(manuscript)
    check_scientific_method_scope(manuscript)
    check_dapo_progress(manuscript)
    check_maxent_factorial_tables(manuscript)
    check_cross_family_table(manuscript)
    algorithm_appendix = manuscript.split(r"\section{Algorithmic Details}", 1)[1].split(
        r"\section{Replay Mechanism Checks}", 1
    )[0]
    for token in (
        r"\usepackage{algorithm}", r"\usepackage{algorithmic}",
        r"\begin{algorithm}[H]", r"\begin{algorithmic}[1]",
        r"\label{alg:verified-replay-update}", r"\mathcal B_x", r"\tau",
        r"\textsc{NextBank}", r"w_{\mathrm{rep}}\gets(G-1)/G^2",
        "atomic state",
    ):
        require(token in manuscript, f"formal algorithm contract missing {token!r}")
    require(
        r"\begin{enumerate}" not in algorithm_appendix,
        "Algorithmic Details regressed to a prose numbered list",
    )
    # The manuscript reports one completed five-domain surface, so it names
    # components and frozen revisions rather than internal cohort codes. Guard
    # the evidence-typing language, and forbid the codes from returning.
    # These tokens span line breaks, so they are matched against a
    # whitespace-normalized copy: rewrapping a paragraph is not a regression,
    # and pinning the exact wrap made it look like one twice.
    collapsed = " ".join(manuscript.split())
    for token in (
        "Replay Mechanism Checks",
        "Uniform verified replay",
        "the same verifier, bank admission, round-robin scheduling",
        "exactly zero in controls",
        "no cross-model aggregate is interpreted as a scaling law",
        "altered multiple mechanisms and are excluded from the appendix",
        "No semantic-entropy term, proposal model, novelty reward, adaptive "
        "replay coefficient, reference KL, or gold support is active",
    ):
        require(
            " ".join(token.split()) in collapsed,
            f"ablation contract missing {token!r}",
        )
    for forbidden in (
        "sole terminal causal performance",
        "Only the separated-support actuator row is a completed causal",
    ):
        require(
            " ".join(forbidden.split()) not in collapsed,
            f"ablation contract regressed to single-contrast language {forbidden!r}",
        )
    for forbidden in (
        "outcome blindness",
        "outcome-blind",
        "exploratory",
        "audit",
        "campaign",
    ):
        require(
            re.search(
                rf"(?<![A-Za-z]){re.escape(forbidden)}(?![A-Za-z])",
                manuscript.lower(),
            )
            is None,
            f"reader-facing process label remains: {forbidden!r}",
        )
    for forbidden in ("Stage-A", "Stage A", "Stage-B", "Stage B", "E-Series"):
        require(
            forbidden not in manuscript,
            f"manuscript reintroduced staged-campaign language {forbidden!r}",
        )
    main_body_prose = manuscript.split(r"\appendix", 1)[0]
    for match in re.finditer(r"(?<![A-Za-z])E\d+(?:-R\d+)*(?![\d])", main_body_prose):
        # Graphic and generated-input filenames may retain internal provenance
        # keys, and a colour definition's hex triplet is not a cohort code;
        # nothing else in the main body may expose one.
        line = main_body_prose[: match.start()].rsplit("\n", 1)[-1]
        require(
            r"\includegraphics" in line
            or r"\input{" in line
            or r"\definecolor" in line,
            f"manuscript prose reintroduced cohort code {match.group(0)!r}",
        )
    for token in (
        "eval_multi_answer-10000-0",
        "eval-15100-93",
        "multi_answer-15900-61",
        "def load_and_validate(",
        'graph_answers == ("113", "131")',
        '"outputs": (3, 41, 13, 3)',
        '"answer": "6 + 3 + 9"',
        '{"answer": "F;E"',
        "eval-94002-high_fiber_snack-19",
        "navel_orange=75;sunflower_seeds=50",
        "grape_tomatoes=75;almonds=50",
        "def draw_mini_graph(",
        # One hue per operation, so the two Countdown keys are
        # distinguishable by which operations they execute.
        "OPERATOR_COLORS = {",
        '"mul": "#20068f"',
        '"div": "#7a02a8"',
        '"add": "#bc3587"',
        # MathIR names the actions its key ran, as PantryPlan does, and prints
        # them from the released menu so the panel shows a legal action neither
        # answer selects rather than only the ones they took.
        'mathir_elided = ("A", "B")',
        '"D": {"action": "sub(c)", "name": "add 1"}',
        'assert sorted(set(mathir_menu) - selected) == ["D"]',
        "NODE_PAINTS = {1: \"#e16462\", 2: \"#9e199d\", 3: \"#2f0596\"}",
        "INGREDIENT_COLORS = {",
    ):
        require(token in example_source, f"ModeBench example source missing {token!r}")
    for token in ("6.27", "4.52", "229.44", "5.00", "18.19", r"\renewcommand{\arraystretch}{1.08}"):
        require(token in manuscript, f"Table 1 contract missing {token!r}")
    # The domain specification table now sits beside the audits that verify it,
    # so the main body must still carry the two numbers a reader needs to size
    # the claim, and the table must be in the appendix rather than before the
    # examples figure.
    require(
        "trains on 384 prompts and evaluates on a fixed 128-prompt split"
        in " ".join(manuscript.split()),
        "main body lost the training/evaluation split sizes",
    )
    require(
        manuscript.index(r"\label{tab:tasks}")
        > manuscript.index(r"\label{fig:modebench-examples}"),
        "the domain specification table should follow the examples figure",
    )
    for token in (
        "DISPLAY_STEPS = [0, 96, 288, 384, 768, 864, 1152, 1248]",
        'title="Qwen2.5-3B\\nDr.GRPO"',
        'title="Qwen2.5-3B\\nRe:MaxRL (ours)"',
        "FONT = style.font_for_canvas(CANVAS_WIDTH)",
    ):
        require(token in source, f"Figure 1 source missing {token!r}")
    require(
        audit.get("schema") == "paper_graph_collapse_toy_v26",
        "wrong Figure 1 audit schema",
    )
    require(
        audit.get("layout_contract") == {
            "panels": ["A", "B", "C"],
            "panel_titles": [
                "Color the three blank nodes; connected nodes must differ.",
                "Qwen2.5-3B Dr.GRPO",
                "Qwen2.5-3B Re:MaxRL",
            ],
            "steps": [0, 96, 288, 384, 768, 864, 1152, 1248],
            "end_pass": 6.5,
            "paired_bars": False,
            "grouping": "warm task card -> blue trajectory card",
            "color_encodings": {
                "panel_a_circle_fill": "vertex_color",
                "panels_bc_bar_fill": "solution_mode",
            },
        },
        "wrong Figure 1 panel layout contract",
    )
    endpoint = "1248"
    require(
        audit["drgrpo_trajectory"][endpoint]["correct"] == 22
        and audit["drgrpo_trajectory"][endpoint]["distinct"] == 1,
        "Figure 1 Dr.GRPO endpoint changed",
    )
    require(
        audit["replay_maxrl_trajectory"][endpoint]["correct"] == 32
        and audit["replay_maxrl_trajectory"][endpoint]["distinct"] == 4,
        "Figure 1 Re:MaxRL endpoint changed",
    )
    require(
        audit["prompt"]["verifier_valid_completions"] == 6
        and audit["prompt"]["instance_id"] == "eval_multi_answer-10000-84"
        and audit["model"] == "Qwen2.5-3B-Instruct"
        and audit["seed"] == 70
        and audit["initial"]["correct"] == 16
        and audit["initial"]["distinct"] == 3
        and [
            audit["drgrpo_trajectory"][str(step)]["distinct"]
            for step in [0, 96, 288, 384, 768, 864, 1152, 1248]
        ] == [3, 3, 3, 2, 2, 1, 1, 1]
        and [
            audit["replay_maxrl_trajectory"][str(step)]["distinct"]
            for step in [0, 96, 288, 384, 768, 864, 1152, 1248]
        ] == [3, 3, 3, 3, 3, 3, 3, 4]
        and audit["terminal_pass8"]
        == {"drgrpo": 1.0, "replay_maxrl": 1.0}
        and audit["selection"]["status"] == "post_hoc_illustration",
        "Figure 1 illustration contract changed",
    )

    for token in (
        '"drgrpo"', '"replay_drgrpo"', '"maxrl"', '"replay_maxrl"',
        '"pass8"', '"distinct8"', "95% t interval across seeds",
        "Arrows add verified replay",
    ):
        require(token in maxrl_source, f"MaxRL results source missing {token!r}")
    require(
        maxrl_audit.get("schema") == "page1-e118-qwen05b-factorial-figure-v1",
        "wrong MaxRL results audit schema",
    )
    require(
        maxrl_audit.get("metrics") == ["pass8", "distinct8"]
        and maxrl_audit.get("methods")
        == ["drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl"],
        "MaxRL factorial contract drifted",
    )
    result_path = Path(maxrl_audit["source"])
    require(result_path.is_file(), "MaxRL result record is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(
        result.get("schema") == "e118-qwen05b-terminal-factorial-v1"
        and result.get("model") == "Qwen2.5-0.5B-Instruct"
        and result.get("target_step") == 3072
        and result.get("seeds") == [43, 44, 45, 46, 47],
        "MaxRL result record drifted",
    )
    for domain, record in result["domains"].items():
        for metric in ("pass8", "distinct8"):
            require(
                record["summaries"]["replay_maxrl"][metric]["mean"]
                > record["summaries"]["maxrl"][metric]["mean"],
                f"MaxRL result claim fails for {domain}/{metric}",
            )
    page_one_source = " ".join(manuscript.split())
    for token in (
        r"Explore and Preserve Solution Modes in RLVR:",
        r"ModeBench and Re:MaxRL",
        r"figures/modecollapse_story.pdf",
        r"Correctness can improve while verified solution support",
        r"Both start at 16/32 correct across three modes",
        r"a dashed ring marks a vertex the prompt leaves blank",
        r"By pass 4.5, Dr.GRPO",
        r"Re:MaxRL returns 32/32 across four",
        r"Numbers above bars are pooled unique verified keys",
    ):
        require(
            " ".join(token.split()) in page_one_source,
            f"missing page-1 token {token!r}",
        )
    for token in (
        r"\subsection{Experiment 2: replay adds value beyond MaxRL}",
        r"figures/e118_all_scale_factorial_progress.pdf",
        r"\label{fig:maxrl-factorial}",
        r"Replay remains useful when fresh rollouts optimize",
        r"Re:MaxRL exceeds MaxRL",
        r"all five domains",
    ):
        require(token in manuscript, f"missing MaxRL results token {token!r}")
    require(
        manuscript.index(r"figures/modecollapse_story.pdf")
        < manuscript.index(r"\section{Correctness and Verified Support}"),
        "explanatory teaser is not in the Introduction",
    )
    require(
        manuscript.index(r"\section{Related Work}")
        < manuscript.index(r"\section{Correctness and Verified Support}")
        < manuscript.index(r"\section{Results}")
        < manuscript.index(r"figures/e118_all_scale_factorial_progress.pdf"),
        "Related Work is not Section 2 or the MaxRL factorial figure is not in Results",
    )

    expected_metrics = {
        "greedy": "neutral pass@1",
        "pass8": "neutral pass@8",
        "distinct8": "mean # distinct correct@8",
        "online_canonical_tracked_outcomes": (
            "cumulative verified discoveries"
        ),
    }
    headline_metrics = literal_assignment(
        HEADLINE_SOURCE,
        "E68_TRAIN_METRICS",
    )
    require(
        list(headline_metrics) == list(expected_metrics),
        f"wrong headline metric keys: {list(headline_metrics)}",
    )
    require(
        [row["title"] for row in headline_metrics.values()]
        == list(expected_metrics.values()),
        "wrong headline metric titles",
    )

    headline_source = HEADLINE_SOURCE.read_text(encoding="utf-8")
    appendix_source = APPENDIX_SOURCE.read_text(encoding="utf-8")
    e70_terminal = json.loads(E70_TERMINAL_AUDIT.read_text(encoding="utf-8"))
    require(
        e70_terminal.get("status") == "pass"
        and e70_terminal.get("summary", {}).get("terminal_runs") == 40,
        "E70 Stage A is not a passing 40-run terminal cohort",
    )
    pantry_terminal = json.loads(
        PANTRY_TERMINAL_AUDIT.read_text(encoding="utf-8")
    )
    require(
        pantry_terminal.get("status") == "pass"
        and pantry_terminal.get("decision") == "pantry_terminal_eligible",
        "PantryPlan Stage B is not terminal-eligible",
    )
    # Line weights, seed styling, and alphas moved into ops/paper_style.py so
    # every figure shares one specification. The guard follows them there: both
    # figure sources must draw from that module, and the module must still
    # define the constants, rather than each figure pinning its own literals.
    style_source = (ROOT / "ops/paper_style.py").read_text()
    for token in (
        "MEAN_LW = 1.35",
        "SEED_LW = 0.5",
        "BAND_ALPHA = 0.13",
        "CONTROL = ",
        "METHOD = ",
        "def style_axis(",
        'PAPER_GRID_AXIS = "both"',
        "axis=PAPER_GRID_AXIS",
    ):
        require(token in style_source, f"shared paper style lost {token!r}")
    for token in (
        "e68_plot._series",
        "def _complete_mean(",
        "e68_plot._set_y_limits",
        "import paper_style as style",
        "style.style_axis(",
        "PAPER_SEEDS = (43, 44, 45, 46, 47)",
        "ppe71_scale384_05b_12pass_scaling_curve.json",
    ):
        require(token in headline_source, f"headline lost E68 style token {token!r}")
    for token in (
        "import paper_style as style",
        "axis.set_ylim(0.0, high * 1.08 if high > 0 else 1.0)",
        # Canvas geometry is now derived from the shared column width rather
        # than a literal, so pin the derivation and the grid, not the inches.
        "figsize=(style.WIDTH,",
        "figure.add_gridspec(",
        "outer_cell.subgridspec(1, len(PANELS)",
        "axis.set_box_aspect(0.62)",
    ):
        require(token in appendix_source, f"appendix lost E68 style token {token!r}")

    expected_panels = (
        ("greedy", "neutral pass@1", False),
        ("pass8", "neutral pass@8", False),
        ("distinct8", "mean # distinct correct@8", True),
        (
            "online_canonical_tracked_outcomes",
            "cumulative verified discoveries",
            True,
        ),
    )
    appendix_panels = literal_assignment(APPENDIX_SOURCE, "PANELS")
    require(appendix_panels == expected_panels, "wrong appendix panels")
    if APPENDIX_PROVENANCE.is_file():
        provenance = json.loads(APPENDIX_PROVENANCE.read_text(encoding="utf-8"))
        require(
            provenance["panels"] == [row[0] for row in expected_panels],
            "wrong appendix provenance panels",
        )

        require(
            provenance["layout"]
            == {
                "domain_card_grid": [4, 2],
                "panels_per_card": [2, 2],
                "figure_inches": [9.2, 12.0],
                "panel_box_aspect": 0.80,
            },
            "wrong appendix card layout",
        )
    # The full-page cohort surface was removed from the manuscript: it was
    # visually redundant with the training-dynamics grid and read as an
    # internal monitoring panel --- timestamped, captioned in "available-seed
    # mean" and "latest mean" language --- rather than as a result. The
    # generator and its provenance are retained and still checked above, so the
    # surface can be regenerated for audit, but it must not return to the paper
    # in live-dashboard form.
    for forbidden in (
        r"\label{fig:clean-cohort}",
        r"height=.84\textheight",
        "available-seed mean",
        "latest mean",
    ):
        require(
            forbidden not in manuscript,
            f"manuscript reintroduced the removed cohort dashboard {forbidden!r}",
        )
    forbidden_metrics = (
        "neutral mean@8",
        "valid coverage@8",
        "open-set predictive entropy EMA",
        "verified replay KL",
        "new verified outcome fraction",
        "mean verified support per prompt",
        "excess@8",
    )
    # Only the headline surface is rendered into the manuscript now; the
    # appendix cohort dashboard was removed, so its PDF is no longer a paper
    # artifact to hold to the manuscript's metric vocabulary.
    for path, expected_repetitions in ((HEADLINE_PDF, 5),):
        text = pdf_text(path)
        for title in expected_metrics.values():
            require(
                text.count(title) == expected_repetitions,
                f"{path.name}: expected {expected_repetitions} rendered "
                f"instances of {title!r}, got {text.count(title)}",
            )
        for forbidden in forbidden_metrics:
            require(
                forbidden not in text,
                f"{path.name}: forbidden rendered metric {forbidden!r}",
            )
    example_text = pdf_text(EXAMPLES_PDF)
    for token in (
        "Graph coloring", "Countdown", "Python factors", "MathIR", "PantryPlan",
        "(6 × 9) / 3", "6 + 3 + 9", "canonical keys",
        "navel_orange=75", "sunflower_seeds=50", "grape_tomatoes=75", "almonds=50",
        "C;F", "F;E",
    ):
        require(token in example_text, f"ModeBench example missing {token!r}")
    # Both keys of every domain must render and differ. Graph and PantryPlan
    # keys are drawn rather than set in type -- paint chips and pictograms --
    # so those two are held by the renderer's own validated structures.
    for first, second in (
        ("div(mul(6,9),3)", "add(9,add(3,6))"),
        ("[2, 2, 7, 3]", "[3, 41, 13, 3]"),
        ("x/2 = 8 → x = 16", "x − 18 = −2 → x = 16"),
    ):
        require(first != second, "paired keys must differ")
        require(first in example_text, f"Figure 2 missing key {first!r}")
        require(second in example_text, f"Figure 2 missing key {second!r}")
    for token in (
        'graph_modes = ((2, 1, 1, 1, 3, 2), (2, 1, 3, 1, 1, 2))',
        '"key_kind": "paints"',
        '"key_kind": "icons"',
        "def draw_paint_row(",
        "def draw_icon_row(",
    ):
        require(token in example_source, f"Figure 2 source missing {token!r}")
    require(
        example_source.count('"span": 1') == 5
        and example_source.count('"span": 2') == 0,
        "Figure 2 must stay five equal-width environment cards on a three-column grid",
    )
    require(
        manuscript.count(r"figures/modebench_examples.pdf") == 1
        and "modebench_pantry_example" not in manuscript,
        "Figure 2 must be one figure, not a split pair",
    )
    mechanism_source = MECHANISM_SOURCE.read_text(encoding="utf-8")
    # The mechanism figure sets each column title on two lines, so the rendered
    # text is compared with whitespace collapsed rather than line by line.
    mechanism_text = re.sub(r"\s+", " ", pdf_text(MECHANISM_PDF))
    for source_token, rendered in (
        ('("Verify", "outputs")', "Verify outputs"),
        ('("Store", "exemplars")', "Store exemplars"),
        ('("Revisit", "recurrently")', "Revisit recurrently"),
        ('("Replay", "uniformly")', "Replay uniformly"),
        ("every banked mode", "every banked mode"),
    ):
        require(source_token in mechanism_source, f"mechanism source missing {source_token!r}")
        require(rendered in mechanism_text, f"mechanism PDF missing {rendered!r}")
    require(
        mechanism_source.count("COLUMN_SPECS") >= 1
        and len(re.findall(r'\(\("[A-D]", "[A-Z 0-9]+"\)', mechanism_source)) == 4,
        "mechanism figure must stay one row of four columns",
    )
    if MAIN_PDF.is_file() and MAIN_PDF.stat().st_mtime >= MANUSCRIPT.stat().st_mtime:
        page_one = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", str(MAIN_PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        require("Figure 1:" in page_one, "Figure 1 is not on page 1")
        # ICLR counts main text through the Conclusion against nine pages.  A
        # source-labeled interim snapshot may temporarily use page 10; deleting
        # either the snapshot or its over-length disclosure restores the strict
        # final limit automatically.
        page_text = {
            page: subprocess.run(
                [
                    "pdftotext", "-f", str(page), "-l", str(page),
                    str(MAIN_PDF), "-",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for page in range(1, 13)
        }

        def heading_page(pattern: str) -> int | None:
            return next(
                (
                    page
                    for page, text in page_text.items()
                    if re.search(pattern, text) is not None
                ),
                None,
            )

        conclusion_page = heading_page(r"(?mi)^\s*(?:\d+\s+)?C\s*ONCLUSION\s*$")
        ai_use_page = heading_page(r"(?mi)^\s*AI\s+U\s*SE\s+S\s*TATEMENT\s*$")
        ethics_page = heading_page(r"(?mi)^\s*E\s*THICS\s+S\s*TATEMENT\s*$")
        reproducibility_page = heading_page(
            r"(?mi)^\s*R\s*EPRODUCIBILITY\s+S\s*TATEMENT\s*$"
        )
        references_page = heading_page(r"(?mi)^\s*R\s*EFERENCES\s*$")
        require(
            conclusion_page is not None and conclusion_page <= main_page_limit,
            f"main content exceeds the {page_contract}",
        )
        require(
            ai_use_page is not None
            and ai_use_page >= conclusion_page
            and ethics_page is not None
            and ethics_page >= ai_use_page,
            "the required AI Use Statement must follow Conclusion and precede Ethics",
        )
        if ai_use_page == main_page_limit + 1:
            first_exempt_line = next(
                (
                    line.strip()
                    for line in page_text[main_page_limit + 1].splitlines()
                    if line.strip()
                    and not line.strip().isdigit()
                    and not line.strip().startswith("Under review as")
                ),
                "",
            )
            require(
                re.fullmatch(r"AI\s+U\s*SE\s+S\s*TATEMENT", first_exempt_line, re.I)
                is not None,
                f"main text spills past page {main_page_limit}: the exempt page "
                f"opens with {first_exempt_line!r}, not the AI Use Statement",
            )
        require(
            reproducibility_page is not None
            and ethics_page is not None
            and reproducibility_page >= ethics_page,
            "Reproducibility Statement must follow the Ethics Statement",
        )
        require(
            references_page is not None
            and reproducibility_page is not None
            and references_page >= reproducibility_page,
            "References must follow the policy statements",
        )
    print(
        "Paper figure contracts: PASS "
        f"({len(abstract_words)}-word abstract, max 211; Figure 1 page 1; "
        f"main body satisfies {page_contract}; 2027 AI/Ethics/Reproducibility "
        "statements precede References; "
        "paired examples A-E; four-stage verified replay A-D; mechanism metric grids)"
    )
if __name__ == "__main__":
    main()
