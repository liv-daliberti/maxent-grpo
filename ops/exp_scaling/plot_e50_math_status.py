#!/usr/bin/env python3
"""Render the current hard-MATH canonical-MaxEnt evidence and live gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
FIGURES = ROOT / "paper/figures"
ADVANCEMENT = ARTIFACTS / "e49t_natural_menu_math_toy_advancement_v1.json"
ROUTE_CAL = ARTIFACTS / "e49t_route_confusion_calibration_v1/result.json"
DECL_CAL = ARTIFACTS / "e49t_declaration_mismatch_calibration_v1/result.json"
CERTIFICATION = (
    ARTIFACTS
    / "e49s_deterministic_mathir_repairs_v1/advancement_decision.json"
)
SAFE_SIGNATURE = (
    ARTIFACTS / "safe_math_strategy_signature_development_audit_v1.json"
)
FULL_INJECTION = (
    ARTIFACTS / "e50g_full_injection_rewrite_audit_v1.json"
)
LIVE_ROUTE_BANK = ARTIFACTS / "e50_live_route_bank_summary.json"
LIVE_ROUTE_SUMMARIZER = (
    ROOT
    / "ops/math_strategy_calibration/summarize_e50_live_route_bank.py"
)
EVAL_PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"

CONTROL = "#5B6472"
TREATMENT = "#0072B2"
GOOD = "#009E73"
WARN = "#E69F00"
BAD = "#D55E00"
PALE = "#D9DEE7"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _refresh_live_route_bank() -> None:
    """Refresh the read-only corpus diagnostic before rendering."""
    if not EVAL_PYTHON.is_file() or not LIVE_ROUTE_SUMMARIZER.is_file():
        return
    completed = subprocess.run(
        [
            str(EVAL_PYTHON),
            str(LIVE_ROUTE_SUMMARIZER),
            "--out",
            str(LIVE_ROUTE_BANK),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        print(
            "warning: live route-bank refresh failed: "
            f"{completed.stderr.strip()}",
            file=sys.stderr,
        )


def _arm(advancement: dict[str, Any], treatment: bool) -> dict[str, Any]:
    key = "online_canonical_haarnoja" if treatment else "grpo"
    return advancement["arms"][key]


def _percent_label(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=FIGURES / "e50_hard_math_status_live.png",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ARTIFACTS / "e50_hard_math_status_live.json",
    )
    args = parser.parse_args()

    _refresh_live_route_bank()
    certification = _read_json(CERTIFICATION)
    route_cal = _read_json(ROUTE_CAL)
    declaration = _read_json(DECL_CAL)
    advancement = _read_json(ADVANCEMENT)
    control = _arm(advancement, treatment=False)
    treatment = _arm(advancement, treatment=True)

    signature = _read_json(SAFE_SIGNATURE)
    injection = _read_json(FULL_INJECTION)
    live_route_bank = _optional_json(LIVE_ROUTE_BANK)
    route_stages = live_route_bank.get("stages") or []
    provisional_eligible = int(
        live_route_bank.get("provisional_eligible_problem_count", 0)
    )
    selected_count = int(
        live_route_bank.get("e50g", {}).get("selected_problem_count", 0)
    )
    e50g_status = live_route_bank.get("e50g", {})
    e50g_failed = (
        e50g_status.get("result_present") is True
        and e50g_status.get("pass") is False
    )

    route_counts = route_cal["counts"]
    declaration_counts = declaration["counts"]
    calibration = [
        (
            "Known routes\naccepted",
            route_counts["correct_positive"] / route_counts["positive_count"],
        ),
        (
            "Duplicates\nrejected",
            1
            - route_counts["duplicate_false_new_count"]
            / route_counts["negative_count"],
        ),
        (
            "Rewrites not\ncalled new",
            1
            - injection["counts"]["false_new_count"]
            / injection["counts"]["comparison_count"],
        ),
        (
            "Matched route\naccepted",
            declaration_counts["correct_matched"]
            / declaration_counts["matched_count"],
        ),
        (
            "Wrong route\nrejected",
            1
            - declaration_counts["mismatch_accepted"]
            / declaration_counts["mismatched_count"],
        ),
    ]

    terminal_routes = advancement["terminal_route_coverage"]
    support = {
        "control": {
            "route_coverage": terminal_routes["grpo"][
                "mean_normalized_strategy_coverage_all_prompts"
            ],
            "support_two": control["bank"][
                "max_support_at_least_two_prompt_fraction"
            ],
            "full_dual": terminal_routes["grpo"]["dual_full_coverage_rate"],
        },
        "treatment": {
            "route_coverage": terminal_routes["online_canonical_haarnoja"][
                "mean_normalized_strategy_coverage_all_prompts"
            ],
            "support_two": treatment["bank"][
                "max_support_at_least_two_prompt_fraction"
            ],
            "full_dual": terminal_routes["online_canonical_haarnoja"][
                "dual_full_coverage_rate"
            ],
        },
    }

    summary = {
        "schema": "e50_hard_math_status_live_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "toy_certifications": {
            "completed": certification["materialization_manifest"]["menu_count"],
            "total": 100,
            "passed": certification["pass"],
        },
        "calibration": {
            "route_confusion_pass": route_cal["pass"],
            "declaration_mismatch_pass": declaration["pass"],
            "known_route_acceptance": calibration[0][1],
            "duplicate_false_new_count": route_counts[
                "duplicate_false_new_count"
            ],
            "answer_only_false_accept_count": route_counts[
                "accepted_negative"
            ],
            "mismatched_declaration_accept_count": declaration_counts[
                "mismatch_accepted"
            ],
            "signature_false_new_count": signature["counts"][
                "false_new_count"
            ],
            "signature_distinct_recall": signature["counts"][
                "distinct_recall"
            ],
            "rewrite_false_new_count": injection["counts"][
                "false_new_count"
            ],
            "rewrite_comparison_count": injection["counts"][
                "comparison_count"
            ],
        },
        "matched_toy": {
            "control_terminal": control["terminal_eval"],
            "treatment_terminal": treatment["terminal_eval"],
            "support": support,
            "treatment_controller": treatment["controller"],
            "advance_to_exact_oat_full": advancement[
                "advance_to_exact_oat_full"
            ],
        },
        "prospective_route_bank": live_route_bank,
        "decision": (
            "Canonicalization and execution binding pass; first toy preserves "
            "task quality but fails route coverage. The prospectively generated "
            "replacement bank is not training-authorized until E50G passes."
        ),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        2, 3, figsize=(15.5, 8.8), constrained_layout=True
    )

    ax = axes[0, 0]
    completed = certification["materialization_manifest"]["menu_count"]
    ax.barh(["Certified menus"], [100], color=PALE, height=0.45)
    ax.barh(["Certified menus"], [completed], color=GOOD, height=0.45)
    ax.text(completed / 2, 0, f"{completed}/100", ha="center", va="center",
            color="white", fontweight="bold", fontsize=14)
    ax.set(xlim=(0, 100), title="A. Deterministic toy certification",
           xlabel="Menus with executable certificates")

    ax = axes[0, 1]
    labels = [row[0] for row in calibration]
    values = [row[1] for row in calibration]
    positions = np.arange(len(values))
    bars = ax.barh(positions, values, color=GOOD)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set(title="B. Canonicalizer integrity", xlabel="Correct gate decision")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            _percent_label(value),
            ha="left",
            va="center",
            fontsize=8,
        )

    ax = axes[0, 2]
    metrics = [
        ("Greedy", "eval/math/accuracy"),
        ("Pass@8", "eval/math/sampled_any_correct_at_8"),
        ("Mean@8", "eval/math/sampled_mean_at_8"),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    control_values = [control["terminal_eval"][key] for _, key in metrics]
    treatment_values = [treatment["terminal_eval"][key] for _, key in metrics]
    ax.bar(x - width / 2, control_values, width, color=CONTROL,
           label="Matched Dr.GRPO")
    ax.bar(x + width / 2, treatment_values, width, color=TREATMENT,
           label="Normalized Haarnoja")
    ax.set_xticks(x, [label for label, _ in metrics])
    ax.set_ylim(0, 0.6)
    ax.set(title="C. First matched toy after 3 epochs", ylabel="Held-out score")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["Mean route\ncoverage", "Prompts with\nsupport ≥2",
              "Full dual-route\ncoverage"]
    x = np.arange(len(labels))
    ax.bar(
        x - width / 2,
        [support["control"][key] for key in (
            "route_coverage", "support_two", "full_dual"
        )],
        width,
        color=CONTROL,
        label="Matched Dr.GRPO",
    )
    ax.bar(
        x + width / 2,
        [support["treatment"][key] for key in (
            "route_coverage", "support_two", "full_dual"
        )],
        width,
        color=TREATMENT,
        label="Normalized Haarnoja",
    )
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 0.5)
    ax.set(title="D. Mechanism gate: not yet passed", ylabel="Prompt fraction")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if route_stages:
        y = np.arange(len(route_stages))
        generated_counts = [int(row["generated"]) for row in route_stages]
        totals = [
            int(row["target"]) if row.get("target") is not None else 50
            for row in route_stages
        ]
        remaining = [
            max(0, total - generated)
            for total, generated in zip(
                totals, generated_counts, strict=True
            )
        ]
        ax.barh(y, generated_counts, color=TREATMENT, label="Generated")
        ax.barh(
            y,
            remaining,
            left=generated_counts,
            color=PALE,
            label="Remaining / not started",
        )
        ax.set_yticks(
            y,
            [
                f"{row['stage']}  {row['label']}"
                for row in route_stages
            ],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 62)
        ax.set(
            title="E. Prospective hard-MATH route bank (live)",
            xlabel=(
                "Problems generated; annotations are provisional exact + "
                "safe-signature eligibility"
            ),
        )
        for index, row in enumerate(route_stages):
            target = (
                str(row["target"])
                if row.get("target") is not None
                else "pending"
            )
            ax.text(
                61,
                index,
                (
                    f"{row['generated']}/{target}  |  "
                    f"+{row['new_eligible_count']} → "
                    f"{row['cumulative_eligible_count']}"
                ),
                va="center",
                ha="right",
                fontsize=8,
                color=GOOD if row["new_eligible_count"] else CONTROL,
                fontweight="bold",
            )
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Waiting for live route-bank summary",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("F. Current conclusion", loc="left")
    ax.text(
        0,
        0.92,
        "What works",
        color=GOOD,
        fontweight="bold",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0,
        0.82,
        "• Exact answer validation + finite-menu route binding\n"
        "• False-new and wrong-declaration vetoes\n"
        "• Matched 3-epoch training without task collapse",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.text(
        0,
        0.49,
        "What does not yet work",
        color=BAD,
        fontweight="bold",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0,
        0.39,
        (
            "• E50G: only "
            f"{e50g_status.get('sound_bound_signature_menu_count', 0)}/"
            f"{e50g_status.get('signature_candidate_count', 0)} "
            "menus survived soundness + signature\n"
            "• E50G: "
            f"{e50g_status.get('bidirectionally_executable_count', 0)} "
            "were bidirectionally executable\n"
            "• E50D materialization and training remain blocked"
            if e50g_failed
            else
            "• Sustained support growth like graph/countdown\n"
            "• Route-coverage advantage over matched Dr.GRPO\n"
            "• Full MATH-500 result"
        ),
        va="top",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.text(
        0,
        0.08,
        (
            f"Live corpus: {provisional_eligible} provisional dual-route "
            f"problems; E50G selected {selected_count}/10.\n"
            + (
                "E50G failed closed; E50D was not materialized."
                if e50g_failed
                else
                "Next gate: E50G exact audits + natural 0.5B "
                "reproducibility."
            )
        ),
        va="bottom",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )

    generated = datetime.now().astimezone()
    figure.suptitle(
        "Hard-MATH online canonical MaxEnt — evidence and live gate\n"
        f"Updated {generated:%Y-%m-%d %H:%M:%S %Z}",
        fontsize=16,
        fontweight="bold",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    figure.savefig(args.out.with_suffix(".pdf"))
    plt.close(figure)
    print(args.out)
    print(args.summary_out)


if __name__ == "__main__":
    main()
