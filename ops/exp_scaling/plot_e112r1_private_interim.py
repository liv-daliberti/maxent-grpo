#!/usr/bin/env python3
"""Render the frozen, private E112-R1 exploratory endpoint subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(ROOT / "ops" / "exp_scaling"))
import paper_style as style  # noqa: E402
from plot_paper_direct_comparator_endpoint_effects import (  # noqa: E402
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    _completion_marker,
    _effect,
    _terminal_metrics,
)


FREEZE = ROOT / "var/artifacts/e112r1_private_interim_unblinding_freeze.json"
DEFAULT_OUTPUT = ROOT / "var/artifacts/private_interim/e112r1_subset_endpoint_effects"
TREATMENT_LEDGER = (
    ROOT / "var/artifacts/e112r1_verified_support_discovery_full_three_scale_jobs.json"
)
COMPARATOR_LEDGERS = (
    ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
    ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
    ROOT / "var/artifacts/e109_repaired_python_replay_comparators_jobs.json",
)
SCALES = ("qwen05b", "falcon1b")
SCALE_LABELS = {"qwen05b": "Qwen 0.5B", "falcon1b": "Falcon 1B"}
PRIVATE_LABEL = "PRIVATE EXPLORATORY INTERIM — NOT FOR PAPER OR SELECTION"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _assert_private_output(output: Path) -> None:
    private_root = (ROOT / "var/artifacts/private_interim").resolve()
    try:
        output.resolve().relative_to(private_root)
    except ValueError as exc:
        raise RuntimeError(
            f"private E112 output must stay under {private_root}, got {output}"
        ) from exc


def _run_map(ledger: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for run in ledger.get("runs", []):
        job_id = int(run["job_id"])
        if job_id in result:
            raise RuntimeError(f"duplicate job ID {job_id}")
        result[job_id] = run
    return result


def build(freeze_path: Path = FREEZE) -> dict[str, Any]:
    freeze_path = freeze_path.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "e112r1_private_interim_unblinding_freeze_v1":
        raise RuntimeError("unexpected E112-R1 freeze schema")
    if freeze.get("campaign_mutation_allowed_from_interim") is not False:
        raise RuntimeError("freeze does not prohibit campaign mutation")
    if freeze.get("paper_efficacy_output_allowed") is not False:
        raise RuntimeError("freeze does not prohibit paper efficacy output")
    if freeze.get("pointmaze") != "excluded":
        raise RuntimeError("PointMaze exclusion is missing")
    if _sha256(TREATMENT_LEDGER) != freeze["ledger_sha256"]:
        raise RuntimeError("released E112-R1 ledger differs from the frozen ledger")
    protocol = ROOT / str(freeze["protocol"])
    if _sha256(protocol) != freeze["protocol_sha256"]:
        raise RuntimeError("private interim protocol differs from its frozen hash")

    treatment_ledger = json.loads(TREATMENT_LEDGER.read_text(encoding="utf-8"))
    treatment_runs = _run_map(treatment_ledger)
    target = int(freeze["target_steps"])
    if int(treatment_ledger["target_steps"]) != target:
        raise RuntimeError("treatment target differs from freeze")

    comparator_runs: dict[int, tuple[dict[str, Any], Path, int]] = {}
    source_paths: set[Path] = {freeze_path, protocol, TREATMENT_LEDGER}
    for ledger_path in COMPARATOR_LEDGERS:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger_target = int(ledger["target_steps"])
        if ledger_target != target:
            raise RuntimeError(f"{ledger_path}: target differs from freeze")
        for job_id, run in _run_map(ledger).items():
            if job_id in comparator_runs:
                raise RuntimeError(f"duplicate comparator job ID {job_id}")
            comparator_runs[job_id] = (run, ledger_path, ledger_target)
        source_paths.add(ledger_path)

    observed_membership: set[tuple[str, str, int, int]] = set()
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for frozen in freeze["cells"]:
        scale = str(frozen["scale"])
        domain = str(frozen["domain"])
        seed = int(frozen["seed"])
        job_id = int(frozen["job_id"])
        membership = (scale, domain, seed, job_id)
        if membership in observed_membership:
            raise RuntimeError(f"duplicate frozen cell {membership}")
        observed_membership.add(membership)
        if scale not in SCALES or domain not in DOMAIN_ORDER:
            raise RuntimeError(f"unexpected frozen cell {membership}")

        treatment_run = treatment_runs.get(job_id)
        if treatment_run is None:
            raise RuntimeError(f"frozen treatment job {job_id} is absent from ledger")
        expected = (str(treatment_run["scale"]), str(treatment_run["domain"]), int(treatment_run["seed"]))
        if expected != (scale, domain, seed):
            raise RuntimeError(f"treatment metadata drift for job {job_id}: {expected}")
        treatment_dir = ROOT / str(frozen["run_dir"])
        if treatment_dir.resolve() != Path(str(treatment_run["run_dir"])).resolve():
            raise RuntimeError(f"treatment directory drift for job {job_id}")
        treatment_marker = ROOT / str(frozen["completion_marker"])
        if _sha256(treatment_marker) != frozen["completion_marker_sha256"]:
            raise RuntimeError(f"completion marker drift for job {job_id}")
        if _completion_marker(treatment_dir, target=target) != treatment_marker:
            raise RuntimeError(f"invalid treatment completion marker for job {job_id}")

        paired = frozen["paired_replay"]
        comparator_job_id = int(paired["job_id"])
        comparator_record = comparator_runs.get(comparator_job_id)
        if comparator_record is None:
            raise RuntimeError(f"paired comparator job {comparator_job_id} not found")
        comparator_run, comparator_ledger_path, comparator_target = comparator_record
        comparator_dir = Path(str(paired["run_dir"]))
        if comparator_dir.resolve() != Path(str(comparator_run["run_dir"])).resolve():
            raise RuntimeError(f"paired comparator directory drift for job {job_id}")
        comparator_marker = _completion_marker(comparator_dir, target=comparator_target)
        if comparator_marker is None:
            raise RuntimeError(f"paired comparator is not terminal for job {job_id}")

        treatment_endpoint, treatment_sources = _terminal_metrics(
            treatment_dir, target=target
        )
        comparator_endpoint, comparator_sources = _terminal_metrics(
            comparator_dir, target=comparator_target
        )
        source_paths.update(
            treatment_sources
            | comparator_sources
            | {treatment_marker, comparator_marker, comparator_ledger_path}
        )
        by_cell.setdefault((scale, domain), []).append(
            {
                "seed": seed,
                "treatment_job_id": job_id,
                "comparator_job_id": comparator_job_id,
                "treatment": treatment_endpoint,
                "comparator": comparator_endpoint,
                "effect": _effect(treatment_endpoint, comparator_endpoint),
            }
        )

    if len(observed_membership) != int(freeze["terminal_cells"]):
        raise RuntimeError("frozen terminal-cell count does not match membership")

    cells: list[dict[str, Any]] = []
    for scale in SCALES:
        for domain in DOMAIN_ORDER:
            seeds = sorted(by_cell.get((scale, domain), []), key=lambda row: row["seed"])
            cells.append(
                {
                    "scale": scale,
                    "model": SCALE_LABELS[scale],
                    "domain": domain,
                    "n": len(seeds),
                    "evidence": "exact_frozen_terminal_prefix" if seeds else "no_frozen_terminal_cell",
                    "per_seed": seeds,
                    "summaries": None,
                }
            )

    return {
        "schema": "e112r1-private-exploratory-endpoint-effects-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": PRIVATE_LABEL,
        "confirmatory": False,
        "campaign_mutation_allowed": False,
        "paper_efficacy_output_allowed": False,
        "pointmaze": "excluded",
        "freeze": _relative(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "frozen_at_utc": freeze["frozen_at_utc"],
        "frozen_terminal_cells": int(freeze["terminal_cells"]),
        "selection_rule": freeze["selection_rule"],
        "analysis_rule": "exact paired seed effects only; no mean, interval, test, or pooling",
        "treatment": "E112-R1 corrected verified-support MaxEnt",
        "comparator": "preregistered matched Re:Dr.GRPO",
        "metrics": {
            "pass8": "E112-R1 pass@8 minus matched Re:Dr.GRPO pass@8",
            "adjusted_breadth8": "E112-R1 (distinct@8-pass@8) minus matched Re:Dr.GRPO (distinct@8-pass@8)",
        },
        "scale_order": list(SCALES),
        "domain_order": list(DOMAIN_ORDER),
        "cells": cells,
        "input_sha256": {
            _relative(path): {
                "byte_length": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(source_paths)
        },
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(SCALES),
        len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 4.25),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    metrics = (("pass8", r"$\Delta P$"), ("adjusted_breadth8", r"$\Delta(D-P)$"))
    values = [0.0]
    for cell in payload["cells"]:
        for row in cell["per_seed"]:
            values.extend(row["effect"].values())
    low = min(-0.1, min(values) - 0.08)
    high = max(0.1, max(values) + 0.08)
    cells = {(row["scale"], row["domain"]): row for row in payload["cells"]}
    treatment_color = "#8B2E2E"

    for scale_index, scale in enumerate(SCALES):
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            axis = axes[scale_index][domain_index]
            style.style_axis(
                axis,
                grid="both",
                title=DOMAIN_LABELS[domain] if scale_index == 0 else None,
            )
            axis.axhline(0.0, color=style.MUTED, lw=0.8, linestyle=(0, (2, 2)))
            cell = cells[(scale, domain)]
            if cell["n"]:
                jitter = [
                    (index - (cell["n"] - 1) / 2) * 0.025
                    for index in range(cell["n"])
                ]
                for x, (metric, _label) in enumerate(metrics):
                    axis.scatter(
                        [x + offset for offset in jitter],
                        [row["effect"][metric] for row in cell["per_seed"]],
                        s=18,
                        facecolor=style.WHITE,
                        edgecolor=treatment_color,
                        linewidth=0.9,
                        zorder=3,
                    )
                axis.text(
                    0.03,
                    0.96,
                    f"exact n={cell['n']}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.5,
                    color=style.MUTED,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "no frozen\nterminal cell",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=style.MUTED,
                )
            axis.set_xlim(-0.42, 1.42)
            axis.set_ylim(low, high)
            axis.set_xticks((0, 1), [label for _metric, label in metrics])
            if domain_index == 0:
                axis.set_ylabel(
                    f"{SCALE_LABELS[scale]}\neffect vs Re:Dr.GRPO",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)

    style.bottom_legend(
        figure,
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=style.WHITE,
                markeredgecolor=treatment_color,
                markersize=4.5,
            )
        ],
        ["frozen paired seed (no mean or interval)"],
        y=0.005,
        ncol=1,
    )
    figure.suptitle(PRIVATE_LABEL, fontsize=style.TITLE_FONT, color=treatment_color, y=0.995)
    figure.text(
        0.5,
        0.955,
        "E112-R1 vs preregistered matched Re:Dr.GRPO · pass 8 · exact frozen subset",
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.text(
        0.5,
        0.925,
        f"Frozen {payload['frozen_at_utc']} · campaign decisions prohibited",
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(left=0.10, right=0.995, top=0.82, bottom=0.18, hspace=0.33, wspace=0.20)
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    args = parser.parse_args()
    output = args.output.resolve()
    _assert_private_output(output)
    payload = build(args.freeze)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    render(payload, output)
    print(f"wrote {output.with_suffix('.json')}")
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
