"""Domain-filtered readers for terminal paper comparison figures.

The live Figure 4 reader opens every run in a scale because its dashboard has
to draw every domain. Terminal paper figures often qualify only one or two
domains; these helpers apply the already-computed scientific gate before any
evaluation log is opened.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import plot_e78_figure4_preview as live
import plot_figure4_with_falcon_preview as wall


def family_snapshot(
    ledger_path: Path,
    allowed_domains: Iterable[str],
) -> dict[str, Any]:
    """Read only allowed static-domain runs."""

    allowed = set(allowed_domains)
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    interval = int(payload["checkpoint_interval_steps"])
    target = int(payload["target_steps"])
    curves: dict[str, dict[str, dict[int, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    selected = [
        run for run in payload["runs"] if str(run["domain"]) in allowed
    ]
    for run in selected:
        curve = live._run_curve(
            Path(run["run_dir"]), interval=interval, target=target
        )
        if curve:
            curves[str(run["domain"])][str(run["arm"])][int(run["seed"])] = curve

    snapshot: dict[str, Any] = {
        "curves": curves,
        "domains": [
            str(domain)
            for domain in payload["domains"]
            if str(domain) in allowed
        ],
        "interval": interval,
        "passes": int(payload["passes"]),
        "steps_per_pass": int(payload["train_rows"]),
        "target": target,
        "total_runs": len(selected),
    }
    return snapshot


def attach_semantic(
    snapshot: dict[str, Any],
    ledger_path: Path,
    allowed_domains: Iterable[str],
    *,
    arm: str,
    comparator: str,
    cohort: str | None,
) -> dict[str, Any]:
    """Attach only semantic-arm runs belonging to terminal domains."""

    snapshot.setdefault("semantic_arms", [])
    if not ledger_path.is_file():
        return snapshot
    allowed = set(allowed_domains)
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    interval = int(payload["checkpoint_interval_steps"])
    target = int(payload["target_steps"])
    selected = [
        run for run in payload["runs"] if str(run["domain"]) in allowed
    ]
    curves: dict[str, dict[int, dict[int, float]]] = {}
    for run in selected:
        curve = live._run_curve(
            Path(run["run_dir"]), interval=interval, target=target
        )
        if curve:
            curves.setdefault(str(run["domain"]), {})[int(run["seed"])] = curve

    superseded: list[str] = []
    if cohort and wall.REPAIR_LEDGER.is_file():
        repair = json.loads(wall.REPAIR_LEDGER.read_text(encoding="utf-8"))
        mine = [
            run
            for run in repair["runs"]
            if str(run.get("parent")) == cohort
            and str(run["domain"]) in allowed
        ]
        superseded = sorted({str(run["domain"]) for run in mine})
        for domain in superseded:
            curves.pop(domain, None)
        repair_interval = int(repair["checkpoint_interval_steps"])
        repair_target = int(repair["target_steps"])
        for run in mine:
            curve = live._run_curve(
                Path(run["run_dir"]),
                interval=repair_interval,
                target=repair_target,
            )
            if curve:
                curves.setdefault(str(run["domain"]), {})[int(run["seed"])] = curve

    snapshot["semantic_arms"].append(
        {
            "arm": arm,
            "comparator": comparator,
            "curves": curves,
            "total_runs": len(selected),
            "observed_runs": sum(len(seeds) for seeds in curves.values()),
            "coefficient": float(payload.get("semantic_coefficient", 0.10)),
            "superseded_domains": superseded,
        }
    )
    return snapshot
