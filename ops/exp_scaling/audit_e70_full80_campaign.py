#!/usr/bin/env python3
"""Aggregate the fail-closed terminal state of the clean E70 80-cell campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "var/artifacts/e70_clean_full80_campaign_audit_latest.json"
STAGE_A_AUDIT = ROOT / "var/artifacts/e70_clean_stage_a_05b_audit_latest.json"
STAGE_B = {
    "pantry_plan": (
        ROOT / "var/artifacts/pantry_stage_b_05b_12pass_identity.json",
        ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json",
    ),
    "point_maze": (
        ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_identity.json",
        ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit.json",
    ),
    "ant_maze": (
        ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_identity.json",
        ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit.json",
    ),
    "point_maze_geometry_shift": (
        ROOT / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_identity.json",
        ROOT / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_audit.json",
    ),
}
CONSTRUCTIVE_QUALIFICATIONS = {
    "v6_gate": ROOT / "var/artifacts/constructive_code_v6_gate_audit.json",
    "v6_viability": ROOT / "var/artifacts/constructive_code_v6_coder_05b_viability.json",
    "v6_paired_smoke": ROOT / "var/artifacts/constructive_code_v6_paired_smoke_audit.json",
    "v7_train_only_sft": ROOT / "var/artifacts/constructive_code_v7_sft.json",
    "v7_post_sft_viability": ROOT / "var/artifacts/constructive_code_v7_post_sft_viability.json",
    "v8_15b_capacity_retry": ROOT / "var/artifacts/constructive_code_v8_coder_15b_viability.json",
}
REPLACEMENT_QUALIFICATIONS = {
    "point_maze_geometry_shift_admission": (
        ROOT / "var/artifacts/point_maze_geometry_shift_v1_admission_audit.json"
    ),
    "point_maze_geometry_shift_viability": (
        ROOT / "var/artifacts/point_maze_geometry_shift_05b_viability_v1.json"
    ),
    "point_maze_geometry_shift_paired_smoke": (
        ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1_audit.json"
    ),
}

REQUESTED_SEMANTIC_DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factor",
    "mathir",
    "constructive_code",
    "pantry_plan",
    "point_maze",
    "ant_maze",
)
FINAL_TRAINING_ROWS = (
    "graph_coloring",
    "countdown",
    "python_factor",
    "mathir",
    "point_maze_geometry_shift",
    "pantry_plan",
    "point_maze",
    "ant_maze",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def identity_cell_count(identity: Mapping[str, Any]) -> int:
    for key in ("jobs", "cells"):
        records = identity.get(key)
        if isinstance(records, Mapping):
            return len(records)
    return 0


def campaign_cells(
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    domains = stage_a.get("domains", {})
    if isinstance(domains, Mapping):
        for domain, payload in domains.items():
            runs = payload.get("runs", []) if isinstance(payload, Mapping) else []
            if not isinstance(runs, list):
                continue
            for run in runs:
                if not isinstance(run, Mapping):
                    continue
                cells.append(
                    {
                        "domain": str(domain),
                        "arm": run.get("arm"),
                        "seed": run.get("seed"),
                        "job_id": run.get("job_id"),
                        "state": (
                            "audited_terminal"
                            if run.get("terminal")
                            else "in_progress"
                        ),
                        "latest_step": run.get("latest_step"),
                        "expected_step": run.get("expected_step"),
                        "training_passes": run.get("training_passes"),
                        "source": "stage_a_audit",
                    }
                )
    for domain, (identity, audit) in stage_b.items():
        jobs = identity.get("jobs", {})
        if not isinstance(jobs, Mapping):
            continue
        terminal = audit.get("status") == "pass"
        for label, job_id in jobs.items():
            arm, separator, seed_label = str(label).partition("/")
            seed = None
            if separator and seed_label.startswith("s"):
                try:
                    seed = int(seed_label[1:])
                except ValueError:
                    pass
            cells.append(
                {
                    "domain": domain,
                    "arm": arm or None,
                    "seed": seed,
                    "job_id": job_id,
                    "state": "audited_terminal" if terminal else "submitted",
                    "source": "stage_b_identity_and_audit",
                }
            )
    return sorted(
        cells,
        key=lambda row: (
            str(row.get("domain")),
            str(row.get("arm")),
            int(row.get("seed") or -1),
        ),
    )


def summarize(
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    qualifications: Mapping[str, Mapping[str, Any]],
    replacement_qualifications: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    violations = []
    stage_summary = stage_a.get("summary", {})
    if not isinstance(stage_summary, Mapping):
        stage_summary = {}
    stage_submitted = int(stage_summary.get("identity_runs", 0) or 0)
    stage_terminal = int(stage_summary.get("terminal_runs", 0) or 0)
    if stage_a.get("status") == "fail":
        violations.append({"code": "stage_a_audit_failed"})

    rows = {}
    submitted = stage_submitted
    terminal = stage_terminal
    for domain, (identity, audit) in stage_b.items():
        cell_count = identity_cell_count(identity)
        audit_status = audit.get("status")
        row_terminal = 10 if audit_status == "pass" else 0
        submitted += cell_count
        terminal += row_terminal
        if cell_count not in {0, 10}:
            violations.append(
                {
                    "code": "partial_stage_b_identity",
                    "domain": domain,
                    "observed_cells": cell_count,
                }
            )
        if audit_status == "fail":
            violations.append({"code": "stage_b_audit_failed", "domain": domain})
        rows[domain] = {
            "submitted_cells": cell_count,
            "audited_terminal_cells": row_terminal,
            "audit_status": audit_status or "missing",
        }

    qualification_status = {}
    negative_qualification_outcomes = []
    for name, audit in qualifications.items():
        status = str(audit.get("status") or "missing")
        qualification_status[name] = status
        if status == "fail":
            negative_qualification_outcomes.append(name)

    replacement_status = {}
    for name, audit in (replacement_qualifications or {}).items():
        replacement_status[name] = str(audit.get("status") or "missing")

    cells = campaign_cells(stage_a, stage_b)
    stage_domains = stage_a.get("domains", {})
    if isinstance(stage_domains, Mapping) and stage_domains and len(cells) != submitted:
        violations.append(
            {
                "code": "cell_registry_count_mismatch",
                "enumerated_cells": len(cells),
                "submitted_cells": submitted,
            }
        )
    if submitted > 80 or terminal > 80 or terminal > submitted:
        violations.append(
            {
                "code": "campaign_count_invariant_failed",
                "submitted": submitted,
                "terminal": terminal,
            }
        )
    passed = submitted == 80 and terminal == 80 and not violations
    return {
        "schema": "e70-clean-full80-replacement-campaign-audit-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else ("fail" if violations else "in_progress"),
        "summary": {
            "expected_cells": 80,
            "submitted_cells": submitted,
            "audited_terminal_cells": terminal,
            "remaining_unsubmitted_cells": max(0, 80 - submitted),
            "remaining_unterminal_cells": max(0, 80 - terminal),
            "registry_enumerated_cells": len(cells),
            "registry_complete": len(cells) == submitted == 80,
        },
        "stage_a": {
            "status": stage_a.get("status", "missing"),
            "submitted_cells": stage_submitted,
            "audited_terminal_cells": stage_terminal,
        },
        "stage_b": rows,
        "cells": cells,
        "semantic_domain_coverage": {
            "status": "incomplete",
            "requested_domains": list(REQUESTED_SEMANTIC_DOMAINS),
            "covered_requested_domains": [
                domain
                for domain in REQUESTED_SEMANTIC_DOMAINS
                if domain != "constructive_code"
            ],
            "missing_requested_domains": ["constructive_code"],
            "independent_domains_covered": 7,
            "final_training_rows": list(FINAL_TRAINING_ROWS),
            "configuration_level_replacement": "point_maze_geometry_shift",
            "replacement_is_independent_domain": False,
            "interpretation": (
                "80 submitted cells cover eight rows but only seven "
                "independent requested semantic domains"
            ),
        },
        "constructive_qualifications": qualification_status,
        "replacement_qualifications": replacement_status,
        "roster_amendment": {
            "removed_row": "constructive_code",
            "replacement_row": "point_maze_geometry_shift",
            "replacement_is_independent_domain": False,
        },
        "negative_qualification_outcomes": negative_qualification_outcomes,
        "violations": violations,
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        read_json(STAGE_A_AUDIT),
        {
            name: (read_json(identity), read_json(audit))
            for name, (identity, audit) in STAGE_B.items()
        },
        {name: read_json(path) for name, path in CONSTRUCTIVE_QUALIFICATIONS.items()},
        {name: read_json(path) for name, path in REPLACEMENT_QUALIFICATIONS.items()},
    )
    atomic_json(args.output, payload)
    summary = payload["summary"]
    print(
        "[e70-full80-audit] "
        f"status={payload['status']} submitted={summary['submitted_cells']}/80 "
        f"terminal={summary['audited_terminal_cells']}/80 "
        f"violations={len(payload['violations'])} output={args.output}",
        flush=True,
    )
    raise SystemExit(1 if payload["status"] == "fail" else 0)


if __name__ == "__main__":
    main()
