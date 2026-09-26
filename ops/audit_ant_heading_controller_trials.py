#!/usr/bin/env python3
"""Audit the three prospectively separated Ant controller admission attempts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "var/artifacts/ant_maze_controller_admission_audit.json"
TRIALS = {
    "v1": {
        "receipt": ROOT / "var/maze_runtime/controllers/ant_heading_v1.training.json",
        "source": ROOT / "ops/train_ant_heading_controller.py",
        "requires_stop": True,
    },
    "v2": {
        "receipt": ROOT / "var/maze_runtime/controllers/ant_heading_v2.training.json",
        "source": ROOT / "ops/train_ant_heading_controller_v2.py",
        "requires_stop": True,
    },
    "v3": {
        "receipt": ROOT / "var/maze_runtime/controllers/ant_heading_v3.training.json",
        "source": ROOT / "ops/train_ant_heading_controller_v3.py",
        "requires_stop": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    results = {}
    for version, contract in TRIALS.items():
        receipt = json.loads(contract["receipt"].read_text())
        summary = receipt["evaluation"]["summary"]
        values = [
            value
            for row in receipt["evaluation"]["episodes"]
            for value in row.get("displacement_xy", [])
        ] + list(summary.values())
        finite = all(math.isfinite(float(value)) for value in values)
        source_hash_ok = (
            receipt["hashes"]["training_source_sha256"]
            == _sha256(contract["source"])
        )
        checks = {
            "all_metrics_finite": finite,
            "source_hash_matches": source_hash_ok,
            "minimum_heading_mean_at_least_2": (
                summary["minimum_heading_mean_projected_displacement"] >= 2.0
            ),
            "overall_heading_mean_at_least_4": (
                summary["mean_heading_projected_displacement"] >= 4.0
            ),
            "early_termination_rate_at_most_0p10": (
                summary["early_termination_rate"] <= 0.10
            ),
        }
        if contract["requires_stop"]:
            checks["mean_stop_displacement_at_most_2"] = (
                summary["mean_stop_displacement"] <= 2.0
            )
        results[version] = {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "summary": summary,
            "receipt_sha256": _sha256(contract["receipt"]),
            "model_sha256": receipt["hashes"]["model_sha256"],
            "source_sha256": _sha256(contract["source"]),
        }
    admitted = [version for version, result in results.items() if result["status"] == "pass"]
    payload = {
        "schema_version": "ant-heading-controller-admission-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if admitted else "fail",
        "decision": (
            f"admit_{admitted[0]}" if len(admitted) == 1 else "ant_maze_ineligible"
        ),
        "admitted_versions": admitted,
        "trials": results,
        "information_boundary": (
            "All three trials used open-plane Ant only. No maze map, 0.5B "
            "completion, MaxEnt outcome, or Dr.GRPO outcome was loaded."
        ),
        "audit_source_sha256": _sha256(Path(__file__).resolve()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(
        "[ant-controller-audit] "
        f"status={payload['status']} decision={payload['decision']}"
    )
    raise SystemExit(0 if admitted else 1)


if __name__ == "__main__":
    main()
