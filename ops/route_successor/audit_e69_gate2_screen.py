#!/usr/bin/env python3
"""Audit and summarize the prospectively frozen E69 Gate 2 screen."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e69_gate2_compute_matched_screen_identity.json"
OUT_JSON = ROOT / "var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json"
OUT_MD = ROOT / "paper/results/e69_gate2_compute_matched_screen_live.md"
REPAIR_IDENTITY = (
    ROOT / "var/artifacts/e69_gate2_math_endpoint_repair_identity.json"
)
PLACEMENT_REPAIR_IDENTITY = (
    ROOT / "var/artifacts/e69_gate2_pending_placement_repair_identity.json"
)
GRAPH_PREEMPTION_REPAIR_IDENTITY = (
    ROOT
    / "var/artifacts/e69_gate2_graph_a5000_preemption_repair_identity.json"
)
R1_EXECUTION_REPAIR_IDENTITY = (
    ROOT / "var/artifacts/e69_gate2_r1_execution_repair_identity.json"
)
R1_EXECUTION_REPAIR_PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e69_gate2_r1_execution_repair_20260728.md"
)
R1_EXECUTION_REPAIR_LAUNCHER = (
    ROOT
    / "ops/route_successor/"
    "launch_e69_gate2_r1_execution_repair.sh"
)
R1_GRAPH_MANIFEST = (
    ROOT
    / "var/artifacts/"
    "gce69_gate2_r1_execution_repair_retry1_comparative_jobs.tsv"
)
R1_PYTHON_MANIFEST = (
    ROOT
    / "var/artifacts/"
    "pye69_gate2_r1_execution_repair_retry1_comparative_jobs.tsv"
)
R1_SOURCE_ACCOUNTING = (
    ROOT / "var/artifacts/e69_gate2_r1_excluded_source_accounting.txt"
)
R2_IMPLEMENTATION_REPAIR_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e69_gate2_r2_route_endpoint_bookkeeping_repair_identity.json"
)
R2_IMPLEMENTATION_REPAIR_PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e69_gate2_r2_route_endpoint_bookkeeping_repair_20260728.md"
)
R2_IMPLEMENTATION_REPAIR_LAUNCHER = (
    ROOT
    / "ops/route_successor/"
    "launch_e69_gate2_r2_route_endpoint_repair.sh"
)
R2_PYTHON_MANIFEST = (
    ROOT
    / "var/artifacts/"
    "pye69_gate2_r2_route_endpoint_repair_comparative_jobs.tsv"
)
R2_FAILED_R1_ACCOUNTING = (
    ROOT / "var/artifacts/e69_gate2_r2_failed_r1_python_accounting.txt"
)
R2_FAILED_R1_LOG = ROOT / "var/artifacts/logs/xdr_train-30168831.out"
R3_MATH_EVAL_REPAIR_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e69_gate2_r3_math_dev_evaluation_split_repair_identity.json"
)
R3_MATH_EVAL_REPAIR_PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e69_gate2_r3_math_dev_evaluation_split_repair_20260729.md"
)
R3_MATH_EVAL_REPAIR_LAUNCHER = (
    ROOT
    / "ops/route_successor/"
    "launch_e69_gate2_r3_math_dev_evaluation_split_repair.sh"
)
R3_MATH_MANIFEST = (
    ROOT
    / "var/artifacts/"
    "mde69_gate2_r3_eval_split_repair_comparative_jobs.tsv"
)
R3_STARTUP_REJECTED_MANIFEST = (
    ROOT
    / "var/artifacts/"
    "mde69_gate2_r3_eval_split_repair_startup_rejected_jobs.tsv"
)
R3_STARTUP_REJECTED_ACCOUNTING = (
    ROOT / "var/artifacts/e69_gate2_r3_startup_rejected_accounting.txt"
)
PRECHECKPOINT_REQUEUE_REPAIR_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e69_gate2_precheckpoint_requeue_attempt_repair_identity.json"
)
ROUTE_TEMPORAL_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e69_gate2_route_temporal_observer_amendment_20260728.md"
)
ROUTE_TEMPORAL_SNAPSHOTS = (
    ROOT / "var/artifacts/e69_gate2_route_temporal_snapshots.json"
)
R1_ROUTE_TEMPORAL_SNAPSHOTS = (
    ROOT / "var/artifacts/e69_gate2_r1_route_temporal_snapshots.json"
)
R2_ROUTE_TEMPORAL_SNAPSHOTS = (
    ROOT / "var/artifacts/e69_gate2_r2_route_temporal_snapshots.json"
)
PASSES = (0, 1, 2, 3, 4, 5, 6)
POOL_SIZE = {
    "graph_coloring": 192,
    "countdown": 384,
    "python_factor": 384,
    "mathir": 384,
    "math_dev": 384,
}
MODEBENCH = ("graph_coloring", "countdown", "python_factor", "mathir")
CONTROL = "grpo"
SUCCESSOR = "verified_route_successor"
ENDPOINT = "verified_first_global_replay_canonical"
METRICS = ("greedy", "mean8", "pass8", "distinct8")
UNCAUGHT = re.compile(
    r"\[rank\d+\]: Traceback \(most recent call last\)|"
    r"CUDA out of memory|torch\.OutOfMemoryError|ChildFailedError|"
    r"RayActorError|segmentation fault",
    re.IGNORECASE,
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric(row: dict[str, Any], name: str) -> Any:
    for prefix in ("train/", "actor/", ""):
        key = f"{prefix}{name}"
        if key in row:
            return row[key]
    return None


def _close(value: Any, expected: float, *, tolerance: float = 1e-8) -> bool:
    return _finite(value) and math.isclose(
        float(value),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_file_hashes(path: Path) -> dict[str, str]:
    return {
        str(candidate.relative_to(path)): _sha256(candidate)
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix != ".pyc"
    }


def _source_tree_sha256(path: Path) -> str:
    rows = []
    for relative, digest in sorted(_source_file_hashes(path).items()):
        rows.append(f"{digest}  ./{relative}\n".encode("utf-8"))
    return hashlib.sha256(b"".join(rows)).hexdigest()


def _effective_jobs(
    identity: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    """Resolve prospectively recorded pre-optimizer job replacements."""

    jobs = {
        domain: [dict(row) for row in rows]
        for domain, rows in identity["jobs"].items()
    }
    repairs: list[dict[str, Any]] = []
    violations: list[str] = []
    if not REPAIR_IDENTITY.is_file():
        pass
    else:
        repair = json.loads(REPAIR_IDENTITY.read_text(encoding="utf-8"))
        if (
            repair.get("schema")
            != "e69_gate2_math_endpoint_startup_repair_v1"
            or repair.get("original_identity_sha256") != _sha256(IDENTITY)
            or repair.get("invalid_job", {}).get("job_id") != 30159730
            or repair.get("invalid_job", {}).get("optimizer_records") != 0
            or repair.get("terminal_outcomes_observed_before_repair") is not False
        ):
            violations.append("E69 Gate 2 MATH endpoint repair identity mismatch")
            return jobs, repairs, violations
        replacement = dict(repair["replacement"])
        if (
            replacement.get("arm") != ENDPOINT
            or int(replacement.get("seed", -1)) != 43
        ):
            violations.append("E69 Gate 2 MATH endpoint replacement cell mismatch")
            return jobs, repairs, violations
        matches = [
            index
            for index, row in enumerate(jobs["math_dev"])
            if int(row["job_id"]) == 30159730
        ]
        if len(matches) != 1:
            violations.append("E69 Gate 2 invalid MATH endpoint cell is not unique")
            return jobs, repairs, violations
        original = jobs["math_dev"][matches[0]]
        if _run_dir(str(original["run_stamp"]), int(original["job_id"])) is not None:
            violations.append("excluded Gate 2 job unexpectedly has a run directory")
            return jobs, repairs, violations
        failure_log = ROOT / "var/artifacts/logs/xdr_train-30159730.err"
        failure_text = (
            failure_log.read_text(encoding="utf-8", errors="replace")
            if failure_log.is_file()
            else ""
        )
        if str(repair["invalid_job"]["failure"]) not in failure_text:
            violations.append("excluded Gate 2 job failure signature is absent")
            return jobs, repairs, violations
        jobs["math_dev"][matches[0]] = replacement
        repairs.append(
            {
                "kind": "preoptimizer_startup",
                "invalid_job_id": 30159730,
                "replacement_job_id": int(replacement["job_id"]),
                "identity": str(REPAIR_IDENTITY.resolve()),
                "identity_sha256": _sha256(REPAIR_IDENTITY),
                "attempt_selection": repair["attempt_selection"],
            }
        )

    if not PLACEMENT_REPAIR_IDENTITY.is_file():
        return jobs, repairs, violations
    placement = json.loads(
        PLACEMENT_REPAIR_IDENTITY.read_text(encoding="utf-8")
    )
    if (
        placement.get("schema") != "e69_gate2_pending_placement_repair_v1"
        or placement.get("original_identity_sha256") != _sha256(IDENTITY)
        or placement.get("invalid_jobs_optimizer_records") != 0
        or placement.get("outcomes_observed_before_repair") is not False
    ):
        violations.append("E69 Gate 2 placement repair identity mismatch")
        return jobs, repairs, violations
    substitutions: list[dict[str, int]] = []
    for domain in ("graph_coloring", "countdown", "python_factor"):
        mappings = placement.get("mappings", {}).get(domain, [])
        if len(mappings) != 4:
            violations.append(f"E69 Gate 2 {domain} placement mapping is incomplete")
            return jobs, repairs, violations
        for mapping in mappings:
            invalid_job_id = int(mapping["invalid_job_id"])
            matches = [
                index
                for index, row in enumerate(jobs[domain])
                if int(row["job_id"]) == invalid_job_id
            ]
            if len(matches) != 1:
                violations.append(
                    f"E69 Gate 2 placement cell {invalid_job_id} is not unique"
                )
                return jobs, repairs, violations
            original = jobs[domain][matches[0]]
            if (
                original["arm"] != mapping["arm"]
                or int(mapping["seed"]) != int(original["seed"])
                or _run_dir(
                    str(original["run_stamp"]),
                    invalid_job_id,
                )
                is not None
            ):
                violations.append(
                    f"E69 Gate 2 excluded placement cell {invalid_job_id} mismatch"
                )
                return jobs, repairs, violations
            replacement = {
                "arm": mapping["arm"],
                "seed": int(mapping["seed"]),
                "job_id": int(mapping["replacement_job_id"]),
                "run_stamp": mapping["run_stamp"],
            }
            jobs[domain][matches[0]] = replacement
            substitutions.append(
                {
                    "invalid_job_id": invalid_job_id,
                    "replacement_job_id": int(mapping["replacement_job_id"]),
                }
            )
    repairs.append(
        {
            "kind": "pending_placement",
            "substitutions": substitutions,
            "identity": str(PLACEMENT_REPAIR_IDENTITY.resolve()),
            "identity_sha256": _sha256(PLACEMENT_REPAIR_IDENTITY),
            "attempt_selection": placement["attempt_selection"],
        }
    )
    if not GRAPH_PREEMPTION_REPAIR_IDENTITY.is_file():
        return jobs, repairs, violations
    graph_repair = json.loads(
        GRAPH_PREEMPTION_REPAIR_IDENTITY.read_text(encoding="utf-8")
    )
    if (
        graph_repair.get("schema")
        != "e69_gate2_graph_a5000_preemption_repair_v1"
        or graph_repair.get("original_identity_sha256") != _sha256(IDENTITY)
        or graph_repair.get("parent_placement_repair_identity_sha256")
        != _sha256(PLACEMENT_REPAIR_IDENTITY)
        or graph_repair.get("terminal_outcomes_observed_before_repair")
        is not False
        or graph_repair.get("repair_decision_basis")
        != "scheduler_preemption_and_node_inventory_only"
    ):
        violations.append("E69 Gate 2 Graph preemption repair identity mismatch")
        return jobs, repairs, violations
    mappings = graph_repair.get("mappings", [])
    invalid_attempts = graph_repair.get("invalid_attempts", [])
    if len(mappings) != 4 or len(invalid_attempts) != 4:
        violations.append("E69 Gate 2 Graph preemption mapping is incomplete")
        return jobs, repairs, violations
    invalid_by_job = {
        int(row["job_id"]): row for row in invalid_attempts
    }
    substitutions = []
    for mapping in mappings:
        invalid_job_id = int(mapping["invalid_job_id"])
        matches = [
            index
            for index, row in enumerate(jobs["graph_coloring"])
            if int(row["job_id"]) == invalid_job_id
        ]
        if len(matches) != 1:
            violations.append(
                f"E69 Gate 2 Graph preemption cell {invalid_job_id} is not unique"
            )
            return jobs, repairs, violations
        original = jobs["graph_coloring"][matches[0]]
        invalid = invalid_by_job.get(invalid_job_id)
        run_dir = _run_dir(str(original["run_stamp"]), invalid_job_id)
        if (
            invalid is None
            or invalid.get("terminal") is not False
            or original["arm"] != mapping["arm"]
            or int(mapping["seed"]) != int(original["seed"])
            or run_dir is None
            or (run_dir / "saved_models").exists()
        ):
            violations.append(
                f"E69 Gate 2 excluded Graph attempt {invalid_job_id} mismatch"
            )
            return jobs, repairs, violations
        replacement = {
            "arm": mapping["arm"],
            "seed": int(mapping["seed"]),
            "job_id": int(mapping["replacement_job_id"]),
            "run_stamp": mapping["run_stamp"],
        }
        jobs["graph_coloring"][matches[0]] = replacement
        substitutions.append(
            {
                "invalid_job_id": invalid_job_id,
                "replacement_job_id": int(mapping["replacement_job_id"]),
            }
        )
    repairs.append(
        {
            "kind": "graph_preemption",
            "substitutions": substitutions,
            "identity": str(GRAPH_PREEMPTION_REPAIR_IDENTITY.resolve()),
            "identity_sha256": _sha256(GRAPH_PREEMPTION_REPAIR_IDENTITY),
            "attempt_selection": graph_repair["attempt_selection"],
        }
    )
    if not R1_EXECUTION_REPAIR_IDENTITY.is_file():
        return jobs, repairs, violations
    r1 = json.loads(
        R1_EXECUTION_REPAIR_IDENTITY.read_text(encoding="utf-8")
    )
    r1_protocol = Path(str(r1.get("protocol", "")))
    if (
        r1.get("schema") != "e69_gate2_r1_execution_repair_v1"
        or r1.get("original_identity_sha256") != _sha256(IDENTITY)
        or r1.get("parent_placement_repair_identity_sha256")
        != _sha256(PLACEMENT_REPAIR_IDENTITY)
        or r1.get("parent_graph_preemption_repair_identity_sha256")
        != _sha256(GRAPH_PREEMPTION_REPAIR_IDENTITY)
        or r1_protocol.resolve() != R1_EXECUTION_REPAIR_PROTOCOL.resolve()
        or not r1_protocol.is_file()
        or r1.get("protocol_sha256") != _sha256(r1_protocol)
        or r1.get("launcher_sha256") != _sha256(R1_EXECUTION_REPAIR_LAUNCHER)
        or r1.get("manifest_sha256", {}).get("graph_coloring")
        != _sha256(R1_GRAPH_MANIFEST)
        or r1.get("manifest_sha256", {}).get("python_factor")
        != _sha256(R1_PYTHON_MANIFEST)
        or r1.get("source_accounting_sha256")
        != _sha256(R1_SOURCE_ACCOUNTING)
        or r1.get("source_hash") != identity.get("source_hash")
        or r1.get("execution_surface_hash")
        != identity.get("execution_surface_hash")
        or r1.get("terminal_outcomes_observed_before_repair") is not True
        or r1.get("outcome_tuning") is not False
        or r1.get("algorithm_or_gate_change") is not False
        or r1.get("repair_decision_basis")
        != "frozen_scheduler_integrity_rule_not_metric_values"
        or r1.get("math500_sealed") is not True
    ):
        violations.append("E69 Gate 2 R1 execution repair identity mismatch")
        return jobs, repairs, violations
    placement = r1.get("placement", {})
    if placement != {
        "account": "mltheory",
        "gres": "gpu:a5000:1",
        "nodelist": "node105,node202,node203,node204",
        "requested_partition": "all",
        "resolved_partition": "mltheory",
    }:
        violations.append("E69 Gate 2 R1 placement identity mismatch")
        return jobs, repairs, violations
    substitutions = []
    expected_sources = {
        "graph_coloring": {30160592, 30160594, 30160595, 30160596},
        "python_factor": {30160205},
    }
    for domain in ("graph_coloring", "python_factor"):
        mappings = r1.get("mappings", {}).get(domain, [])
        if (
            len(mappings) != len(expected_sources[domain])
            or {int(row["invalid_job_id"]) for row in mappings}
            != expected_sources[domain]
        ):
            violations.append(f"E69 Gate 2 R1 {domain} mapping is incomplete")
            return jobs, repairs, violations
        for mapping in mappings:
            invalid_job_id = int(mapping["invalid_job_id"])
            matches = [
                index
                for index, row in enumerate(jobs[domain])
                if int(row["job_id"]) == invalid_job_id
            ]
            if len(matches) != 1:
                violations.append(
                    f"E69 Gate 2 R1 source cell {invalid_job_id} is not unique"
                )
                return jobs, repairs, violations
            original = jobs[domain][matches[0]]
            if (
                original["arm"] != mapping["arm"]
                or int(original["seed"]) != int(mapping["seed"])
            ):
                violations.append(
                    f"E69 Gate 2 R1 source cell {invalid_job_id} drift"
                )
                return jobs, repairs, violations
            replacement = {
                "arm": mapping["arm"],
                "seed": int(mapping["seed"]),
                "job_id": int(mapping["replacement_job_id"]),
                "run_stamp": mapping["run_stamp"],
            }
            jobs[domain][matches[0]] = replacement
            substitutions.append(
                {
                    "domain": domain,
                    "invalid_job_id": invalid_job_id,
                    "replacement_job_id": int(mapping["replacement_job_id"]),
                }
            )
    repairs.append(
        {
            "kind": "execution_r1",
            "substitutions": substitutions,
            "identity": str(R1_EXECUTION_REPAIR_IDENTITY.resolve()),
            "identity_sha256": _sha256(R1_EXECUTION_REPAIR_IDENTITY),
            "attempt_selection": r1["attempt_selection"],
            "outcome_tuning": False,
        }
    )
    if not R2_IMPLEMENTATION_REPAIR_IDENTITY.is_file():
        return jobs, repairs, violations
    r2 = json.loads(
        R2_IMPLEMENTATION_REPAIR_IDENTITY.read_text(encoding="utf-8")
    )
    r2_protocol = Path(str(r2.get("protocol", "")))
    parent_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e69_gate2_{r2.get('parent_source_hash', '')}"
        / "src"
    )
    repaired_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e69_gate2_r2_{r2.get('source_hash', '')}"
        / "src"
    )
    changed_files = (
        set(_source_file_hashes(parent_source))
        | set(_source_file_hashes(repaired_source))
        if parent_source.is_dir() and repaired_source.is_dir()
        else set()
    )
    changed_files = {
        name
        for name in changed_files
        if _source_file_hashes(parent_source).get(name)
        != _source_file_hashes(repaired_source).get(name)
    }
    if (
        r2.get("schema")
        != "e69_gate2_r2_route_endpoint_bookkeeping_repair_v1"
        or r2.get("original_identity_sha256") != _sha256(IDENTITY)
        or r2.get("parent_r1_identity_sha256")
        != _sha256(R1_EXECUTION_REPAIR_IDENTITY)
        or r2_protocol.resolve()
        != R2_IMPLEMENTATION_REPAIR_PROTOCOL.resolve()
        or not r2_protocol.is_file()
        or r2.get("protocol_sha256") != _sha256(r2_protocol)
        or r2.get("launcher_sha256")
        != _sha256(R2_IMPLEMENTATION_REPAIR_LAUNCHER)
        or r2.get("manifest_sha256") != _sha256(R2_PYTHON_MANIFEST)
        or r2.get("failed_r1_accounting_sha256")
        != _sha256(R2_FAILED_R1_ACCOUNTING)
        or r2.get("failed_r1_log_sha256") != _sha256(R2_FAILED_R1_LOG)
        or r2.get("parent_route_temporal_snapshot_sha256")
        != _sha256(R1_ROUTE_TEMPORAL_SNAPSHOTS)
        or r2.get("parent_source_hash") != r1.get("source_hash")
        or not repaired_source.is_dir()
        or _source_tree_sha256(repaired_source) != r2.get("source_hash")
        or changed_files != {"oat_drgrpo/verified_route_library.py"}
        or r2.get("old_route_library_sha256")
        != _sha256(parent_source / "oat_drgrpo/verified_route_library.py")
        or r2.get("repaired_route_library_sha256")
        != _sha256(repaired_source / "oat_drgrpo/verified_route_library.py")
        or r2.get("execution_surface_hash")
        != r1.get("execution_surface_hash")
        or r2.get("r1_failure_observed_before_repair") is not True
        or r2.get("repaired_outcomes_observed_before_freeze") is not False
        or r2.get("outcome_tuning") is not False
        or r2.get("algorithm_or_gate_change") is not False
        or r2.get("implementation_contract_repair") is not True
        or r2.get("repair_decision_basis")
        != "hard_invariant_exception_and_frozen_identity_semantics_only"
        or r2.get("math500_sealed") is not True
    ):
        violations.append("E69 Gate 2 R2 implementation repair identity mismatch")
        return jobs, repairs, violations
    if r2.get("placement") != {
        "account": "mltheory",
        "gres": "gpu:a5000:1",
        "nodelist": "node105,node202,node203,node204",
        "requested_partition": "all",
        "resolved_partition": "mltheory",
    }:
        violations.append("E69 Gate 2 R2 placement identity mismatch")
        return jobs, repairs, violations
    mapping = r2.get("mapping", {})
    matches = [
        index
        for index, row in enumerate(jobs["python_factor"])
        if int(row["job_id"]) == 30168831
    ]
    if (
        len(matches) != 1
        or mapping.get("domain") != "python_factor"
        or mapping.get("arm") != SUCCESSOR
        or int(mapping.get("seed", -1)) != 43
        or int(mapping.get("invalid_job_id", -1)) != 30168831
        or jobs["python_factor"][matches[0]]["arm"] != SUCCESSOR
        or int(jobs["python_factor"][matches[0]]["seed"]) != 43
    ):
        violations.append("E69 Gate 2 R2 Python mapping is invalid")
        return jobs, repairs, violations
    replacement = {
        "arm": SUCCESSOR,
        "seed": 43,
        "job_id": int(mapping["replacement_job_id"]),
        "run_stamp": str(mapping["run_stamp"]),
    }
    jobs["python_factor"][matches[0]] = replacement
    repairs.append(
        {
            "kind": "implementation_r2",
            "substitutions": [
                {
                    "domain": "python_factor",
                    "invalid_job_id": 30168831,
                    "replacement_job_id": int(mapping["replacement_job_id"]),
                }
            ],
            "identity": str(R2_IMPLEMENTATION_REPAIR_IDENTITY.resolve()),
            "identity_sha256": _sha256(R2_IMPLEMENTATION_REPAIR_IDENTITY),
            "attempt_selection": r2["attempt_selection"],
            "outcome_tuning": False,
            "implementation_contract_repair": True,
        }
    )
    if not R3_MATH_EVAL_REPAIR_IDENTITY.is_file():
        return jobs, repairs, violations
    r3 = json.loads(
        R3_MATH_EVAL_REPAIR_IDENTITY.read_text(encoding="utf-8")
    )
    r3_protocol = Path(str(r3.get("protocol", "")))
    r3_parent_source = repaired_source
    r3_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e69_gate2_r3_{r3.get('source_hash', '')}"
        / "src"
    )
    if r3_parent_source.is_dir() and r3_source.is_dir():
        parent_hashes = _source_file_hashes(r3_parent_source)
        r3_hashes = _source_file_hashes(r3_source)
        r3_changed_files = {
            name
            for name in set(parent_hashes) | set(r3_hashes)
            if parent_hashes.get(name) != r3_hashes.get(name)
        }
    else:
        r3_changed_files = set()
    if (
        r3.get("schema")
        != "e69_gate2_r3_math_dev_evaluation_split_repair_v1"
        or r3.get("original_identity_sha256") != _sha256(IDENTITY)
        or r3.get("parent_r2_identity_sha256")
        != _sha256(R2_IMPLEMENTATION_REPAIR_IDENTITY)
        or r3_protocol.resolve() != R3_MATH_EVAL_REPAIR_PROTOCOL.resolve()
        or not r3_protocol.is_file()
        or r3.get("protocol_sha256") != _sha256(r3_protocol)
        or r3.get("launcher_sha256") != _sha256(R3_MATH_EVAL_REPAIR_LAUNCHER)
        or r3.get("manifest_sha256") != _sha256(R3_MATH_MANIFEST)
        or r3.get("startup_rejected_manifest_sha256")
        != _sha256(R3_STARTUP_REJECTED_MANIFEST)
        or r3.get("startup_rejected_accounting_sha256")
        != _sha256(R3_STARTUP_REJECTED_ACCOUNTING)
        or r3.get("startup_rejected_jobs") != [30172460, 30172461]
        or r3.get("parent_source_hash") != r2.get("source_hash")
        or not r3_source.is_dir()
        or _source_tree_sha256(r3_source) != r3.get("source_hash")
        or r3_changed_files
        != {"oat_drgrpo/args.py", "oat_drgrpo/learner/init.py"}
        or r3.get("execution_surface_hash")
        != r2.get("execution_surface_hash")
        or r3.get("gate_outcomes_observed_before_freeze") is not False
        or r3.get("training_telemetry_observed_before_freeze") is not True
        or r3.get("outcome_tuning") is not False
        or r3.get("algorithm_or_gate_change") is not False
        or r3.get("implementation_contract_repair") is not True
        or r3.get("math500_sealed") is not True
        or r3.get("only_configuration_change")
        != {
            "name": "OAT_ZERO_TEST_SPLIT",
            "invalid": "math",
            "replacement": "math_dev",
        }
    ):
        violations.append(
            "E69 Gate 2 R3 MATH evaluation-split repair identity mismatch"
        )
        return jobs, repairs, violations
    mappings = r3.get("mappings", [])
    excluded = {
        int(row["job_id"]): row
        for row in r3.get("excluded_attempts", [])
        if isinstance(row, dict) and "job_id" in row
    }
    if (
        len(mappings) != 2
        or len(excluded) != 2
        or {int(row.get("invalid_job_id", -1)) for row in mappings}
        != {30159729, 30160101}
        or {str(row.get("arm")) for row in mappings}
        != {CONTROL, ENDPOINT}
    ):
        violations.append("E69 Gate 2 R3 MATH replacement grid mismatch")
        return jobs, repairs, violations
    substitutions = []
    for mapping in mappings:
        invalid_job_id = int(mapping["invalid_job_id"])
        matches = [
            index
            for index, row in enumerate(jobs["math_dev"])
            if int(row["job_id"]) == invalid_job_id
        ]
        if len(matches) != 1:
            violations.append(
                f"E69 Gate 2 R3 MATH cell {invalid_job_id} is not unique"
            )
            return jobs, repairs, violations
        original = jobs["math_dev"][matches[0]]
        evidence = excluded[invalid_job_id]
        run_dir = _run_dir(str(original["run_stamp"]), invalid_job_id)
        log_path = ROOT / f"var/artifacts/logs/xdr_train-{invalid_job_id}.out"
        prefix_size = int(evidence.get("log_prefix_size_bytes", -1))
        if (
            run_dir is None
            or original["arm"] != mapping.get("arm")
            or int(original["seed"]) != int(mapping.get("seed", -1))
            or (run_dir / "eval_mode_coverage_draws.jsonl").exists()
            or not log_path.is_file()
            or prefix_size < 1
            or log_path.stat().st_size < prefix_size
            or hashlib.sha256(
                log_path.read_bytes()[:prefix_size]
            ).hexdigest()
            != evidence.get("log_prefix_sha256")
            or evidence.get("evaluation_records_at_freeze") != 0
        ):
            violations.append(
                f"E69 Gate 2 R3 excluded MATH attempt {invalid_job_id} mismatch"
            )
            return jobs, repairs, violations
        replacement = {
            "arm": str(mapping["arm"]),
            "seed": int(mapping["seed"]),
            "job_id": int(mapping["replacement_job_id"]),
            "run_stamp": str(mapping["run_stamp"]),
        }
        jobs["math_dev"][matches[0]] = replacement
        substitutions.append(
            {
                "domain": "math_dev",
                "invalid_job_id": invalid_job_id,
                "replacement_job_id": int(mapping["replacement_job_id"]),
            }
        )
    repairs.append(
        {
            "kind": "implementation_r3_math_eval_split",
            "substitutions": substitutions,
            "identity": str(R3_MATH_EVAL_REPAIR_IDENTITY.resolve()),
            "identity_sha256": _sha256(R3_MATH_EVAL_REPAIR_IDENTITY),
            "attempt_selection": r3["attempt_selection"],
            "outcome_tuning": False,
            "implementation_contract_repair": True,
        }
    )
    return jobs, repairs, violations


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _load_precheckpoint_attempt_repairs() -> tuple[
    dict[int, dict[str, Any]],
    dict[str, Any] | None,
    list[str],
]:
    violations: list[str] = []
    if not PRECHECKPOINT_REQUEUE_REPAIR_IDENTITY.is_file():
        return {}, None, [
            "E69 Gate 2 pre-checkpoint requeue repair identity is absent"
        ]
    try:
        identity = json.loads(
            PRECHECKPOINT_REQUEUE_REPAIR_IDENTITY.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return {}, None, [
            f"E69 Gate 2 pre-checkpoint requeue identity is invalid: {exc}"
        ]
    if (
        identity.get("schema")
        != "e69_gate2_precheckpoint_requeue_attempt_repair_v1"
    ):
        violations.append("E69 pre-checkpoint requeue repair schema drift")
    protocol = ROOT / str(identity.get("protocol", ""))
    if (
        not protocol.is_file()
        or identity.get("protocol_sha256") != _sha256(protocol)
    ):
        violations.append("E69 pre-checkpoint requeue repair protocol drift")
    raw_jobs = identity.get("jobs")
    if not isinstance(raw_jobs, dict) or set(raw_jobs) != {
        "30160592",
        "30160205",
    }:
        violations.append("E69 pre-checkpoint requeue repair job grid drift")
        raw_jobs = {}
    jobs = {
        int(job_id): dict(row)
        for job_id, row in raw_jobs.items()
        if isinstance(row, dict)
    }
    repair = {
        "kind": "precheckpoint_requeue_attempt",
        "identity": str(PRECHECKPOINT_REQUEUE_REPAIR_IDENTITY.resolve()),
        "identity_sha256": _sha256(PRECHECKPOINT_REQUEUE_REPAIR_IDENTITY),
        "selection_rule": identity.get("selection_rule"),
        "jobs": sorted(jobs),
    }
    return jobs, repair, violations


def _step_from_training_row(row: dict[str, Any]) -> int | None:
    value = row.get("trainer/global_step", row.get("misc/global_step"))
    return int(value) if _finite(value) else None


def _validate_attempt_boundary(
    run_dir: Path,
    *,
    job_id: int,
    domain: str,
    arm: str,
    row: dict[str, Any] | None,
    label: str,
) -> tuple[int, list[str]]:
    violations: list[str] = []
    if row is None:
        return 1, violations
    if row.get("domain") != domain or row.get("arm") != arm:
        violations.append(f"{label}: pre-checkpoint repair cell drift")
    path = run_dir / "train_metrics.jsonl"
    if not path.is_file():
        return int(row.get("accepted_start_line", 1)), violations
    raw_lines = path.read_bytes().splitlines(keepends=True)
    start_line = int(row.get("accepted_start_line", 0))
    prefix_last_line = int(row.get("abandoned_prefix_last_line", -1))
    if start_line != prefix_last_line + 1 or start_line < 2:
        violations.append(f"{label}: invalid accepted-attempt boundary")
        return max(start_line, 1), violations
    if len(raw_lines) < start_line:
        violations.append(f"{label}: accepted-attempt start line is absent")
        return start_line, violations
    observed_prefix = hashlib.sha256(
        b"".join(raw_lines[:prefix_last_line])
    ).hexdigest()
    if observed_prefix != row.get("abandoned_prefix_sha256"):
        violations.append(f"{label}: abandoned attempt prefix hash drift")
    try:
        prefix_terminal = json.loads(
            raw_lines[prefix_last_line - 1].decode("utf-8")
        )
        accepted_initial = json.loads(raw_lines[start_line - 1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        violations.append(f"{label}: invalid reset-boundary JSON: {exc}")
    else:
        if _step_from_training_row(prefix_terminal) != int(
            row.get("abandoned_prefix_last_step", -1)
        ):
            violations.append(f"{label}: abandoned prefix terminal step drift")
        if _step_from_training_row(accepted_initial) != int(
            row.get("accepted_start_step", -1)
        ):
            violations.append(f"{label}: accepted attempt did not restart at zero")

    eval_path = run_dir / "eval_mode_coverage_draws.jsonl"
    if not eval_path.is_file():
        violations.append(f"{label}: repeated step-0 evaluation evidence is absent")
    else:
        eval_lines = eval_path.read_bytes().splitlines(keepends=True)
        if len(eval_lines) < 4:
            violations.append(
                f"{label}: repeated step-0 evaluation evidence is incomplete"
            )
        else:
            hashes = [
                hashlib.sha256(b"".join(eval_lines[start:stop])).hexdigest()
                for start, stop in ((0, 2), (2, 4))
            ]
            expected = [
                row.get("abandoned_step0_evaluation_sha256"),
                row.get("accepted_step0_evaluation_sha256"),
            ]
            if hashes != expected or hashes[0] != hashes[1]:
                violations.append(
                    f"{label}: repeated step-0 evaluation identity drift"
                )
    return start_line, violations


def _scan_logs(job_id: int) -> list[str]:
    failures: list[str] = []
    for suffix in ("out", "err"):
        path = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        failures.extend(match.group(0) for match in UNCAUGHT.finditer(text))
    return failures


def _training_audit(
    path: Path,
    *,
    label: str,
    arm: str,
    response_limit: int,
    accepted_start_line: int = 1,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if not path.is_file():
        return {
            "records": 0,
            "latest_step": -1,
            "route_terminal": {},
        }, violations
    records = 0
    latest_step = -1
    control_groups = 0
    control_rows = 0
    control_prompt_tokens = 0
    control_response_tokens = 0
    control_charged_response_tokens = 0
    replay_prompt_tokens = 0
    replay_response_tokens = 0
    replay_charged_response_tokens = 0
    replay_records = 0
    route_terminal: dict[str, float] = {}
    previous_step: int | None = None
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if line_number < accepted_start_line:
            continue
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            violations.append(f"{label}: invalid train JSON line {line_number}")
            continue
        if not isinstance(row, dict):
            violations.append(f"{label}: non-object train line {line_number}")
            continue
        step = _step_from_training_row(row)
        if step is not None:
            if previous_step is not None and step < previous_step:
                violations.append(
                    f"{label}: unregistered optimizer-step regression "
                    f"{previous_step}->{step} at line {line_number}"
                )
            previous_step = step
            latest_step = step
        if not any(key.startswith(("train/", "actor/")) for key in row):
            continue
        records += 1
        for key, value in row.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                violations.append(f"{label}: non-finite {key} line {line_number}")

        expected = {
            "counterfactual_fixed_control_groups_generated": 3.0,
            "counterfactual_fixed_control_rows_generated": 48.0,
            "counterfactual_fixed_control_rows_sent_to_ppo": 0.0,
        }
        for name, target in expected.items():
            value = _metric(row, name)
            if not _close(value, target):
                violations.append(
                    f"{label}: {name}={value!r}, expected {target} "
                    f"at line {line_number}"
                )
        control_groups += int(
            _metric(row, "counterfactual_fixed_control_groups_generated") or 0
        )
        control_rows += int(
            _metric(row, "counterfactual_fixed_control_rows_generated") or 0
        )
        control_prompt_tokens += int(
            _metric(
                row,
                "counterfactual_fixed_control_realized_prompt_tokens",
            )
            or 0
        )
        control_response_tokens += int(
            _metric(
                row,
                "counterfactual_fixed_control_realized_response_tokens",
            )
            or 0
        )
        charged_control = _metric(
            row,
            "counterfactual_fixed_control_charged_response_token_budget",
        )
        if not _close(charged_control, 48 * response_limit):
            violations.append(
                f"{label}: fixed-control charged budget={charged_control!r}, "
                f"expected {48 * response_limit} at line {line_number}"
            )
        control_charged_response_tokens += int(charged_control or 0)

        replay_charged = _metric(
            row,
            "canonical_replay_charged_response_token_budget",
        )
        if not _close(replay_charged, 16 * response_limit):
            violations.append(
                f"{label}: replay charged budget={replay_charged!r}, "
                f"expected {16 * response_limit} at line {line_number}"
            )
        replay_charged_response_tokens += int(replay_charged or 0)
        replay_prompt_tokens += int(
            _metric(row, "canonical_replay_realized_prompt_tokens") or 0
        )
        replay_response_tokens += int(
            _metric(row, "canonical_replay_realized_response_tokens") or 0
        )

        replay_passes = _metric(row, "canonical_replay_score_passes")
        if _finite(replay_passes):
            replay_records += 1
            if not _close(replay_passes, 2.0):
                violations.append(
                    f"{label}: replay score passes={replay_passes!r} "
                    f"at line {line_number}"
                )
            if arm == CONTROL:
                for name in (
                    "canonical_replay_compute_only",
                    "canonical_replay_compute_only_configured",
                ):
                    if not _close(_metric(row, name), 1.0):
                        violations.append(
                            f"{label}: Dr.GRPO {name} is not one "
                            f"at line {line_number}"
                        )
                for name in (
                    "canonical_replay_weighted_loss",
                    "canonical_replay_applied_score_gradient_sum",
                    "canonical_replay_applied_score_gradient_l2",
                ):
                    if not _close(_metric(row, name), 0.0):
                        violations.append(
                            f"{label}: Dr.GRPO {name} is nonzero "
                            f"at line {line_number}"
                        )
        for key, value in row.items():
            if key.endswith(
                (
                    "rows_sent_to_ppo",
                    "gold_support_feedback",
                    "eval_feedback",
                    "desired_mode_count_feedback",
                    "projection_active",
                )
            ) and _finite(value) and not _close(value, 0.0):
                violations.append(
                    f"{label}: forbidden nonzero {key} at line {line_number}"
                )
        for name in (
            "verified_route_post_replay_cross_prompt_neutral_reproductions",
            "verified_route_cross_prompt_replay_updates",
            "verified_route_cross_prompt_replay_groups",
            "verified_route_proposal_graduations",
        ):
            value = _metric(row, name)
            if _finite(value):
                route_terminal[name.removeprefix("verified_route_")] = float(value)

    return (
        {
            "records": records,
            "latest_step": latest_step,
            "accepted_start_line": accepted_start_line,
            "fixed_control_groups": control_groups,
            "fixed_control_rows": control_rows,
            "fixed_control_realized_prompt_tokens": control_prompt_tokens,
            "fixed_control_realized_response_tokens": control_response_tokens,
            "fixed_control_charged_response_tokens": (
                control_charged_response_tokens
            ),
            "replay_records": replay_records,
            "replay_realized_prompt_tokens": replay_prompt_tokens,
            "replay_realized_response_tokens": replay_response_tokens,
            "replay_charged_response_tokens": replay_charged_response_tokens,
            "route_terminal": route_terminal,
        },
        violations,
    )


def _evaluations(path: Path) -> tuple[dict[int, dict[str, float]], list[str]]:
    by_key: dict[tuple[int, str, int | None], dict[str, Any]] = {}
    violations: list[str] = []
    if not path.is_file():
        return {}, violations
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            violations.append(f"{path}: invalid JSON line {line_number}")
            continue
        if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
            continue
        kind = str(row.get("evaluation_kind", ""))
        if kind not in {
            "deterministic_greedy_trace_neutral",
            "fixed_seed_sampled_k_neutral",
        }:
            continue
        step = int(row["step"])
        draw = row.get("draw_index")
        key = (step, kind, draw if isinstance(draw, int) else None)
        prior = by_key.get(key)
        if prior is not None and prior.get("metrics") != row.get("metrics"):
            violations.append(f"{path}: conflicting repeated evaluation {key}")
        by_key[key] = row

    result: dict[int, dict[str, float]] = defaultdict(dict)
    for (step, kind, _draw), row in by_key.items():
        metrics = row["metrics"]
        if kind == "deterministic_greedy_trace_neutral":
            result[step]["greedy"] = float(metrics["any_correct_at_k"])
        else:
            result[step]["mean8"] = float(metrics["mean_at_k"])
            result[step]["pass8"] = float(metrics["any_correct_at_k"])
            result[step]["distinct8"] = float(
                metrics["distinct_correct_modes_at_k"]
            )
    return dict(result), violations


def evaluate_gate(
    curves: dict[str, dict[str, dict[int, dict[str, float]]]],
    route_terminal: dict[str, dict[str, float]],
    route_temporal: dict[str, dict[str, Any]] | None = None,
    *,
    require_python_reproduction: bool = False,
) -> dict[str, Any]:
    """Evaluate only the prospectively frozen E69-minus-control thresholds."""

    checks: dict[str, bool] = {}
    deltas: dict[str, dict[int, dict[str, float]]] = {}
    for domain in POOL_SIZE:
        treatment = ENDPOINT if domain == "math_dev" else SUCCESSOR
        deltas[domain] = {}
        for pass_index in PASSES:
            deltas[domain][pass_index] = {
                metric: (
                    curves[domain][treatment][pass_index][metric]
                    - curves[domain][CONTROL][pass_index][metric]
                )
                for metric in METRICS
            }

    for pass_index in (5, 6):
        checks[f"quality_noninferior_pass_{pass_index}"] = all(
            deltas[domain][pass_index][metric] >= -0.02 - 1e-12
            for domain in POOL_SIZE
            for metric in ("greedy", "pass8")
        )
        checks[f"three_modebench_distinct_gains_pass_{pass_index}"] = (
            sum(
                deltas[domain][pass_index]["distinct8"] > 0
                for domain in MODEBENCH
            )
            >= 3
        )
        checks[f"mathir_pass_and_distinct_gain_pass_{pass_index}"] = (
            deltas["mathir"][pass_index]["pass8"] > 0
            and deltas["mathir"][pass_index]["distinct8"] > 0
        )

    math_delta = deltas["math_dev"][6]
    checks["math_parent_quality_gate"] = (
        math_delta["greedy"] >= -0.01 - 1e-12
        and math_delta["pass8"] >= -0.01 - 1e-12
        and (math_delta["greedy"] > 0 or math_delta["pass8"] > 0)
    )
    checks["math_greedy_mean_gate"] = (
        math_delta["greedy"] >= -0.01 - 1e-12
        and math_delta["mean8"] >= -0.01 - 1e-12
        and (math_delta["greedy"] > 0 or math_delta["mean8"] > 0)
    )
    if route_temporal is None:
        # Backward-compatible pure-function fallback. Production Gate 2 uses
        # the prospective checkpoint observer validated in main().
        route_reproductions = {
            domain: route_terminal.get(domain, {}).get(
                "post_replay_cross_prompt_neutral_reproductions",
                0.0,
            )
            for domain in MODEBENCH
        }
        mechanism_observer = "legacy_in_process_counter"
    else:
        route_reproductions = {
            domain: route_temporal.get(domain, {}).get(
                "post_replay_neutral_reproduction_pairs",
                0,
            )
            for domain in MODEBENCH
        }
        mechanism_observer = "prospective_checkpoint_temporal_lower_bound"
    checks["post_replay_reuse_three_domains"] = (
        sum(value > 0 for value in route_reproductions.values()) >= 3
    )
    if require_python_reproduction:
        checks["python_post_replay_neutral_reproduction"] = (
            route_reproductions["python_factor"] > 0
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "mechanism_observer": mechanism_observer,
        "route_reproductions": route_reproductions,
        "deltas": {
            domain: {str(key): value for key, value in by_pass.items()}
            for domain, by_pass in deltas.items()
        },
    }


def _load_route_temporal_snapshots() -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    r1_active = R1_EXECUTION_REPAIR_IDENTITY.is_file()
    r2_active = R2_IMPLEMENTATION_REPAIR_IDENTITY.is_file()
    amendment = (
        R2_IMPLEMENTATION_REPAIR_PROTOCOL
        if r2_active
        else (
            R1_EXECUTION_REPAIR_PROTOCOL
            if r1_active
            else ROUTE_TEMPORAL_AMENDMENT
        )
    )
    snapshots_path = (
        R2_ROUTE_TEMPORAL_SNAPSHOTS
        if r2_active
        else (
            R1_ROUTE_TEMPORAL_SNAPSHOTS
            if r1_active
            else ROUTE_TEMPORAL_SNAPSHOTS
        )
    )
    if not amendment.is_file():
        return {}, ["E69 route temporal amendment is absent"]
    if not snapshots_path.is_file():
        return {}, ["E69 route temporal snapshot artifact is absent"]
    try:
        payload = json.loads(snapshots_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"E69 route temporal snapshot artifact is invalid: {exc}"]
    expected_schema = (
        "e69_gate2_r2_route_temporal_snapshots_v1"
        if r2_active
        else (
            "e69_gate2_r1_route_temporal_snapshots_v1"
            if r1_active
            else "e69_gate2_route_temporal_snapshots_v1"
        )
    )
    if payload.get("schema") != expected_schema:
        violations.append("E69 route temporal snapshot schema drift")
    amendment_hash = hashlib.sha256(amendment.read_bytes()).hexdigest()
    if payload.get("amendment_sha256") != amendment_hash:
        violations.append("E69 route temporal amendment hash drift")
    if r2_active:
        r2 = json.loads(
            R2_IMPLEMENTATION_REPAIR_IDENTITY.read_text(encoding="utf-8")
        )
        if (
            payload.get("r2_identity_sha256")
            != _sha256(R2_IMPLEMENTATION_REPAIR_IDENTITY)
            or payload.get("parent_snapshot_sha256")
            != r2.get("parent_route_temporal_snapshot_sha256")
            or payload.get("carried_parent_domains")
            != ["countdown", "graph_coloring", "mathir"]
            or payload.get("fresh_r2_domains") != ["python_factor"]
        ):
            violations.append("E69-R2 route temporal provenance drift")
    elif r1_active:
        r1 = json.loads(
            R1_EXECUTION_REPAIR_IDENTITY.read_text(encoding="utf-8")
        )
        if (
            payload.get("r1_identity_sha256")
            != _sha256(R1_EXECUTION_REPAIR_IDENTITY)
            or payload.get("parent_snapshot_sha256")
            != r1.get("parent_route_temporal_snapshot_sha256")
            or payload.get("carried_parent_domains")
            != ["countdown", "mathir"]
            or payload.get("fresh_r1_domains")
            != ["graph_coloring", "python_factor"]
        ):
            violations.append("E69-R1 route temporal provenance drift")
    snapshots = payload.get("snapshots")
    temporal = payload.get("temporal_reproductions")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MODEBENCH):
        violations.append("E69 route temporal snapshot domain grid drift")
        snapshots = {}
    if not isinstance(temporal, dict) or set(temporal) != set(MODEBENCH):
        violations.append("E69 route temporal summary domain grid drift")
        temporal = {}
    for domain in MODEBENCH:
        expected_steps = {
            str(POOL_SIZE[domain] * pass_index) for pass_index in range(1, 7)
        }
        observed = snapshots.get(domain, {})
        if not isinstance(observed, dict) or set(observed) != expected_steps:
            violations.append(
                f"{domain}: route temporal observer lacks six exact checkpoints"
            )
        row = temporal.get(domain, {})
        if (
            not isinstance(row, dict)
            or int(row.get("snapshot_count", -1)) != 6
            or int(row.get("post_replay_neutral_reproduction_pairs", -1)) < 0
        ):
            violations.append(f"{domain}: route temporal summary is invalid")
    return payload, violations


def _write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# E69 Gate 2 compute-matched screen",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "MATH-500 remains sealed. MATH E66/E68/E69 are one physical "
        "endpoint-only run with three reporting aliases.",
        "",
        "| Domain | Arm | Pass | Greedy | Mean@8 | Pass@8 | Distinct@8 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for domain, arms in payload.get("curves", {}).items():
        for arm, by_pass in arms.items():
            for pass_index in PASSES:
                row = by_pass.get(str(pass_index))
                if not row:
                    continue
                lines.append(
                    f"| {domain} | {arm} | {pass_index} | "
                    f"{row['greedy']:.4f} | {row['mean8']:.4f} | "
                    f"{row['pass8']:.4f} | {row['distinct8']:.4f} |"
                )
    if payload.get("outcome_gate"):
        lines.extend(["", "## Frozen gate checks", ""])
        for name, passed in payload["outcome_gate"]["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    if payload.get("repairs"):
        lines.extend(["", "## Pre-optimizer infrastructure repair", ""])
        for repair in payload["repairs"]:
            if repair["kind"] == "preoptimizer_startup":
                lines.append(
                    f"- Excluded job {repair['invalid_job_id']} and used exact "
                    f"replacement job {repair['replacement_job_id']} under the "
                    "prospectively recorded startup-repair identity."
                )
            elif repair["kind"] == "precheckpoint_requeue_attempt":
                lines.append(
                    "- Excluded the exact checkpoint-free attempt prefixes "
                    f"for jobs {', '.join(map(str, repair['jobs']))}; the "
                    "accepted attempts restart from initialization under the "
                    "prospectively recorded attempt-repair identity."
                )
            elif repair["kind"] == "execution_r1":
                lines.append(
                    f"- Replaced {len(repair['substitutions'])} Graph/Python "
                    "cells under the frozen E69-R1 execution-only identity; "
                    "terminal outcomes were disclosed, outcome tuning is "
                    "forbidden, and excluded traces are not spliced."
                )
            elif repair["kind"] == "implementation_r2":
                lines.append(
                    "- Replaced the failed R1 Python successor from "
                    "initialization under the prospective E69-R2 "
                    "route/endpoint bookkeeping repair; the hard exception "
                    "is preserved, no repaired outcome preceded the freeze, "
                    "and no failed prefix is spliced."
                )
            elif repair["kind"] == "implementation_r3_math_eval_split":
                substitutions = repair["substitutions"]
                old_jobs = ", ".join(
                    str(row["invalid_job_id"]) for row in substitutions
                )
                new_jobs = ", ".join(
                    str(row["replacement_job_id"]) for row in substitutions
                )
                lines.append(
                    "- Excluded MATH-dev jobs "
                    f"{old_jobs} and restarted the paired cohort as jobs "
                    f"{new_jobs} under the prospective E69-R3 evaluation-split "
                    "contract repair; both arms restart from initialization, "
                    "outcome tuning is forbidden, and the excluded prefixes "
                    "remain archived."
                )
            else:
                lines.append(
                    f"- Replaced {len(repair['substitutions'])} never-started "
                    "pending executable-domain jobs under the prospectively "
                    "recorded matched-placement identity."
                )
    if payload.get("pending"):
        lines.extend(["", "## Pending", ""])
        lines.extend(f"- {value}" for value in payload["pending"])
    if payload.get("violations"):
        lines.extend(["", "## Integrity violations", ""])
        lines.extend(f"- {value}" for value in payload["violations"])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not IDENTITY.is_file():
        raise SystemExit(f"E69 Gate 2 identity is absent: {IDENTITY}")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    if identity.get("schema") != "e69_gate2_compute_matched_screen_v1":
        raise SystemExit("unexpected E69 Gate 2 identity schema")

    violations: list[str] = []
    pending: list[str] = []
    physical_runs: list[dict[str, Any]] = []
    curves: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(dict)
    route_terminal: dict[str, dict[str, float]] = {}
    effective_jobs, repairs, repair_violations = _effective_jobs(identity)
    violations.extend(repair_violations)
    attempt_jobs, attempt_repair, attempt_violations = (
        _load_precheckpoint_attempt_repairs()
    )
    violations.extend(attempt_violations)
    if attempt_repair is not None:
        repairs.append(attempt_repair)
    for domain, jobs in effective_jobs.items():
        expected_step = POOL_SIZE[domain] * 6
        response_limit = 1024 if domain == "math_dev" else (
            64 if domain == "mathir" else 192
        )
        for job in jobs:
            arm = str(job["arm"])
            job_id = int(job["job_id"])
            label = f"{domain}/{arm}/job{job_id}"
            run_dir = _run_dir(str(job["run_stamp"]), job_id)
            if run_dir is None:
                pending.append(f"{label}: run directory absent")
                physical_runs.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "seed": int(job["seed"]),
                        "job_id": job_id,
                        "run_stamp": str(job["run_stamp"]),
                        "run_dir": None,
                        "terminal": False,
                    }
                )
                violations.extend(
                    f"{label}: {failure}" for failure in _scan_logs(job_id)
                )
                continue
            accepted_start_line, boundary_violations = (
                _validate_attempt_boundary(
                    run_dir,
                    job_id=job_id,
                    domain=domain,
                    arm=arm,
                    row=attempt_jobs.get(job_id),
                    label=label,
                )
            )
            violations.extend(boundary_violations)
            training, run_violations = _training_audit(
                run_dir / "train_metrics.jsonl",
                label=label,
                arm=arm,
                response_limit=response_limit,
                accepted_start_line=accepted_start_line,
            )
            violations.extend(run_violations)
            violations.extend(
                f"{label}: {failure}" for failure in _scan_logs(job_id)
            )
            evaluations, eval_violations = _evaluations(
                run_dir / "eval_mode_coverage_draws.jsonl"
            )
            violations.extend(f"{label}: {value}" for value in eval_violations)
            pass_curve: dict[int, dict[str, float]] = {}
            for pass_index in PASSES:
                step = POOL_SIZE[domain] * pass_index
                row = evaluations.get(step)
                if row is None or not all(metric in row for metric in METRICS):
                    pending.append(f"{label}: missing evaluation pass {pass_index}")
                else:
                    pass_curve[pass_index] = {
                        metric: float(row[metric]) for metric in METRICS
                    }
            curves[domain][arm] = pass_curve
            terminal = (
                int(training["latest_step"]) >= expected_step
                and len(pass_curve) == len(PASSES)
            )
            if not terminal:
                pending.append(
                    f"{label}: latest step {training['latest_step']}/{expected_step}"
                )
            if arm == SUCCESSOR:
                route_terminal[domain] = dict(training["route_terminal"])
            physical_runs.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "seed": int(job["seed"]),
                    "job_id": job_id,
                    "run_stamp": str(job["run_stamp"]),
                    "run_dir": str(run_dir.resolve()),
                    "terminal": terminal,
                    "training": training,
                    "evaluations": {
                        str(key): value for key, value in pass_curve.items()
                    },
                }
            )

    if "math_dev" in curves and ENDPOINT in curves["math_dev"]:
        curves["math_dev"][SUCCESSOR] = curves["math_dev"][ENDPOINT]
        curves["math_dev"]["verified_entropy_gated_singleton_escape_canonical"] = (
            curves["math_dev"][ENDPOINT]
        )

    complete = (
        len(physical_runs) == 18
        and all(run.get("terminal") for run in physical_runs)
        and not pending
    )
    outcome_gate = None
    route_temporal: dict[str, Any] = {}
    if complete and not violations:
        route_temporal, temporal_violations = _load_route_temporal_snapshots()
        violations.extend(temporal_violations)
        if not violations:
            outcome_gate = evaluate_gate(
                curves,
                route_terminal,
                route_temporal["temporal_reproductions"],
                require_python_reproduction=(
                    R1_EXECUTION_REPAIR_IDENTITY.is_file()
                    or R2_IMPLEMENTATION_REPAIR_IDENTITY.is_file()
                ),
            )
    status = (
        "fail"
        if violations
        else outcome_gate["status"]
        if outcome_gate is not None
        else "in_progress"
    )
    payload = {
        "schema": "e69_gate2_compute_matched_screen_audit_v1",
        "status": status,
        "identity": str(IDENTITY.resolve()),
        "math500_sealed": True,
        "summary": {
            "expected_physical_runs": 18,
            "observed_physical_runs": len(physical_runs),
            "terminal_physical_runs": sum(
                bool(run.get("terminal")) for run in physical_runs
            ),
            "pending_items": len(set(pending)),
            "integrity_violations": len(set(violations)),
        },
        "physical_runs": physical_runs,
        "curves": {
            domain: {
                arm: {str(key): value for key, value in by_pass.items()}
                for arm, by_pass in arms.items()
            }
            for domain, arms in curves.items()
        },
        "route_terminal": route_terminal,
        "route_temporal": route_temporal,
        "repairs": repairs,
        "outcome_gate": outcome_gate,
        "pending": sorted(set(pending)),
        "violations": sorted(set(violations)),
    }
    _atomic_json(OUT_JSON, payload)
    _write_markdown(payload)
    print(
        f"[e69-gate2-audit] status={status} "
        f"terminal={payload['summary']['terminal_physical_runs']}/18 "
        f"pending={payload['summary']['pending_items']} "
        f"violations={payload['summary']['integrity_violations']}"
    )


if __name__ == "__main__":
    main()
