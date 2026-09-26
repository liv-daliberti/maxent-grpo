#!/usr/bin/env python3
"""Audit the effective Falcon R1 plus Qwen R1-M1 DAPO gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e113r1_dapo_recovery_smokes as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
R1_LEDGER = ROOT / "var/artifacts/e113r1_dapo_recovery_smoke_jobs.json"
M1_LEDGER = ROOT / "var/artifacts/e113r1m1_qwen_memory_recovery_jobs.json"
S1_PROTOCOL = (
    ROOT / "paper/preregistration/e113r1s1_falcon_postreceipt_shutdown_20260819.md"
)
FALCON_LOG = (
    ROOT / "var/artifacts/logs/e113r1-f1-graph-dapo-smoke-30790112.out"
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def accounting(job_id: int) -> tuple[str, str]:
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=State,ExitCode",
            "-P",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return "UNKNOWN", "UNKNOWN"
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    parts = line.split("|")
    if len(parts) < 2:
        return "UNKNOWN", "UNKNOWN"
    return parts[0].split()[0], parts[1]


def falcon_postreceipt_shutdown_is_classified(
    audit_row: dict[str, Any],
    job_id: int,
) -> tuple[bool, list[str]]:
    violations: list[str] = []
    if not S1_PROTOCOL.is_file():
        violations.append(f"missing Falcon shutdown protocol: {S1_PROTOCOL}")
    if job_id != 30790112:
        violations.append(f"Falcon R1 job drifted to {job_id}")
    if audit_row.get("passed") is not True:
        violations.extend(audit_row.get("violations", []))
    if audit_row.get("accepted_updates") != 32:
        violations.append("Falcon does not have 32 unique accepted updates")
    if audit_row.get("raw_accepted_records") != 33:
        violations.append("Falcon raw terminal record count is not 33")
    if audit_row.get("duplicate_accepted_records") != 1:
        violations.append("Falcon duplicate terminal record count is not one")
    if audit_row.get("sampled_rows") != 704.0:
        violations.append("Falcon cumulative query count drifted from 704")
    log = FALCON_LOG.read_text(encoding="utf-8", errors="replace")
    completed = log.find("eval/log done step=32")
    allocator = log.find("Trying to free a pointer not allocated here")
    fatal = log.find("Fatal Python error: Aborted")
    if not (0 <= completed < allocator < fatal):
        violations.append("Falcon completion-to-shutdown log ordering drifted")
    state, exit_code = accounting(job_id)
    if state != "FAILED" or exit_code != "137:0":
        violations.append(
            f"Falcon accounting is {state}/{exit_code}, expected FAILED/137:0"
        )
    return not violations, violations


def audit(r1_path: Path = R1_LEDGER, m1_path: Path = M1_LEDGER) -> dict[str, Any]:
    violations: list[str] = []
    if not r1_path.is_file():
        violations.append(f"missing original R1 ledger: {r1_path}")
    if not m1_path.is_file():
        violations.append(f"missing Qwen M1 ledger: {m1_path}")
    if violations:
        return {
            "schema": "e113r1m1_dapo_effective_gate_audit_v1",
            "passed": False,
            "violations": violations,
            "smokes": [],
        }

    r1 = load(r1_path)
    m1 = load(m1_path)
    if r1.get("schema") != base.EXPECTED_SCHEMA or r1.get("released") is not True:
        violations.append("original R1 ledger is not the released frozen record")
    if m1.get("schema") != "e113r1m1_qwen_memory_recovery_jobs_v1":
        violations.append("M1 ledger schema drifted")
    if m1.get("released") is not True:
        violations.append("M1 ledger is not released")
    if m1.get("scientific_cells") != 0 or m1.get("runs") != []:
        violations.append("M1 ledger contains scientific cells")
    if m1.get("smoke_domain") != "graph_coloring":
        violations.append("M1 domain drifted")
    if m1.get("capacity_repair") != {
        "partition": "lowprio",
        "account": "mltheory",
        "nodelist": "node[103-104,205-208,805]",
        "gres": "gpu:a6000:1",
        "optimizer_offload": False,
        "activation_offload": False,
    }:
        violations.append("M1 capacity-only repair drifted")

    try:
        # The runner writes its completion receipt after emitting a duplicate
        # terminal summary, so a 32-update trajectory has outer terminal step
        # 33. The base auditor still requires the exact unique policy-step set
        # 1..32 and reports the duplicate count separately.
        falcon = base.audit_smoke(
            "falcon1b",
            r1["smokes"]["falcon1b"],
            expected_receipt_step=33,
        )
        qwen = base.audit_smoke(
            "qwen05b",
            m1["smokes"]["qwen05b"],
            expected_receipt_step=33,
        )
    except (KeyError, TypeError) as error:
        violations.append(f"effective smoke mapping is incomplete: {error}")
        smokes: list[dict[str, Any]] = []
    else:
        smokes = [falcon, qwen]
        falcon_ok, falcon_violations = falcon_postreceipt_shutdown_is_classified(
            falcon,
            int(r1["smokes"]["falcon1b"]["job_id"]),
        )
        if not falcon_ok:
            violations.extend(falcon_violations)
        qwen_state, qwen_exit = accounting(int(m1["smokes"]["qwen05b"]["job_id"]))
        if qwen.get("passed") is True and (
            qwen_state != "COMPLETED" or qwen_exit != "0:0"
        ):
            violations.append(
                f"Qwen M1 accounting is {qwen_state}/{qwen_exit}, "
                "expected COMPLETED/0:0"
            )
    return {
        "schema": "e113r1m1_dapo_effective_gate_audit_v1",
        "r1_ledger": str(r1_path),
        "m1_ledger": str(m1_path),
        "passed": not violations
        and len(smokes) == 2
        and all(smoke["passed"] for smoke in smokes),
        "violations": violations,
        "smokes": smokes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-ledger", type=Path, default=R1_LEDGER)
    parser.add_argument("--m1-ledger", type=Path, default=M1_LEDGER)
    args = parser.parse_args()
    report = audit(args.r1_ledger.resolve(), args.m1_ledger.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
