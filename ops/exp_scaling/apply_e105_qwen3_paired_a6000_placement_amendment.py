#!/usr/bin/env python3
"""Move untouched matched E80-R1 pairs only after E105's mechanism gate passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e80r1_qwen3b_aligned_verified_replay as e80  # noqa: E402
import launch_e105_group_centered_semantic_repair_full_three_scale as e105  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e105_qwen3_paired_a6000_placement_amendment_20260817.md"
)
E80_LEDGER = ROOT / e105.COMPARATOR_LEDGERS["qwen3b"]
E105_LEDGER = ROOT / e105.LEDGER
OUT = ROOT / "var/artifacts/e105_qwen3_paired_a6000_placement_amendment.json"
PROSPECTIVE_PYTHON_CELLS = (
    ("python_factors", 73),
    ("python_factors", 74),
)
HISTORICAL_CANDIDATE_CELLS = (
    ("mathir", 71),
    ("mathir", 72),
    ("mathir", 73),
    ("mathir", 74),
    ("pantry_plan", 71),
    ("pantry_plan", 72),
    ("pantry_plan", 73),
    ("pantry_plan", 74),
)
CANDIDATE_CELLS = PROSPECTIVE_PYTHON_CELLS + HISTORICAL_CANDIDATE_CELLS
ARMS = ("control", "replay")
ORIGINAL_PARTITION = "mltheory"
ORIGINAL_NODE_LIST = "node302"
ORIGINAL_GRES = "gres/gpu:a100:1"
TARGET_PARTITION = "lowprio"
TARGET_NODE_LIST = "node[103-104,205-208,805]"
TARGET_GRES = "gres/gpu:a6000:1"
ACCOUNT = "mltheory"
TIME_LIMIT = "3-00:00:00"


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scheduler_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"cannot inspect E80-R1 job {job_id}: {detail}")
    return result.stdout.strip()


def require(record: str, job_id: int, needles: tuple[str, ...]) -> None:
    missing = [needle for needle in needles if needle not in record]
    if missing:
        raise RuntimeError(f"E80-R1 job {job_id} lacks scheduler fields {missing}")


def candidate_pairs(
    ledger: dict[str, Any],
) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    if ledger.get("schema") != "e80r1_qwen3b_aligned_verified_replay_jobs_v1":
        raise RuntimeError("unexpected E80-R1 ledger schema")
    if ledger.get("released") is not True:
        raise RuntimeError("E80-R1 was not durably released")
    if tuple(ledger.get("domains", [])) != tuple(e105.DOMAINS):
        raise RuntimeError("E80-R1 domains drifted")
    if any("pointmaze" in str(domain).lower() for domain in ledger["domains"]):
        raise RuntimeError("E80-R1 candidate ledger contains PointMaze")
    index = {
        (str(run["domain"]), int(run["seed"]), str(run["arm"])): run
        for run in ledger.get("runs", [])
    }
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for domain, seed in HISTORICAL_CANDIDATE_CELLS:
        try:
            pairs[(domain, seed)] = {
                arm: index[(domain, seed, arm)] for arm in ARMS
            }
        except KeyError as exc:
            raise RuntimeError(
                f"E80-R1 lacks candidate pair {domain}/s{seed}"
            ) from exc
    job_ids = [
        int(run["job_id"])
        for pair in pairs.values()
        for run in pair.values()
    ]
    if len(job_ids) != 16 or len(set(job_ids)) != 16:
        raise RuntimeError("E80-R1 historical candidate set is not sixteen unique jobs")
    return pairs


def scientific_needles(
    ledger: dict[str, Any], run: dict[str, Any]
) -> tuple[str, ...]:
    arm = str(run["arm"])
    return (
        f"Account={ACCOUNT}",
        f"TimeLimit={TIME_LIMIT}",
        "NumCPUs=16",
        "MinMemoryNode=128G",
        f"OAT_ZERO_SOURCE_ROOT={Path(str(ledger['snapshot_root'])) / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={Path(str(ledger['snapshot_root'])) / 'ops'}",
        f"OAT_ZERO_VARIANT={e80.VARIANTS[arm]}",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_MAX_TRAIN=384",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_AUTO_RESUME=1",
        "OAT_ZERO_WATCHDOG_REQUEUE=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        f"OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY={int(arm == 'control')}",
    )


def validate_original(
    ledger: dict[str, Any], run: dict[str, Any], record: str
) -> None:
    job_id = int(run["job_id"])
    require(
        record,
        job_id,
        scientific_needles(ledger, run)
        + (
            f"Partition={ORIGINAL_PARTITION}",
            f"ReqNodeList={ORIGINAL_NODE_LIST}",
            f"TresPerNode={ORIGINAL_GRES}",
        ),
    )


def validate_recovered_amended(
    ledger: dict[str, Any], run: dict[str, Any], record: str
) -> None:
    """Validate placement after an interrupted apply, even if work has started."""

    job_id = int(run["job_id"])
    require(
        record,
        job_id,
        scientific_needles(ledger, run)
        + (
            f"Partition={TARGET_PARTITION}",
            f"ReqNodeList={TARGET_NODE_LIST}",
            f"TresPerNode={TARGET_GRES}",
        ),
    )


def pending_at_zero(record: str) -> bool:
    return "JobState=PENDING" in record and "RunTime=00:00:00" in record


def validate_amended(
    ledger: dict[str, Any], run: dict[str, Any], record: str
) -> None:
    job_id = int(run["job_id"])
    require(
        record,
        job_id,
        scientific_needles(ledger, run)
        + (
            "JobState=PENDING",
            "RunTime=00:00:00",
            f"Partition={TARGET_PARTITION}",
            f"ReqNodeList={TARGET_NODE_LIST}",
            f"TresPerNode={TARGET_GRES}",
        ),
    )


def update_command(job_id: int, *, amended: bool) -> list[str]:
    if amended:
        partition, nodes, gres = TARGET_PARTITION, TARGET_NODE_LIST, "gpu:a6000:1"
    else:
        partition, nodes, gres = ORIGINAL_PARTITION, ORIGINAL_NODE_LIST, "gpu:a100:1"
    return [
        "scontrol",
        "update",
        f"JobId={job_id}",
        f"Partition={partition}",
        f"Account={ACCOUNT}",
        f"NodeList={nodes}",
        f"Gres={gres}",
        f"TimeLimit={TIME_LIMIT}",
    ]


def execute(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"scheduler update failed: {detail}")


def state_summary(record: str) -> dict[str, str]:
    def value(key: str) -> str:
        marker = f"{key}="
        for token in record.split():
            if token.startswith(marker):
                return token.removeprefix(marker)
        return "UNKNOWN"

    return {"state": value("JobState"), "runtime": value("RunTime")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"placement protocol is absent: {PROTOCOL}")
    if E105_LEDGER.exists():
        raise SystemExit("E105 already exists; placement must be frozen before submission")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate placement amendment: {OUT}")

    # This freshly revalidates the complete, outcome-blind mechanism gate and
    # all snapshot-bound evidence, including the real Qwen-3B A6000 smoke.
    _snapshot, gate_path, gate, _e104_audit = e105.check_gate(ROOT)
    if set(gate.get("nonzero_semantic_scales", [])) != set(e105.SCALE_SEEDS):
        raise SystemExit("combined gate lacks a nonzero semantic update at every scale")

    ledger = load(E80_LEDGER)
    pairs = candidate_pairs(ledger)
    current = {
        int(run["job_id"]): scheduler_record(int(run["job_id"]))
        for pair in pairs.values()
        for run in pair.values()
    }
    before = dict(current)
    eligible: list[tuple[str, int]] = []
    needs_update: list[tuple[str, int]] = []
    recovered: list[tuple[str, int]] = []
    skipped: list[dict[str, Any]] = []
    for cell, pair in pairs.items():
        records = {arm: current[int(run["job_id"])] for arm, run in pair.items()}
        original = all(
            all(needle in records[arm] for needle in scientific_needles(ledger, run))
            and f"Partition={ORIGINAL_PARTITION}" in records[arm]
            and f"ReqNodeList={ORIGINAL_NODE_LIST}" in records[arm]
            and f"TresPerNode={ORIGINAL_GRES}" in records[arm]
            for arm, run in pair.items()
        )
        amended = all(
            all(needle in records[arm] for needle in scientific_needles(ledger, run))
            and f"Partition={TARGET_PARTITION}" in records[arm]
            and f"ReqNodeList={TARGET_NODE_LIST}" in records[arm]
            and f"TresPerNode={TARGET_GRES}" in records[arm]
            for arm, run in pair.items()
        )
        if amended:
            for arm, run in pair.items():
                validate_recovered_amended(ledger, run, records[arm])
                original_record = str(run.get("held_scheduler_record", ""))
                validate_original(ledger, run, original_record)
                before[int(run["job_id"])] = original_record
            eligible.append(cell)
            recovered.append(cell)
        elif original:
            for arm, run in pair.items():
                validate_original(ledger, run, records[arm])
            if all(pending_at_zero(records[arm]) for arm in ARMS):
                eligible.append(cell)
                needs_update.append(cell)
            else:
                skipped.append(
                    {
                        "domain": cell[0],
                        "seed": cell[1],
                        "jobs": {
                            arm: {
                                "job_id": int(run["job_id"]),
                                **state_summary(records[arm]),
                            }
                            for arm, run in pair.items()
                        },
                    }
                )
        else:
            raise RuntimeError(f"E80-R1 pair has mixed or unknown placement: {cell}")

    commands = [
        update_command(int(pairs[cell][arm]["job_id"]), amended=True)
        for cell in needs_update
        for arm in ARMS
    ]
    if not args.apply:
        for command in commands:
            print(" ".join(shlex.quote(token) for token in command))
        print(f"[e105-q3-paired-placement] eligible_pairs={len(eligible)} skipped={len(skipped)}")
        return 0

    changed: list[int] = []
    after = {
        int(run["job_id"]): current[int(run["job_id"])]
        for cell in recovered
        for run in pairs[cell].values()
    }
    try:
        for command in commands:
            execute(command)
            changed.append(int(command[2].split("=", 1)[1]))
        after.update({job_id: scheduler_record(job_id) for job_id in changed})
        for cell in needs_update:
            for run in pairs[cell].values():
                validate_amended(ledger, run, after[int(run["job_id"])])
    except Exception:
        for job_id in changed:
            execute(update_command(job_id, amended=False))
        raise

    moved_pairs = [
        {
            "domain": domain,
            "seed": seed,
            "control_job_id": int(pairs[(domain, seed)]["control"]["job_id"]),
            "replay_job_id": int(pairs[(domain, seed)]["replay"]["job_id"]),
        }
        for domain, seed in eligible
    ]
    paired_a6000_cells = [
        {"domain": domain, "seed": seed}
        for domain, seed in (*PROSPECTIVE_PYTHON_CELLS, *eligible)
    ]
    payload = {
        "schema": "e105_qwen3_paired_a6000_placement_amendment_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": e105.e104.digest(PROTOCOL),
        "script": str(Path(__file__).resolve()),
        "script_sha256": e105.e104.digest(Path(__file__)),
        "e80r1_ledger": str(E80_LEDGER),
        "e80r1_ledger_sha256": e105.e104.digest(E80_LEDGER),
        "combined_gate": str(gate_path),
        "combined_gate_sha256": e105.e104.digest(gate_path),
        "combined_gate_complete": True,
        "combined_gate_passed": True,
        "outcome_metrics_inspected": False,
        "scientific_configuration_changed": False,
        "pointmaze": "excluded",
        "candidate_cells": [
            {"domain": domain, "seed": seed} for domain, seed in CANDIDATE_CELLS
        ],
        "historical_candidate_cells": [
            {"domain": domain, "seed": seed}
            for domain, seed in HISTORICAL_CANDIDATE_CELLS
        ],
        "prospective_python_cells": [
            {"domain": domain, "seed": seed}
            for domain, seed in PROSPECTIVE_PYTHON_CELLS
        ],
        "recovered_after_interrupted_apply": bool(recovered),
        "recovered_moved_pairs": [
            {"domain": domain, "seed": seed}
            for domain, seed in recovered
        ],
        "paired_a6000_cells": paired_a6000_cells,
        "moved_pairs": moved_pairs,
        "skipped_pairs": skipped,
        "placement": {
            "partition": TARGET_PARTITION,
            "account": ACCOUNT,
            "node_list": TARGET_NODE_LIST,
            "gres": "gpu:a6000:1",
            "cpus": 16,
            "memory": "128G",
            "time_limit": TIME_LIMIT,
        },
        "before": {str(job_id): record for job_id, record in before.items()},
        "after": {str(job_id): record for job_id, record in after.items()},
    }
    e80.atomic_json(OUT, payload)
    print(
        f"[e105-q3-paired-placement] moved_pairs={len(moved_pairs)} "
        f"skipped={len(skipped)} artifact={OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
