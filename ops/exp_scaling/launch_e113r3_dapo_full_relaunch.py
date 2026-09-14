#!/usr/bin/env python3
"""Gate and launch the fresh 50-cell E113-R3 DAPO comparative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e113r1m1_dapo_effective_gate as gate_audit  # noqa: E402
import launch_e113_dapo_direct_baseline as e113  # noqa: E402
import launch_e113r1m1_qwen_memory_recovery as m1  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402


COHORT = "e113r3"
LEDGER = "var/artifacts/e113r3_dapo_full_relaunch_jobs.json"
PROTOCOL = "paper/preregistration/e113r3_dapo_full_relaunch_20260819.md"
R1_LEDGER = "var/artifacts/e113r1_dapo_recovery_smoke_jobs.json"
M1_LEDGER = m1.LEDGER
M1_S2_LEDGER = "var/artifacts/e113r1m1s2_qwen_backfill_amendment.json"
M1_S3_LEDGER = "var/artifacts/e113r1m1s3_qwen_partition_amendment.json"
S1_PROTOCOL = "paper/preregistration/e113r1s1_falcon_postreceipt_shutdown_20260819.md"
ORIGINAL_LEDGER = e113.LEDGER
CLOSED_R2_PROTOCOL = "paper/preregistration/e113r2_dapo_full_relaunch_20260819.md"
R3_S1_PROTOCOL = "paper/preregistration/e113r3s1_qwen_partition_20260819.md"
R3_S2_PROTOCOL = "paper/preregistration/e113r3s2_held_partition_normalization_20260819.md"
SCIENTIFIC_CELLS = 50
QWEN_PARTITION = "all"
CANCELED_VALIDATION_PROBE = 30790897


def run_stamp(family: str, domain: str, seed: int) -> str:
    return f"e113r3_{family}_{e113.e78.DOMAIN_TAGS[domain]}_dapo_s{seed}"


def job_name(family: str, domain: str, seed: int) -> str:
    family_tag = "q05" if family == "qwen05b" else "f1"
    return f"e113r3-{family_tag}-{e113.e78.DOMAIN_TAGS[domain][:5]}-s{seed}"


def target_path(root: Path, family: str, domain: str, seed: int) -> Path:
    return root / "var/data" / (
        f"xdr_{e113.MODEL_TAGS[family]}_dapo_{run_stamp(family, domain, seed)}"
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_original(root: Path) -> dict[str, Any]:
    path = root / ORIGINAL_LEDGER
    payload = load_json(path)
    if payload.get("schema") != "e113_dapo_direct_baseline_jobs_v1":
        raise SystemExit("original E113 ledger schema drifted")
    if payload.get("released") is not True or len(payload.get("runs", [])) != 50:
        raise SystemExit("original E113 ledger is not the released 50-cell record")
    return payload


def validate_m1_ledger(root: Path, original: dict[str, Any]) -> dict[str, Any]:
    path = root / M1_LEDGER
    payload = load_json(path)
    expected = {
        "schema": "e113r1m1_qwen_memory_recovery_jobs_v1",
        "released": True,
        "scientific_cells": 0,
        "target_steps": 32,
        "passes": 1,
        "smoke_domain": "graph_coloring",
        "r1_ledger": str(root / R1_LEDGER),
        "r1_ledger_sha256": e113.e78.digest(root / R1_LEDGER),
        "snapshot_root": original["snapshot_root"],
        "snapshot_identity": original["snapshot_identity"],
        "objective": e113.objective(),
        "runs": [],
        "capacity_repair": {
            "partition": m1.PARTITION,
            "account": m1.ACCOUNT,
            "nodelist": m1.NODELIST,
            "gres": m1.GRES,
            "optimizer_offload": False,
            "activation_offload": False,
        },
    }
    drift = [key for key, value in expected.items() if payload.get(key) != value]
    smokes = payload.get("smokes")
    if not isinstance(smokes, dict) or set(smokes) != {"qwen05b"}:
        drift.append("smokes")
    protocol = root / m1.PROTOCOL
    launcher = root / "ops/exp_scaling/launch_e113r1m1_qwen_memory_recovery.py"
    if payload.get("protocol_sha256") != e113.e78.digest(protocol):
        drift.append("protocol_sha256")
    if payload.get("launcher_sha256") != e113.e78.digest(launcher):
        drift.append("launcher_sha256")
    if drift:
        raise SystemExit("E113-R1-M1 ledger provenance drifted: " + ", ".join(drift))
    return payload


def validate_scheduler_amendments(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    m1_path = root / M1_LEDGER
    s2_path = root / M1_S2_LEDGER
    s3_path = root / M1_S3_LEDGER
    s2 = load_json(s2_path)
    s3 = load_json(s3_path)
    expected_s2 = {
        "schema": "e113r1m1s2_qwen_backfill_amendment_v1",
        "installed": True,
        "outcomes_inspected": False,
        "job_id": 30790590,
        "same_job_id": True,
        "same_scientific_cells": True,
        "scientific_cells": 0,
        "change": {"TimeLimit": {"before": "08:00:00", "after": "02:00:00"}},
        "unchanged_host_memory": "64G",
        "m1_ledger": str(m1_path),
        "m1_ledger_sha256": e113.e78.digest(m1_path),
    }
    s2_protocol = root / "paper/preregistration/e113r1m1s2_qwen_backfill_20260819.md"
    s2_amender = root / "ops/exp_scaling/amend_e113r1m1_qwen_backfill.py"
    drift = [key for key, value in expected_s2.items() if s2.get(key) != value]
    if s2.get("protocol_sha256") != e113.e78.digest(s2_protocol):
        drift.append("s2.protocol_sha256")
    if s2.get("amender_sha256") != e113.e78.digest(s2_amender):
        drift.append("s2.amender_sha256")
    expected_s3 = {
        "schema": "e113r1m1s3_qwen_partition_amendment_v1",
        "installed": True,
        "outcomes_inspected": False,
        "job_id": 30790590,
        "same_job_id": True,
        "same_scientific_cells": True,
        "scientific_cells": 0,
        "change": {"Partition": {"before": "lowprio", "after": "all"}},
        "unchanged_account": "mltheory",
        "unchanged_time_limit": "02:00:00",
        "unchanged_host_memory": "64G",
        "m1_ledger": str(m1_path),
        "m1_ledger_sha256": e113.e78.digest(m1_path),
        "s2_amendment": str(s2_path),
        "s2_amendment_sha256": e113.e78.digest(s2_path),
    }
    s3_protocol = root / "paper/preregistration/e113r1m1s3_qwen_partition_20260819.md"
    s3_amender = root / "ops/exp_scaling/amend_e113r1m1_qwen_partition.py"
    drift.extend(
        f"s3.{key}" for key, value in expected_s3.items() if s3.get(key) != value
    )
    if s3.get("protocol_sha256") != e113.e78.digest(s3_protocol):
        drift.append("s3.protocol_sha256")
    if s3.get("amender_sha256") != e113.e78.digest(s3_amender):
        drift.append("s3.amender_sha256")
    if "TimeLimit=02:00:00" not in str(s2.get("scheduler_record_after", "")):
        drift.append("s2.scheduler_record_after")
    if "Partition=all" not in str(s3.get("scheduler_record_after", "")):
        drift.append("s3.scheduler_record_after")
    if drift:
        raise SystemExit("M1 scheduler amendment provenance drifted: " + ", ".join(drift))
    return s2, s3


def require_effective_gate(root: Path) -> dict[str, Any]:
    original = load_original(root)
    validate_m1_ledger(root, original)
    validate_scheduler_amendments(root)
    report = gate_audit.audit(root / R1_LEDGER, root / M1_LEDGER)
    if report.get("passed") is not True:
        details = list(report.get("violations", []))
        for smoke in report.get("smokes", []):
            details.extend(smoke.get("violations", []))
        detail = "; ".join(str(value) for value in details) or "not terminal"
        raise SystemExit(f"E113-R3 effective gate has not passed: {detail}")
    return report


def build_env(
    root: Path,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    falcon_root: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = e113.build_env(root, family, run, snapshot, falcon_root)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = target_path(root, family, domain, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(family, domain, seed),
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
            "OAT_ZERO_WATCHDOG_MAX_RESTARTS": "0",
        }
    )
    if family == "qwen05b":
        env.update(
            {
                "OAT_ZERO_ADAM_OFFLOAD": "0",
                "OAT_ZERO_ACTIVATION_OFFLOADING": "0",
            }
        )
    return env, target


def command(
    root: Path,
    family: str,
    run: dict[str, Any],
    env: dict[str, str],
) -> list[str]:
    result = e113.sbatch_command(root, family, run, env)
    replacements = {"--job-name=": job_name(family, str(run["domain"]), int(run["seed"]))}
    if family == "qwen05b":
        replacements.update(
            {
                "--partition=": QWEN_PARTITION,
                "--account=": m1.ACCOUNT,
                "--nodelist=": m1.NODELIST,
                "--gres=": m1.GRES,
            }
        )
    output: list[str] = []
    for token in result:
        replacement = next(
            (
                f"{prefix}{value}"
                for prefix, value in replacements.items()
                if token.startswith(prefix)
            ),
            None,
        )
        output.append(replacement if replacement is not None else token)
    return output


def audit_held(
    job_id: str,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    target: Path,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E113-R3 job {job_id}")
    domain, seed = str(run["domain"]), int(run["seed"])
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "RunTime=00:00:00",
        "Dependency=(null)",
        f"JobName={job_name(family, domain, seed)}",
        f"SAVE_PATH={target}",
        f"RUN_STAMP={run_stamp(family, domain, seed)}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        "OAT_ZERO_VARIANT=dapo",
        "OAT_ZERO_CRITIC_TYPE=grpo",
        "OAT_ZERO_DAPO_ENABLED=1",
        "OAT_ZERO_DAPO_CLIP_LOW=0.20",
        "OAT_ZERO_DAPO_CLIP_HIGH=0.28",
        "OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES=10",
        f"OAT_ZERO_MAX_QUERIES={e113.MAX_QUERIES}",
        f"OAT_ZERO_MAX_TRAIN={e113.TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={e113.PASSES}",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=0",
        "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=0",
        "OAT_ZERO_WATCHDOG_REQUEUE=0",
        "OAT_ZERO_WATCHDOG_MAX_RESTARTS=0",
    ]
    if family == "qwen05b":
        required.extend(
            (
                f"Partition={QWEN_PARTITION}",
                f"Account={m1.ACCOUNT}",
                f"ReqNodeList={m1.NODELIST}",
                "TresPerNode=gres/gpu:a6000:1",
                "OAT_ZERO_ADAM_OFFLOAD=0",
                "OAT_ZERO_ACTIVATION_OFFLOADING=0",
            )
        )
    else:
        node, gpu = e79.placement(domain, seed)
        required.extend((f"ReqNodeList={node}", f"gres/gpu:{gpu}=1"))
    missing = [value for value in required if value not in result.stdout]
    if missing:
        raise RuntimeError(f"held E113-R3 job {job_id} lacks {missing}")
    return result.stdout


def normalize_qwen_partition(job_id: str) -> None:
    """Undo the cluster's account-partition rewrite while the job is held."""

    subprocess.run(
        ["scontrol", "update", f"JobId={job_id}", f"Partition={QWEN_PARTITION}"],
        check=True,
    )


def validate_canceled_probe(root: Path) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(CANCELED_VALIDATION_PROBE)],
        capture_output=True,
        text=True,
        check=False,
    )
    record = result.stdout.strip()
    required = (
        f"JobId={CANCELED_VALIDATION_PROBE}",
        "JobName=e113r3-q05-count-s43",
        "JobState=CANCELLED",
        "RunTime=00:00:00",
        "Partition=mltheory",
        "SubmitLine=sbatch",
        "--partition=all",
        "Reason=JobHeldUser",
    )
    missing = [value for value in required if value not in record]
    target = target_path(root, "qwen05b", "countdown", 43)
    if target.exists():
        missing.append(f"zero-output target absent: {target}")
    if result.returncode or missing:
        raise SystemExit(f"canceled held validation probe drifted: {missing}")
    return record


def prepare_cells(
    root: Path,
    snapshot: Path,
    falcon_root: Path,
    *,
    require_fresh: bool = True,
) -> list[dict[str, Any]]:
    controls = {
        family: e113.control_index(root, family) for family in e113.FAMILY_SEEDS
    }
    cells: list[dict[str, Any]] = []
    for family in e113.FAMILY_SEEDS:
        for run in e113.references(root, family):
            domain, seed = str(run["domain"]), int(run["seed"])
            env, target = build_env(root, family, run, snapshot, falcon_root)
            if require_fresh and target.exists():
                raise SystemExit(f"refusing to overwrite E113-R3 output: {target}")
            cells.append(
                {
                    "family": family,
                    "run": run,
                    "env": env,
                    "target": target,
                    "control": controls[family][(domain, seed)],
                    "command": command(root, family, run, env),
                }
            )
    if len(cells) != SCIENTIFIC_CELLS:
        raise SystemExit(f"E113-R3 grid drifted to {len(cells)} cells")
    return cells


def main() -> int:
    root = e113.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    required_paths = (
        root / PROTOCOL,
        root / R1_LEDGER,
        root / M1_LEDGER,
        root / M1_S2_LEDGER,
        root / M1_S3_LEDGER,
        root / S1_PROTOCOL,
        root / ORIGINAL_LEDGER,
        root / CLOSED_R2_PROTOCOL,
        root / R3_S1_PROTOCOL,
        root / R3_S2_PROTOCOL,
    )
    for required in required_paths:
        if not required.is_file():
            raise SystemExit(f"required E113-R3 input is absent: {required}")
    ledger_path = root / LEDGER
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E113-R3 submission: {ledger_path}")

    original = load_original(root)
    validate_m1_ledger(root, original)
    validate_scheduler_amendments(root)
    snapshot = Path(str(original["snapshot_root"])).resolve()
    e113.verify_snapshot(snapshot)
    falcon_root = e79.model_root(root)
    cells = prepare_cells(root, snapshot, falcon_root)
    gate_report = gate_audit.audit(root / R1_LEDGER, root / M1_LEDGER)

    if args.dry_run or not args.submit:
        if not args.summary_only:
            for cell in cells:
                print(shlex.join(cell["command"]))
        print(
            f"[e113r3] scientific={len(cells)} gate_passed="
            f"{gate_report.get('passed') is True} snapshot={snapshot}"
        )
        return 0

    gate_report = require_effective_gate(root)
    canceled_probe_record = validate_canceled_probe(root)
    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in cells:
            family = str(cell["family"])
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            job_id = e113.submit_held(cell["command"])
            submitted.append(job_id)
            if family == "qwen05b":
                normalize_qwen_partition(job_id)
            held = audit_held(job_id, family, run, snapshot, cell["target"])
            control = cell["control"]
            records.append(
                {
                    "model_family": family,
                    "model": e113.MODEL_NAMES[family],
                    "domain": domain,
                    "arm": "dapo",
                    "seed": seed,
                    "run_stamp": run_stamp(family, domain, seed),
                    "run_dir": str(cell["target"]),
                    "job_id": int(job_id),
                    "placement_repair": "a6000_capacity_only" if family == "qwen05b" else "none",
                    "paired_control": {
                        "cohort": "e78" if family == "qwen05b" else "e79",
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        protocol_path = root / PROTOCOL
        r1_path = root / R1_LEDGER
        m1_path = root / M1_LEDGER
        m1_s2_path = root / M1_S2_LEDGER
        m1_s3_path = root / M1_S3_LEDGER
        s1_path = root / S1_PROTOCOL
        original_path = root / ORIGINAL_LEDGER
        r3_s1_path = root / R3_S1_PROTOCOL
        r3_s2_path = root / R3_S2_PROTOCOL
        payload = {
            "schema": "e113r3_dapo_full_relaunch_jobs_v1",
            "cohort": COHORT,
            "released": False,
            "scientific_cells": SCIENTIFIC_CELLS,
            "protocol": str(protocol_path),
            "protocol_sha256": e113.e78.digest(protocol_path),
            "launcher_sha256": e113.e78.digest(Path(__file__)),
            "effective_gate_auditor": str(root / "ops/exp_scaling/audit_e113r1m1_dapo_effective_gate.py"),
            "effective_gate_audit": gate_report,
            "r1_gate_ledger": str(r1_path),
            "r1_gate_ledger_sha256": e113.e78.digest(r1_path),
            "m1_gate_ledger": str(m1_path),
            "m1_gate_ledger_sha256": e113.e78.digest(m1_path),
            "m1_scheduler_amendment": str(m1_s2_path),
            "m1_scheduler_amendment_sha256": e113.e78.digest(m1_s2_path),
            "m1_partition_amendment": str(m1_s3_path),
            "m1_partition_amendment_sha256": e113.e78.digest(m1_s3_path),
            "falcon_shutdown_protocol": str(s1_path),
            "falcon_shutdown_protocol_sha256": e113.e78.digest(s1_path),
            "closed_r2_protocol": str(root / CLOSED_R2_PROTOCOL),
            "qwen_partition_amendment": str(r3_s1_path),
            "qwen_partition_amendment_sha256": e113.e78.digest(r3_s1_path),
            "held_partition_normalization": str(r3_s2_path),
            "held_partition_normalization_sha256": e113.e78.digest(r3_s2_path),
            "canceled_held_validation_probe": {
                "job_id": CANCELED_VALIDATION_PROBE,
                "runtime_seconds": 0,
                "released": False,
                "science_cell": False,
                "ledger_written": False,
                "reason": "submission_plugin_partition_rewrite",
                "scheduler_record": canceled_probe_record,
            },
            "original_e113_ledger": str(original_path),
            "original_e113_ledger_sha256": e113.e78.digest(original_path),
            "paired_control_ledgers": original["paired_control_ledgers"],
            "source_manifest": original["source_manifest"],
            "source_manifest_sha256": original["source_manifest_sha256"],
            "snapshot_root": str(snapshot),
            "snapshot_identity": original["snapshot_identity"],
            "models": e113.MODEL_NAMES,
            "falcon_model_revision": e79.MODEL_REVISION,
            "domains": list(e113.DOMAINS),
            "seeds": {key: list(value) for key, value in e113.FAMILY_SEEDS.items()},
            "passes": e113.PASSES,
            "train_rows": e113.TRAIN_ROWS,
            "target_steps": e113.TARGET_STEPS,
            "checkpoint_interval_steps": e113.CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(2 * e113.PASSES + 1)],
            "max_queries": e113.MAX_QUERIES,
            "qwen_capacity_repair": {
                "partition": QWEN_PARTITION,
                "account": m1.ACCOUNT,
                "nodelist": m1.NODELIST,
                "gres": m1.GRES,
                "optimizer_offload": False,
                "activation_offload": False,
            },
            "wrapper_requeue_after_learner_failure": False,
            "objective": e113.objective(),
            "smokes": {},
            "runs": records,
        }
        e113.e78.atomic_json(ledger_path, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e113.e78.atomic_json(ledger_path, payload)
    except Exception:
        e113.cancel(submitted)
        raise

    print(f"[e113r3] released {len(records)} scientific cells")
    print(f"[e113r3] ledger {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
