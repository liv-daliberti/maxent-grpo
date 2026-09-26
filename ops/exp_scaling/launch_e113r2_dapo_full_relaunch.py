#!/usr/bin/env python3
"""Gate and launch the fresh 50-cell E113-R2 DAPO comparative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e113r1_dapo_recovery_smokes as r1_audit  # noqa: E402
import launch_e113_dapo_direct_baseline as e113  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402


COHORT = "e113r2"
LEDGER = "var/artifacts/e113r2_dapo_full_relaunch_jobs.json"
PROTOCOL = "paper/preregistration/e113r2_dapo_full_relaunch_20260819.md"
R1_LEDGER = "var/artifacts/e113r1_dapo_recovery_smoke_jobs.json"
ORIGINAL_LEDGER = e113.LEDGER
SCIENTIFIC_CELLS = 50


def run_stamp(family: str, domain: str, seed: int) -> str:
    return f"e113r2_{family}_{e113.e78.DOMAIN_TAGS[domain]}_dapo_s{seed}"


def job_name(family: str, domain: str, seed: int) -> str:
    family_tag = "q05" if family == "qwen05b" else "f1"
    domain_tag = e113.e78.DOMAIN_TAGS[domain][:5]
    return f"e113r2-{family_tag}-{domain_tag}-s{seed}"


def target_path(root: Path, family: str, domain: str, seed: int) -> Path:
    return (
        root
        / "var/data"
        / f"xdr_{e113.MODEL_TAGS[family]}_dapo_{run_stamp(family, domain, seed)}"
    )


def build_env(
    root: Path,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    falcon_root: Path | None,
) -> tuple[dict[str, str], Path]:
    env, _ = e113.build_env(root, family, run, snapshot, falcon_root)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = target_path(root, family, domain, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(family, domain, seed),
            # DAPO's ten-batch exhaustion is a terminal scientific feasibility
            # result, not a wrapper-level invitation to try the update again.
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
            "OAT_ZERO_WATCHDOG_MAX_RESTARTS": "0",
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
    expected = job_name(family, str(run["domain"]), int(run["seed"]))
    return [
        f"--job-name={expected}" if token.startswith("--job-name=") else token
        for token in result
    ]


def validate_r1_ledger(
    root: Path,
    original: dict[str, Any],
) -> dict[str, Any]:
    path = root / R1_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    amendment = Path(str(payload.get("amendment", "")))
    expected = {
        "schema": "e113r1_dapo_recovery_smoke_jobs_v1",
        "released": True,
        "scientific_cells": 0,
        "target_steps": 32,
        "passes": 1,
        "smoke_domain": "graph_coloring",
        "max_generation_batches": 10,
        "max_queries_per_smoke": 5120,
        "original_e113_ledger": str(root / ORIGINAL_LEDGER),
        "original_e113_ledger_sha256": e113.e78.digest(root / ORIGINAL_LEDGER),
        "snapshot_root": original["snapshot_root"],
        "snapshot_identity": original["snapshot_identity"],
        "objective": e113.objective(),
        "runs": [],
    }
    drift = [key for key, value in expected.items() if payload.get(key) != value]
    if not amendment.is_file():
        drift.append("amendment")
    elif payload.get("amendment_sha256") != e113.e78.digest(amendment):
        drift.append("amendment_sha256")
    r1_launcher = root / "ops/exp_scaling/launch_e113r1_dapo_recovery_smokes.py"
    if payload.get("launcher_sha256") != e113.e78.digest(r1_launcher):
        drift.append("launcher_sha256")
    smokes = payload.get("smokes")
    if not isinstance(smokes, dict) or set(smokes) != set(e113.FAMILY_SEEDS):
        drift.append("smokes")
    if drift:
        raise SystemExit("E113-R1 ledger provenance drifted: " + ", ".join(drift))
    return payload


def require_r1_gate(root: Path) -> dict[str, Any]:
    original = load_original(root)
    validate_r1_ledger(root, original)
    report = r1_audit.audit(root / R1_LEDGER)
    if report.get("passed") is not True:
        violations = list(report.get("violations", []))
        for smoke in report.get("smokes", []):
            violations.extend(smoke.get("violations", []))
        detail = "; ".join(str(value) for value in violations) or "not terminal"
        raise SystemExit(f"E113-R1 recovery gate has not passed: {detail}")
    return report


def load_original(root: Path) -> dict[str, Any]:
    path = root / ORIGINAL_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113_dapo_direct_baseline_jobs_v1":
        raise SystemExit("original E113 ledger schema drifted")
    if payload.get("released") is not True or len(payload.get("runs", [])) != 50:
        raise SystemExit("original E113 ledger is not the released 50-cell record")
    return payload


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
        raise RuntimeError(f"cannot inspect held E113-R2 job {job_id}")
    domain, seed = str(run["domain"]), int(run["seed"])
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
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
        required.append(f"ReqNodeList={run['source_node']}")
    else:
        node, gpu = e79.placement(domain, seed)
        required.extend((f"ReqNodeList={node}", f"gres/gpu:{gpu}=1"))
    missing = [value for value in required if value not in result.stdout]
    if missing:
        raise RuntimeError(f"held E113-R2 job {job_id} lacks {missing}")
    return result.stdout


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
                raise SystemExit(f"refusing to overwrite E113-R2 output: {target}")
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
        raise SystemExit(f"E113-R2 grid drifted to {len(cells)} cells")
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

    protocol_path = root / PROTOCOL
    ledger_path = root / LEDGER
    r1_path = root / R1_LEDGER
    original_path = root / ORIGINAL_LEDGER
    for required in (protocol_path, r1_path, original_path):
        if not required.is_file():
            raise SystemExit(f"required E113-R2 input is absent: {required}")
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E113-R2 submission: {ledger_path}")

    original = load_original(root)
    validate_r1_ledger(root, original)
    snapshot = Path(str(original["snapshot_root"])).resolve()
    e113.verify_snapshot(snapshot)
    falcon_root = e79.model_root(root)
    cells = prepare_cells(root, snapshot, falcon_root)
    gate_report = r1_audit.audit(r1_path)

    if args.dry_run or not args.submit:
        if not args.summary_only:
            for cell in cells:
                print(shlex.join(cell["command"]))
        print(
            f"[e113r2] scientific={len(cells)} gate_passed="
            f"{gate_report.get('passed') is True} snapshot={snapshot}"
        )
        return 0

    gate_report = require_r1_gate(root)
    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in cells:
            family = str(cell["family"])
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            job_id = e113.submit_held(cell["command"])
            submitted.append(job_id)
            held = audit_held(
                job_id,
                family,
                run,
                snapshot,
                cell["target"],
            )
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
                    "paired_control": {
                        "cohort": "e78" if family == "qwen05b" else "e79",
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e113r2_dapo_full_relaunch_jobs_v1",
            "cohort": COHORT,
            "released": False,
            "scientific_cells": SCIENTIFIC_CELLS,
            "protocol": str(protocol_path),
            "protocol_sha256": e113.e78.digest(protocol_path),
            "launcher_sha256": e113.e78.digest(Path(__file__)),
            "r1_gate_ledger": str(r1_path),
            "r1_gate_ledger_sha256": e113.e78.digest(r1_path),
            "r1_gate_audit": gate_report,
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
            "seeds": {
                key: list(value) for key, value in e113.FAMILY_SEEDS.items()
            },
            "passes": e113.PASSES,
            "train_rows": e113.TRAIN_ROWS,
            "target_steps": e113.TARGET_STEPS,
            "checkpoint_interval_steps": e113.CHECKPOINT_INTERVAL,
            "registered_passes": [
                index / 2 for index in range(2 * e113.PASSES + 1)
            ],
            "max_queries": e113.MAX_QUERIES,
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

    print(f"[e113r2] released {len(records)} scientific cells")
    print(f"[e113r2] ledger {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
