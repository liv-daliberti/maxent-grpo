#!/usr/bin/env python3
"""Submit E106's Python-only parser repair at all three paper scales."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e76_tuned_scale as snapshot_util  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


DOMAIN = "python_factors"
SCALES = tuple(e104.SCALE_SEEDS)
PROTOCOL = (
    "paper/preregistration/"
    "e106_python_lambda_normalization_three_scale_20260817.md"
)
DIAGNOSIS = "var/artifacts/e106_python_lambda_normalization_diagnosis.json"
LEDGER = "var/artifacts/e106_python_lambda_normalization_three_scale_jobs.json"
AUDIT = "var/artifacts/e106_python_lambda_normalization_combined_gate.json"
UNIT_EVIDENCE = "var/artifacts/e106_python_lambda_normalization_unit_tests.json"
BASE_LEDGER = e104.LEDGER
BASE_SNAPSHOT = (
    "var/artifacts/source_snapshots/e76_tuned_scale_52c279b1e48ddfa9"
)
SNAPSHOT = (
    "var/artifacts/source_snapshots/e106_python_lambda_b853595e3b158046"
)
SNAPSHOT_SHA256 = (
    "b853595e3b158046f73dee899a2c2a0558d4167e5b6971a36aa50565d9337d02"
)
SURFACE_VERSION = "python-factor-response-v2-latex-lambda"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for top in ("src", "ops"):
        base = root / top
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            result[str(path.relative_to(root))] = digest(path)
    return result


def verify_snapshot(root: Path, snapshot: Path) -> None:
    base = (root / BASE_SNAPSHOT).resolve()
    snapshot = snapshot.resolve()
    identity_path = snapshot / "SNAPSHOT_IDENTITY.json"
    if not identity_path.is_file():
        raise SystemExit(f"E106 snapshot identity is absent: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema") != "e106_python_lambda_runtime_snapshot_v1":
        raise SystemExit("E106 snapshot schema mismatch")
    if identity.get("sha256") != SNAPSHOT_SHA256:
        raise SystemExit("E106 snapshot identity hash mismatch")
    actual = snapshot_util.tree_hash(
        (snapshot / "src/oat_drgrpo", snapshot / "ops")
    )
    if actual != SNAPSHOT_SHA256:
        raise SystemExit(f"E106 snapshot content drifted: {actual}")
    before = tree_files(base)
    after = tree_files(snapshot)
    changed = sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )
    if changed != ["src/oat_drgrpo/math_grader.py"]:
        raise SystemExit(f"E106 snapshot has forbidden divergence: {changed}")
    grader = snapshot / "src/oat_drgrpo/math_grader.py"
    text = grader.read_text(encoding="utf-8")
    for needle in (
        SURFACE_VERSION,
        "_normalize_python_factor_lambda_surface",
        'r"^\\\\lambda(?:\\s+|\\\\,\\s*)n\\s*:\\s*"',
    ):
        if needle not in text:
            raise SystemExit(f"E106 snapshot lacks parser marker: {needle}")


def run_stamp(scale: str) -> str:
    return f"e106_{scale}_python_lambda_normalized_s{e104.SCALE_SEEDS[scale]}"


def save_path(root: Path, scale: str) -> Path:
    return root / "var/data" / (
        f"xdr_{e104.MODEL_TAGS[scale]}_{e104.VARIANT}_{run_stamp(scale)}"
    )


def job_name(scale: str) -> str:
    tag = {"qwen05b": "q05", "falcon1b": "f1", "qwen3b": "q3"}[scale]
    return f"e106-{tag}-python"


def python_template(root: Path, scale: str) -> dict[str, Any]:
    selected = [
        run for run in e104.references(root, scale) if run["domain"] == DOMAIN
    ]
    if len(selected) != 1:
        raise SystemExit(f"E106 {scale} expected one Python template")
    return selected[0]


def build_env(
    root: Path, scale: str, run: dict[str, Any], snapshot: Path
) -> tuple[dict[str, str], Path]:
    env, _ = e104.build_env(root, scale, run, snapshot)
    target = save_path(root, scale)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(scale),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        }
    )
    return env, target


def sbatch_command(
    root: Path, scale: str, run: dict[str, Any], env: dict[str, str]
) -> list[str]:
    command = e104.sbatch_command(root, scale, run, env)
    return [
        f"--job-name={job_name(scale)}" if token.startswith("--job-name=") else token
        for token in command
    ]


def held_job_audit(job_id: str, scale: str, snapshot: Path) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E106 job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"RUN_STAMP={run_stamp(scale)}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        f"OAT_ZERO_SEED={e104.SCALE_SEEDS[scale]}",
        f"OAT_ZERO_MAX_TRAIN={e104.SMOKE_TRAIN_ROWS}",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held E106 job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    root = repo_root()
    snapshot = (args.snapshot_root or (root / SNAPSHOT)).resolve()
    if snapshot != (root / SNAPSHOT).resolve():
        raise SystemExit("E106 requires its exact content-addressed snapshot")
    verify_snapshot(root, snapshot)
    required = [
        root / PROTOCOL,
        root / DIAGNOSIS,
        root / BASE_LEDGER,
        root / "tests/test_python_modebench.py",
        root / UNIT_EVIDENCE,
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required E106 input is absent: {path}")
    diagnosis = json.loads((root / DIAGNOSIS).read_text(encoding="utf-8"))
    if diagnosis.get("schema") != "e106_python_lambda_normalization_diagnosis_v1":
        raise SystemExit("E106 diagnosis schema mismatch")
    if diagnosis.get("post_e104_update_outcomes_inspected") is not False:
        raise SystemExit("E106 diagnosis is not outcome-blind")
    unit_evidence = json.loads(
        (root / UNIT_EVIDENCE).read_text(encoding="utf-8")
    )
    if (
        unit_evidence.get("schema")
        != "e106_python_lambda_normalization_unit_tests_v1"
        or unit_evidence.get("passed") is not True
        or unit_evidence.get("returncode") != 0
        or unit_evidence.get("snapshot_sha256") != SNAPSHOT_SHA256
        or "53 passed" not in str(unit_evidence.get("stdout", ""))
    ):
        raise SystemExit("E106 frozen unit-test evidence is not passing")
    ledger_path = root / LEDGER
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E106 submission: {ledger_path}")

    planned: list[dict[str, Any]] = []
    for scale in SCALES:
        template = python_template(root, scale)
        env, target = build_env(root, scale, template, snapshot)
        if args.submit and target.exists():
            raise SystemExit(f"refusing to overwrite E106 run: {target}")
        planned.append(
            {
                "scale": scale,
                "model_tag": e104.MODEL_TAGS[scale],
                "domain": DOMAIN,
                "seed": e104.SCALE_SEEDS[scale],
                "run_stamp": run_stamp(scale),
                "run_dir": str(target),
                "command": sbatch_command(root, scale, template, env),
            }
        )
    if len(planned) != 3:
        raise SystemExit(f"E106 expected three cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
        print(f"[e106] dry_run=True cells=3 snapshot={snapshot}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "E106 submission failed")
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid E106 job id: {result.stdout!r}")
            submitted.append(job_id)
            scale = str(cell["scale"])
            held = held_job_audit(job_id, scale, snapshot)
            records.append(
                {key: cell[key] for key in (
                    "scale", "model_tag", "domain", "seed", "run_stamp", "run_dir"
                )}
                | {
                    "job_id": int(job_id),
                    "stdout": str(root / "var/artifacts/logs" / f"{job_name(scale)}-{job_id}.out"),
                    "stderr": str(root / "var/artifacts/logs" / f"{job_name(scale)}-{job_id}.err"),
                    "held_scheduler_record": held,
                }
            )
        payload = {
            "schema": "e106_python_lambda_normalization_three_scale_jobs_v1",
            "protocol": str(root / PROTOCOL),
            "protocol_sha256": digest(root / PROTOCOL),
            "diagnosis": str(root / DIAGNOSIS),
            "diagnosis_sha256": digest(root / DIAGNOSIS),
            "launcher_sha256": digest(Path(__file__)),
            "base_e104_ledger": str(root / BASE_LEDGER),
            "base_e104_ledger_sha256": digest(root / BASE_LEDGER),
            "snapshot_root": str(snapshot),
            "snapshot_sha256": SNAPSHOT_SHA256,
            "surface_version": SURFACE_VERSION,
            "scales": list(SCALES),
            "domain": DOMAIN,
            "seeds": e104.SCALE_SEEDS,
            "target_steps": e104.SMOKE_TARGET_STEPS,
            "supersedes": "the three E104 python_factors mechanism cells",
            "pointmaze": "excluded",
            "audit": str(root / AUDIT),
            "unit_evidence": str(root / UNIT_EVIDENCE),
            "unit_evidence_sha256": digest(root / UNIT_EVIDENCE),
            "runs": records,
            "released": False,
        }
        e81.atomic_json(ledger_path, payload)
        for job_id in submitted:
            released = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if released.returncode != 0:
                raise RuntimeError(f"release failed for E106 job {job_id}")
        payload["released"] = True
        e81.atomic_json(ledger_path, payload)
    except Exception:
        e81.cancel(submitted)
        if ledger_path.exists():
            ledger_path.unlink()
        raise
    print(f"[e106] cells=3 released=3 snapshot={snapshot} ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
