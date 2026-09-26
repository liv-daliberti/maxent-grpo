#!/usr/bin/env python3
"""Submit the frozen cross-scale E95 plain-GRPO control suite.

E95 asks whether correct-mode collapse is specific to Dr.GRPO's debiasing or
also occurs with ordinary GRPO. A training seed covers all five ModeBench
domains. The frozen suite is one 3B seed and five seeds at each smaller scale.

Every cell inherits the exact submitted argv of its parent control. The
parent's immutable runtime is copied and overlaid with only the three files
needed to expose ``critic_type=grpo``. Jobs are submitted held, audited, and
recorded atomically before any job is released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VARIANT = "grpo_plain_control"
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "Qwen2.5-3B": {
        "parent_ledger": "e80r1_qwen3b_aligned_verified_replay_jobs.json",
        "ledger": "e95_plain_grpo_Qwen25-3B_jobs.json",
        "seeds": (70,),
        "tag": "qwen3b",
    },
    "Falcon3-1B": {
        "parent_ledger": "e79_falcon1b_aligned_verified_replay_jobs.json",
        "ledger": "e95_plain_grpo_Falcon3-1B_jobs.json",
        "seeds": (55, 56, 57, 58, 59),
        "tag": "falcon1b",
    },
    "Qwen2.5-0.5B": {
        "parent_ledger": "e78_verified_replay_only_05b_jobs.json",
        "ledger": "e95_plain_grpo_Qwen25-05B_jobs.json",
        "seeds": (43, 44, 45, 46, 47),
        "tag": "qwen05b",
    },
}
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "ops/run_experiment.sh",
    "ops/train.sh",
)
SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/args.py", 'critic_type not in ("drgrpo", "grpo")'),
    ("ops/run_experiment.sh", "grpo_plain_control)"),
    ("ops/run_experiment.sh", "export OAT_ZERO_CRITIC_TYPE=grpo"),
    ("ops/train.sh", '--critic_type "${OAT_ZERO_CRITIC_TYPE:-drgrpo}"'),
)
PROTOCOL = "paper/preregistration/e95_plain_grpo_cross_scale_20260812.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def submit_line(run: dict[str, Any]) -> list[str]:
    """Return the exact sbatch argv recorded for a parent cell."""

    record = str(run.get("held_scheduler_record", ""))
    if "SubmitLine=" not in record:
        raise RuntimeError(
            f"{run.get('domain')}/{run.get('seed')}: no SubmitLine recorded"
        )
    line = record.split("SubmitLine=", 1)[1]
    for stop in (" WorkDir=", " StdErr=", " StdOut=", " StdIn=", " TresPer"):
        if stop in line:
            line = line.split(stop, 1)[0]
    return shlex.split(line.strip())


def export_pairs(argv: list[str]) -> dict[str, str]:
    token = next((item for item in argv if item.startswith("--export=")), "")
    if not token:
        raise RuntimeError("parent submission has no --export token")
    return {
        pair.split("=", 1)[0]: pair.split("=", 1)[1]
        for pair in token[len("--export=") :].split(",")
        if "=" in pair
    }


def base_snapshot(run: dict[str, Any]) -> Path:
    exports = export_pairs(submit_line(run))
    source = Path(exports.get("OAT_ZERO_SOURCE_ROOT", "")).resolve()
    ops = Path(exports.get("OAT_ZERO_OPS_SNAPSHOT_ROOT", "")).resolve()
    if source.name != "src" or ops.name != "ops" or source.parent != ops.parent:
        raise RuntimeError(
            f"{run['domain']}/s{run['seed']}: parent runtime roots do not pair"
        )
    if not (source / "oat_drgrpo/__init__.py").is_file():
        raise RuntimeError(f"invalid parent source snapshot: {source}")
    return source.parent


def ensure_snapshot(base: Path, family_tag: str) -> tuple[Path, list[str]]:
    """Derive an immutable plain-GRPO runtime from one parent runtime."""

    identity = hashlib.sha256()
    identity.update(str(base).encode("utf-8"))
    for relative in PATCHED_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"missing E95 runtime patch: {source}")
        identity.update(relative.encode("utf-8"))
        identity.update(digest(source).encode("ascii"))
    identity_hex = identity.hexdigest()
    target = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e95_plain_grpo_{family_tag}_{identity_hex[:16]}"
    )
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            for entry in ("src", "ops"):
                shutil.copytree(
                    base / entry,
                    staging / entry,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            for relative in PATCHED_FILES:
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
                shutil.copystat(ROOT / relative, destination)
            (staging / "SNAPSHOT_IDENTITY.json").write_text(
                json.dumps(
                    {
                        "schema": "e95_plain_grpo_runtime_snapshot_v1",
                        "derived_from": str(base),
                        "patched_files": list(PATCHED_FILES),
                        "sha256": identity_hex,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    missing = [
        f"{relative}: {needle!r}"
        for relative, needle in SNAPSHOT_REQUIREMENTS
        if needle not in (target / relative).read_text(encoding="utf-8")
    ]
    if missing:
        raise RuntimeError("E95 snapshot is not plain-GRPO capable: " + "; ".join(missing))
    return target, list(PATCHED_FILES)


def selected_controls(family: str, ledger: dict[str, Any]) -> list[dict[str, Any]]:
    spec = FAMILY_SPECS[family]
    expected = {
        (domain, seed) for domain in DOMAINS for seed in tuple(spec["seeds"])
    }
    controls = [
        run
        for run in ledger.get("runs", [])
        if str(run.get("arm")) == "control"
        and (str(run.get("domain")), int(run.get("seed", -1))) in expected
    ]
    found = {(str(run["domain"]), int(run["seed"])) for run in controls}
    if found != expected or len(controls) != len(expected):
        raise RuntimeError(
            f"{family}: parent controls do not cover exactly {len(expected)} frozen cells"
        )
    order = {domain: index for index, domain in enumerate(DOMAINS)}
    return sorted(controls, key=lambda run: (order[str(run["domain"])], int(run["seed"])))


def stem(family: str, run: dict[str, Any]) -> str:
    return (
        f"e95_{FAMILY_SPECS[family]['tag']}_{run['domain']}"
        f"_grpo_s{int(run['seed'])}"
    )


def job_name(family: str, run: dict[str, Any]) -> str:
    domain = str(run["domain"]).replace("_", "")[:6]
    return f"e95-{FAMILY_SPECS[family]['tag'][:5]}-{domain}-s{int(run['seed'])}"


def swap(
    argv: list[str],
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    nice: int,
) -> list[str]:
    """Change only objective, immutable runtime, outputs, name, hold, and nice."""

    output: list[str] = []
    replaced_nice = False
    for token in argv:
        if token.startswith("--export="):
            removed = {
                "OAT_ZERO_VARIANT",
                "OAT_ZERO_SOURCE_ROOT",
                "OAT_ZERO_OPS_SNAPSHOT_ROOT",
                "SAVE_PATH",
                "RUN_STAMP",
            }
            kept = [
                pair
                for pair in token[len("--export=") :].split(",")
                if pair.split("=", 1)[0] not in removed
            ]
            run_stem = stem(family, run)
            kept.extend(
                (
                    f"OAT_ZERO_VARIANT={VARIANT}",
                    f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
                    f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
                    f"SAVE_PATH={ROOT / 'var/checkpoints' / run_stem}",
                    f"RUN_STAMP={run_stem}",
                )
            )
            output.append("--export=" + ",".join(kept))
        elif token.startswith("--job-name="):
            output.append(f"--job-name={job_name(family, run)}")
        elif token.startswith("--nice="):
            output.append(f"--nice={nice}")
            replaced_nice = True
        elif token == "--hold":
            output.append(token)
        else:
            output.append(token)
    if "--hold" not in output:
        output.insert(2, "--hold")
    if not replaced_nice:
        output.insert(-1, f"--nice={nice}")
    return output


def preflight(snapshot: Path, family: str, run: dict[str, Any], nice: int) -> None:
    """Exercise variant selection and argument plumbing without starting training."""

    command = swap(submit_line(run), family, run, snapshot, nice)
    env = os.environ.copy()
    env.update(export_pairs(command))
    with tempfile.TemporaryDirectory(prefix="e95-preflight-", dir="/tmp") as temporary:
        env.update(
            {
                "SAVE_PATH": str(Path(temporary) / "run"),
                "RUN_STAMP": "e95_plain_grpo_preflight",
                "OAT_ZERO_TRAIN_SCRIPT": "/bin/true",
                "OAT_ZERO_AUTO_RESUME": "0",
                "OAT_ZERO_WATCHDOG_REQUEUE": "0",
            }
        )
        result = subprocess.run(
            [str(snapshot / "ops/run_experiment.sh")],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    combined = result.stdout + result.stderr
    if result.returncode or "[experiment] variant=grpo_plain_control" not in combined:
        raise RuntimeError(
            f"{family}: plain-GRPO runtime preflight failed:\n{combined[-4000:]}"
        )


def submit_held(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch response: {result.stdout!r}")
    return job_id


def audit_held(
    job_id: str,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E95 job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName={job_name(family, run)}",
        f"OAT_ZERO_SEED={int(run['seed'])}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
    )
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held E95 job {job_id} lacks {missing}")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def existing_ledger_metadata(path: Path, replace: bool) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if not replace:
        raise RuntimeError(f"refusing duplicate E95 submission: {path}")
    payload = json.loads(path.read_text())
    return {
        "ledger_sha256": digest(path),
        "runs": payload.get("runs", []),
        "released": bool(payload.get("released", False)),
    }


def payload_for(
    family: str,
    parent: dict[str, Any],
    snapshot: Path,
    patched: list[str],
    records: list[dict[str, Any]],
    superseded: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    protocol = ROOT / PROTOCOL
    return {
        "schema": "e95_plain_grpo_control_jobs_v2",
        "experiment": "E95",
        "family": family,
        "inherits_cells_from": spec["parent_ledger"],
        "protocol": str(protocol),
        "protocol_sha256": digest(protocol),
        "launcher_sha256": digest(Path(__file__)),
        "snapshot_root": str(snapshot),
        "snapshot_patched_files": patched,
        "variant": VARIANT,
        "objective": "plain_GRPO_reward_only_control",
        "scientific_difference": "against reward-only Dr.GRPO: critic_type=grpo only",
        "arms": [VARIANT],
        "seeds": list(spec["seeds"]),
        "domains": list(DOMAINS),
        "passes": parent.get("passes"),
        "train_rows": parent.get("train_rows"),
        "target_steps": parent.get("target_steps"),
        "checkpoint_interval_steps": parent.get("checkpoint_interval_steps"),
        "released": False,
        "supersedes": superseded,
        "runs": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("all", *FAMILY_SPECS), default="all")
    parser.add_argument("--nice", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    protocol = ROOT / PROTOCOL
    if not protocol.is_file():
        raise SystemExit(f"missing frozen E95 protocol: {protocol}")

    families = list(FAMILY_SPECS) if args.family == "all" else [args.family]
    plans: dict[str, dict[str, Any]] = {}
    for family in families:
        spec = FAMILY_SPECS[family]
        parent_path = ROOT / "var/artifacts" / str(spec["parent_ledger"])
        parent = json.loads(parent_path.read_text())
        controls = selected_controls(family, parent)
        bases = {base_snapshot(run) for run in controls}
        if len(bases) != 1:
            raise RuntimeError(f"{family}: selected controls span multiple runtimes")
        snapshot, patched = ensure_snapshot(bases.pop(), str(spec["tag"]))
        preflight(snapshot, family, controls[0], args.nice)
        ledger_path = ROOT / "var/artifacts" / str(spec["ledger"])
        superseded = existing_ledger_metadata(ledger_path, args.replace_existing)
        commands = [
            swap(submit_line(run), family, run, snapshot, args.nice)
            for run in controls
        ]
        plans[family] = {
            "parent": parent,
            "controls": controls,
            "snapshot": snapshot,
            "patched": patched,
            "ledger": ledger_path,
            "superseded": superseded,
            "commands": commands,
        }

    if args.dry_run or not args.submit:
        for family, plan in plans.items():
            seeds = FAMILY_SPECS[family]["seeds"]
            print(
                f"[e95] {family}: seeds={list(seeds)} cells={len(plan['controls'])} "
                f"snapshot={plan['snapshot']}"
            )
            print("  first: " + shlex.join(plan["commands"][0]))
        print(f"[e95] total cells={sum(len(plan['controls']) for plan in plans.values())}")
        return 0

    submitted: list[str] = []
    records: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    try:
        for family, plan in plans.items():
            for run, command in zip(plan["controls"], plan["commands"]):
                job_id = submit_held(command)
                submitted.append(job_id)
                held = audit_held(job_id, family, run, plan["snapshot"])
                records[family].append(
                    {
                        "arm": VARIANT,
                        "domain": str(run["domain"]),
                        "seed": int(run["seed"]),
                        "job_id": int(job_id),
                        "run_stamp": stem(family, run),
                        "run_dir": str(ROOT / "var/checkpoints" / stem(family, run)),
                        "parent_control_job_id": int(run["job_id"]),
                        "held_scheduler_record": held,
                    }
                )
        payloads = {
            family: payload_for(
                family,
                plan["parent"],
                plan["snapshot"],
                plan["patched"],
                records[family],
                plan["superseded"],
            )
            for family, plan in plans.items()
        }
        for family, payload in payloads.items():
            atomic_json(plans[family]["ledger"], payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        for family, payload in payloads.items():
            payload["released"] = True
            atomic_json(plans[family]["ledger"], payload)
    except Exception:
        cancel(submitted)
        raise

    for family in families:
        print(
            f"[e95] released {len(records[family])} {family} cells; "
            f"ledger={plans[family]['ledger']}"
        )
    print(f"[e95] released total={len(submitted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
