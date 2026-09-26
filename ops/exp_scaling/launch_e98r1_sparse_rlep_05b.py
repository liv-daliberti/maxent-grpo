#!/usr/bin/env python3
"""Submit E98-R1 sparse RLEP smoke, audit, and scientific cells."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e98_rlep_pool as pool_audit  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e98_rlep_05b as e98  # noqa: E402


PROTOCOL = ROOT / "paper/preregistration/e98r1_sparse_rlep_dr_05b_20260813.md"
SOURCE_LEDGER = ROOT / "var/artifacts/e98_rlep_dr_05b_jobs.json"
LEDGER = ROOT / "var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json"
SMOKE_AUDIT = "ops/exp_scaling/audit_e98r1_sparse_rlep_smoke.py"
SNAPSHOT_PREFIX = "e98r1_sparse_rlep_dr"
ARM = "rlep_dr_sparse"
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "src/oat_drgrpo/learner/grpo.py",
    "src/oat_drgrpo/rlep.py",
    "ops/run_experiment.sh",
    "ops/train.sh",
    "ops/exp_scaling/audit_e98_rlep_pool.py",
    SMOKE_AUDIT,
)
SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/args.py", "rlep_sparse_fallback: bool"),
    ("src/oat_drgrpo/learner/grpo.py", "rlep_replay_eligible"),
    ("src/oat_drgrpo/rlep.py", "allow_sparse"),
    ("ops/run_experiment.sh", "OAT_ZERO_RLEP_SPARSE_FALLBACK"),
    ("ops/train.sh", "--rlep-sparse-fallback"),
    ("ops/exp_scaling/audit_e98_rlep_pool.py", "--allow-sparse"),
    (SMOKE_AUDIT, "E98R1_SMOKE_COMPLETE.json"),
)


def run_stamp(domain: str, seed: int) -> str:
    return f"e98r1_sparse_rlep_{e81.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(domain: str, seed: int) -> Path:
    return ROOT / "var/data" / f"xdr_{e81.MODEL_TAG}_rlep_{run_stamp(domain, seed)}"


def gpu_command(
    run: dict[str, Any],
    env: dict[str, str],
    *,
    name: str,
    dependency: str = "",
) -> list[str]:
    command = e98.gpu_command(ROOT, run, env, name=name, dependency=dependency)
    return ["--nice=0" if item == "--nice=100" else item for item in command]


def training_env(
    run: dict[str, Any], *, snapshot: Path, pool: Path
) -> tuple[dict[str, str], Path]:
    env, _ = e98.training_env(ROOT, run, snapshot=snapshot, pool=pool)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(domain, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, seed),
            "OAT_ZERO_RLEP_SPARSE_FALLBACK": "1",
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        }
    )
    return env, target


def smoke_env(
    run: dict[str, Any], *, snapshot: Path, pool: Path
) -> tuple[dict[str, str], Path]:
    env, _ = training_env(run, snapshot=snapshot, pool=pool)
    target = ROOT / "var/data/e98r1_sparse_rlep_smoke_graph_s43"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": "e98r1_sparse_rlep_smoke_graph_s43",
            "OAT_ZERO_MAX_TRAIN": "32",
            "OAT_ZERO_MAX_QUERIES": "100000000",
            "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
            "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": "32",
            "OAT_ZERO_SAVE_STEPS": "32",
            "OAT_ZERO_SAVE_FROM": "32",
            "OAT_ZERO_AUTO_RESUME": "0",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        }
    )
    return env, target


def smoke_audit_command(
    *, snapshot: Path, smoke_id: str, smoke_target: Path
) -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = snapshot / SMOKE_AUDIT
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--job-name=e98r1-rlep-smoke-audit",
        f"--dependency=afterok:{smoke_id}",
        f"--export=ALL,PYTHONPATH={snapshot / 'src'}",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:15:00",
        "--nice=0",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        (
            f"{python} {script} --run-root {smoke_target} "
            "--expected-terminal-step 32"
        ),
    ]


def verify_snapshot(snapshot: Path) -> None:
    missing = [
        f"{name}: {needle!r}"
        for name, needle in SNAPSHOT_REQUIREMENTS
        if needle not in (snapshot / name).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit(
            "E98-R1 snapshot lacks sparse RLEP contracts:\n  " + "\n  ".join(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    if not PROTOCOL.is_file() or not SOURCE_LEDGER.is_file():
        raise SystemExit("E98-R1 protocol or source E98 ledger is absent")
    if args.submit and LEDGER.exists():
        raise SystemExit(f"refusing duplicate E98-R1 ledger: {LEDGER}")

    source = json.loads(SOURCE_LEDGER.read_text(encoding="utf-8"))
    if not source.get("released"):
        raise SystemExit("source E98 ledger was not released")
    pool_records = {
        (str(row["domain"]), int(row["seed"])): row
        for row in source["collection"]["records"]
    }
    runs = e98.selected_references(ROOT)
    pairs = e98.selected_pairs(ROOT)
    expected = {
        (domain, seed) for domain in e98.DOMAINS for seed in e98.SEEDS
    }
    if set(pool_records) != expected:
        raise SystemExit("source E98 pool grid is incomplete")

    base_snapshot = Path(str(source["snapshot_root"])).resolve()
    snapshot, patched = e81.ensure_paired_snapshot(
        ROOT,
        base_snapshot,
        prefix=SNAPSHOT_PREFIX,
        patched_files=PATCHED_FILES,
    )
    verify_snapshot(snapshot)

    cells: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for run in runs:
        domain, seed = str(run["domain"]), int(run["seed"])
        pool = Path(str(pool_records[(domain, seed)]["pool_root"])).resolve()
        if not pool.is_dir():
            raise SystemExit(f"missing immutable E98 pool: {pool}")
        target = save_path(domain, seed)
        if target.exists():
            raise SystemExit(f"refusing pre-existing E98-R1 target: {target}")
        if args.submit:
            audit = pool_audit.audit_pool(
                pool, expected_prompts=e98.TRAIN_ROWS, allow_sparse=True
            )
        else:
            candidate = pool / "RLEP_SPARSE_POOL_COMPLETE.json"
            audit = (
                json.loads(candidate.read_text(encoding="utf-8"))
                if candidate.is_file()
                else {"pool_root": str(pool), "status": "audit_on_submit"}
            )
        env, _ = training_env(run, snapshot=snapshot, pool=pool)
        cells.append(
            {
                "run": run,
                "pool": pool,
                "target": target,
                "env": env,
                "control": pairs[(domain, seed)]["control"],
            }
        )
        audit_records.append(
            {
                "domain": domain,
                "seed": seed,
                "source_collection_job_id": int(
                    pool_records[(domain, seed)]["collection_job_id"]
                ),
                "pool_root": str(pool),
                "receipt": str(pool / "RLEP_SPARSE_POOL_COMPLETE.json"),
                "audit": audit,
            }
        )

    smoke_cell = next(
        cell
        for cell in cells
        if cell["run"]["domain"] == "graph_coloring"
        and int(cell["run"]["seed"]) == 43
    )
    smoke_vars, smoke_target = smoke_env(
        smoke_cell["run"], snapshot=snapshot, pool=smoke_cell["pool"]
    )
    if smoke_target.exists():
        raise SystemExit(f"refusing pre-existing E98-R1 smoke: {smoke_target}")

    if args.dry_run or not args.submit:
        print(
            shlex.join(
                gpu_command(
                    smoke_cell["run"],
                    smoke_vars,
                    name="e98r1-rlep-smoke",
                )
            )
        )
        print("<smoke audit depends on smoke>")
        for cell in cells:
            domain = str(cell["run"]["domain"])
            seed = int(cell["run"]["seed"])
            print(
                shlex.join(
                    gpu_command(
                        cell["run"],
                        cell["env"],
                        name=f"e98r1-{e81.DOMAIN_TAGS[domain][:6]}-s{seed}",
                        dependency="SMOKE_AUDIT_ID",
                    )
                )
            )
        print(
            f"[e98r1] sparse audits=15 smoke=1 science=15 "
            f"snapshot={snapshot} patched={patched}"
        )
        return 0

    submitted: list[str] = []
    run_records: list[dict[str, Any]] = []
    try:
        smoke_id = e98.submit_held(
            gpu_command(
                smoke_cell["run"],
                smoke_vars,
                name="e98r1-rlep-smoke",
            )
        )
        submitted.append(smoke_id)
        smoke_held = e98.audit_held(
            smoke_id,
            name="e98r1-rlep-smoke",
            expected=(
                f"ReqNodeList={smoke_cell['run']['source_node']}",
                "OAT_ZERO_VARIANT=rlep",
                "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
                "OAT_ZERO_MAX_TRAIN=32",
                "OAT_ZERO_NUM_PROMPT_EPOCH=1",
            ),
        )

        smoke_audit_id = e98.submit_held(
            smoke_audit_command(
                snapshot=snapshot,
                smoke_id=smoke_id,
                smoke_target=smoke_target,
            )
        )
        submitted.append(smoke_audit_id)
        smoke_audit_held = e98.audit_held(
            smoke_audit_id,
            name="e98r1-rlep-smoke-audit",
            expected=(
                f"Dependency=afterok:{smoke_id}",
                str(smoke_target),
                "--expected-terminal-step",
            ),
        )

        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            name = f"e98r1-{e81.DOMAIN_TAGS[domain][:6]}-s{seed}"
            job_id = e98.submit_held(
                gpu_command(
                    run,
                    cell["env"],
                    name=name,
                    dependency=smoke_audit_id,
                )
            )
            submitted.append(job_id)
            held = e98.audit_held(
                job_id,
                name=name,
                expected=(
                    f"Dependency=afterok:{smoke_audit_id}",
                    f"ReqNodeList={run['source_node']}",
                    f"OAT_ZERO_SEED={seed}",
                    "OAT_ZERO_VARIANT=rlep",
                    "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                    "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY=0",
                    "OAT_ZERO_NUM_PROMPT_EPOCH=8",
                ),
            )
            control = cell["control"]
            run_records.append(
                {
                    "domain": domain,
                    "arm": ARM,
                    "seed": seed,
                    "source_node": str(run["source_node"]),
                    "run_stamp": run_stamp(domain, seed),
                    "run_dir": str(cell["target"]),
                    "job_id": int(job_id),
                    "smoke_audit_dependency_job_id": int(smoke_audit_id),
                    "pool_root": str(cell["pool"]),
                    "paired_e78_control": {
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e98r1_sparse_rlep_dr_05b_jobs_v1",
            "cohort": "e98r1",
            "released": False,
            "model": "Qwen2.5-0.5B-Instruct",
            "protocol": str(PROTOCOL),
            "protocol_sha256": e81.digest(PROTOCOL),
            "launcher_sha256": e81.digest(Path(__file__)),
            "source_e98_ledger": str(SOURCE_LEDGER),
            "source_e98_ledger_sha256": e81.digest(SOURCE_LEDGER),
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "domains": list(e98.DOMAINS),
            "seeds": list(e98.SEEDS),
            "arms": [ARM],
            "passes": e98.PASSES,
            "train_rows": e98.TRAIN_ROWS,
            "target_steps": e98.TARGET_STEPS,
            "checkpoint_interval_steps": e98.CHECKPOINT_INTERVAL,
            "variant": e98.VARIANT,
            "rlep_replay_count_on_eligible_prompt": e98.REPLAY_COUNT,
            "ineligible_prompt_fallback": "unchanged_16_row_drgrpo_update",
            "pool_audits": audit_records,
            "objective": (
                "sparse_prompt_matched_RLEP_16_plus_2_on_eligible_"
                "else_E78_DrGRPO_16"
            ),
            "scientific_difference": (
                "against E78 control: prior-policy-supported frequency-preserving "
                "success replay only on prompts with two frozen verified trajectories"
            ),
            "smoke": {
                "scientific": False,
                "job_id": int(smoke_id),
                "audit_job_id": int(smoke_audit_id),
                "run_dir": str(smoke_target),
                "target_steps": 32,
                "held_scheduler_record": smoke_held,
                "held_audit_scheduler_record": smoke_audit_held,
            },
            "runs": run_records,
        }
        e81.atomic_json(LEDGER, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e81.atomic_json(LEDGER, payload)
    except Exception:
        e98.cancel(submitted)
        raise

    print(
        f"[e98r1] released smoke {smoke_id}, audit {smoke_audit_id}, "
        f"and {len(run_records)} dependent scientific cells at nice=0"
    )
    print(f"[e98r1] ledger {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
