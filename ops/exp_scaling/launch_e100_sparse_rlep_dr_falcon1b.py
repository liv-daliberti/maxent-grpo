#!/usr/bin/env python3
"""Submit E100 Falcon sparse-RLEP collection, gates, and science cells."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_decoding_frontier as e72  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e82_falcon_semantic_maxent_verified_replay as e82  # noqa: E402
import launch_e98_rlep_05b as e98  # noqa: E402
import launch_e97_ucpo_05b as e97  # noqa: E402


DOMAINS = e79.DOMAINS
SEEDS = e79.SEEDS
PASSES = e79.PASSES
TRAIN_ROWS = e79.TRAIN_ROWS
CHECKPOINT_INTERVAL = e79.CHECKPOINT_INTERVAL
TARGET_STEPS = e79.TARGET_STEPS
ARM = "rlep_dr_sparse"
VARIANT = "rlep"
REPLAY_COUNT = 2
COLLECTION_K = 16
COLLECTION_DRAWS = 4
COLLECTION_TEMPERATURE = 0.7
COLLECTION_TOP_P = 0.95
LEDGER = ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e100_sparse_rlep_dr_falcon1b_20260814.md"
PAIR_LEDGER = ROOT / e82.PAIR_LEDGER
SOURCE_MANIFEST = ROOT / e79.SOURCE_MANIFEST
SMOKE_AUDIT = "ops/exp_scaling/audit_e100_sparse_rlep_smoke.py"
SHARED_SMOKE_AUDIT = "ops/exp_scaling/audit_e98r1_sparse_rlep_smoke.py"
SNAPSHOT_PREFIX = "e100_sparse_rlep_falcon1b"
PATCHED_FILES = tuple(
    sorted(
        set(e97.PATCHED_FILES)
        | {
            "ops/exp_scaling/audit_e98_rlep_pool.py",
            SHARED_SMOKE_AUDIT,
            SMOKE_AUDIT,
        }
    )
)
SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/args.py", "rlep_sparse_fallback: bool"),
    ("src/oat_drgrpo/learner/grpo.py", "rlep_replay_eligible"),
    ("src/oat_drgrpo/rlep.py", "allow_sparse"),
    ("ops/run_experiment.sh", "OAT_ZERO_RLEP_SPARSE_FALLBACK"),
    ("ops/train.sh", "--rlep-sparse-fallback"),
    ("ops/exp_scaling/audit_e98_rlep_pool.py", "--allow-sparse"),
    (SMOKE_AUDIT, "E100_SMOKE_COMPLETE.json"),
)


def verify_snapshot(snapshot: Path) -> None:
    missing = [
        f"{name}: {needle!r}"
        for name, needle in SNAPSHOT_REQUIREMENTS
        if needle not in (snapshot / name).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit("E100 snapshot lacks sparse-RLEP contracts:\n  " + "\n  ".join(missing))


def pool_seed(domain: str, seed: int) -> int:
    return 1_000_000 + 100 * DOMAINS.index(domain) + int(seed)


def pool_path(domain: str, seed: int) -> Path:
    return ROOT / "var/data/e100_rlep_pools" / domain / f"s{seed}"


def run_stamp(domain: str, seed: int) -> str:
    return f"e100_sparse_rlep_falcon_{e79.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(domain: str, seed: int) -> Path:
    return ROOT / "var/data" / f"xdr_{e79.MODEL_TAG}_rlep_{run_stamp(domain, seed)}"


def collection_env(
    run: dict[str, Any],
    *,
    snapshot: Path,
    seed_export: Path,
    collection_data: Path,
    target: Path,
) -> dict[str, str]:
    domain, seed = str(run["domain"]), int(run["seed"])
    job = {
        "source": run,
        "k": COLLECTION_K,
        "temperature": COLLECTION_TEMPERATURE,
        "top_p": COLLECTION_TOP_P,
        "draws": COLLECTION_DRAWS,
        "seed": seed,
    }
    env = e72.build_export_vars(ROOT, job, target)
    env.update(
        {
            "OAT_ZERO_PRETRAIN": str(seed_export),
            "OAT_ZERO_EVAL_DATA": str(collection_data),
            "OAT_ZERO_TEST_SPLIT": "multi_answer",
            "OAT_ZERO_EVAL_MODE_COVERAGE_K": str(COLLECTION_K),
            "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE": repr(COLLECTION_TEMPERATURE),
            "OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P": repr(COLLECTION_TOP_P),
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(COLLECTION_DRAWS),
            "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": str(pool_seed(domain, seed)),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "RUN_STAMP": f"e100_rlep_pool_{e79.DOMAIN_TAGS[domain]}_s{seed}",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(
        e79.task_surface(
            domain, str(run["inherited_eval_config"]["prompt_template"])
        )
    )
    return env


def training_env(
    run: dict[str, Any],
    *,
    snapshot: Path,
    falcon_root: Path,
    pool: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = e79.build_env(ROOT, run, "control", snapshot, falcon_root)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(e98.training_objective(pool))
    env.update(
        {
            "OAT_ZERO_RLEP_SPARSE_FALLBACK": "1",
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        }
    )
    return env, target


def smoke_env(
    run: dict[str, Any],
    *,
    snapshot: Path,
    falcon_root: Path,
    pool: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = training_env(run, snapshot=snapshot, falcon_root=falcon_root, pool=pool)
    target = ROOT / "var/data/e100_sparse_rlep_smoke_graph_s55"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": "e100_sparse_rlep_smoke_graph_s55",
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


def gpu_command(
    run: dict[str, Any],
    env: dict[str, str],
    *,
    name: str,
    dependency: str = "",
) -> list[str]:
    command = e79.sbatch_command(ROOT, run, "control", env)
    command = [
        f"--job-name={name}" if item.startswith("--job-name=") else item
        for item in command
    ]
    if dependency:
        command.insert(-1, f"--dependency=afterok:{dependency}")
    return command


def pool_audit_command(
    run: dict[str, Any], *, snapshot: Path, pool: Path, dependency: str
) -> list[str]:
    domain, seed = str(run["domain"]), int(run["seed"])
    node, _ = e79.placement(domain, seed)
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = snapshot / "ops/exp_scaling/audit_e98_rlep_pool.py"
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name=e100-{e79.DOMAIN_TAGS[domain][:6]}-audit-s{seed}",
        f"--dependency=afterok:{dependency}",
        f"--export=ALL,PYTHONPATH={snapshot / 'src'}",
        "--partition=cs",
        "--account=allcs",
        f"--nodelist={node}",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:15:00",
        "--nice=100",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        (
            f"{python} {script} --pool-root {pool} "
            f"--expected-prompts {TRAIN_ROWS} --allow-sparse"
        ),
    ]


def smoke_audit_command(*, snapshot: Path, smoke_id: str, target: Path) -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = snapshot / SMOKE_AUDIT
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--job-name=e100-rlep-smoke-audit",
        f"--dependency=afterok:{smoke_id}",
        f"--export=ALL,PYTHONPATH={snapshot / 'src'}",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:15:00",
        "--nice=100",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        f"{python} {script} --run-root {target} --expected-terminal-step 32",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    for required in (PROTOCOL, SOURCE_MANIFEST, PAIR_LEDGER):
        if not required.is_file():
            raise SystemExit(f"required E100 input is absent: {required}")
    if args.submit and LEDGER.exists():
        raise SystemExit(f"refusing duplicate E100 submission: {LEDGER}")

    pair_payload = e82.pair_ledger(ROOT)
    pairs = e82.pair_index(pair_payload)
    runs = e79.references(ROOT)
    falcon_root = e79.model_root(ROOT)
    base_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(str(pair_payload["snapshot_root"])).resolve()
    )
    snapshot, patched = e81.ensure_paired_snapshot(
        ROOT,
        base_snapshot,
        prefix=SNAPSHOT_PREFIX,
        patched_files=PATCHED_FILES,
    )
    verify_snapshot(snapshot)

    collection_views: dict[str, tuple[Path, str]] = {}
    for domain in DOMAINS:
        template = next(run for run in runs if str(run["domain"]) == domain)
        collection_views[domain] = e98.materialize_collection_data(ROOT, template)

    cells: list[dict[str, Any]] = []
    for run in runs:
        domain, seed = str(run["domain"]), int(run["seed"])
        control = pairs[(domain, seed)]["control"]
        node, gpu = e79.placement(domain, seed)
        if (str(control["node"]), str(control["gpu"])) != (node, gpu):
            raise SystemExit(f"{domain}/s{seed}: E79 control placement drift")
        seed_export = e98.terminal_control_export(control)
        pool = pool_path(domain, seed)
        target = save_path(domain, seed)
        if pool.exists() or target.exists():
            raise SystemExit(f"refusing to overwrite E100 artifacts for {domain}/s{seed}")
        collection_data, data_hash = collection_views[domain]
        collect_env = collection_env(
            run,
            snapshot=snapshot,
            seed_export=seed_export,
            collection_data=collection_data,
            target=pool,
        )
        train_env, _ = training_env(
            run, snapshot=snapshot, falcon_root=falcon_root, pool=pool
        )
        cells.append(
            {
                "run": run,
                "control": control,
                "seed_export": seed_export,
                "pool": pool,
                "target": target,
                "collection_data": collection_data,
                "collection_data_sha256": data_hash,
                "collect_env": collect_env,
                "train_env": train_env,
            }
        )

    smoke_cell = next(
        cell for cell in cells
        if str(cell["run"]["domain"]) == "graph_coloring"
        and int(cell["run"]["seed"]) == 55
    )
    smoke_vars, smoke_target = smoke_env(
        smoke_cell["run"],
        snapshot=snapshot,
        falcon_root=falcon_root,
        pool=smoke_cell["pool"],
    )
    if smoke_target.exists():
        raise SystemExit(f"refusing to overwrite E100 smoke: {smoke_target}")

    if args.dry_run or not args.submit:
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            print(shlex.join(gpu_command(run, cell["collect_env"], name=f"e100-{e79.DOMAIN_TAGS[domain][:6]}-pool-s{seed}")))
            print(shlex.join(pool_audit_command(run, snapshot=snapshot, pool=cell["pool"], dependency="POOL_JOB_ID")))
        print(shlex.join(gpu_command(smoke_cell["run"], smoke_vars, name="e100-rlep-smoke", dependency="GRAPH55_AUDIT_ID")))
        print("<smoke audit depends on smoke>")
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            print(shlex.join(gpu_command(run, cell["train_env"], name=f"e100-{e79.DOMAIN_TAGS[domain][:6]}-rlep-s{seed}", dependency="POOL_AUDIT_ID:SMOKE_AUDIT_ID")))
        print(f"[e100] pools=25 audits=25 smoke=1 smoke_audit=1 science=25 snapshot={snapshot} patched={patched}")
        return 0

    submitted: list[str] = []
    pool_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    try:
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            collect_name = f"e100-{e79.DOMAIN_TAGS[domain][:6]}-pool-s{seed}"
            collect_id = e98.submit_held(gpu_command(run, cell["collect_env"], name=collect_name))
            submitted.append(collect_id)
            collect_record = e98.audit_held(
                collect_id,
                name=collect_name,
                expected=(
                    f"ReqNodeList={e79.placement(domain, seed)[0]}",
                    f"OAT_ZERO_PRETRAIN={cell['seed_export']}",
                    f"OAT_ZERO_PROMPT_TEMPLATE={e79.task_surface(domain, str(run['inherited_eval_config']['prompt_template']))['OAT_ZERO_PROMPT_TEMPLATE']}",
                    "OAT_ZERO_EVAL_ONLY=1",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_K=16",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=0.7",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P=0.95",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
                    "OAT_ZERO_TEST_SPLIT=multi_answer",
                ),
            )
            audit_name = f"e100-{e79.DOMAIN_TAGS[domain][:6]}-audit-s{seed}"
            audit_id = e98.submit_held(
                pool_audit_command(
                    run, snapshot=snapshot, pool=cell["pool"], dependency=collect_id
                )
            )
            submitted.append(audit_id)
            audit_record = e98.audit_held(
                audit_id,
                name=audit_name,
                expected=(f"Dependency=afterok:{collect_id}", str(cell["pool"]), "--allow-sparse"),
            )
            cell["collect_id"] = collect_id
            cell["audit_id"] = audit_id
            pool_records.append(
                {
                    "domain": domain,
                    "seed": seed,
                    "node": e79.placement(domain, seed)[0],
                    "gpu": e79.placement(domain, seed)[1],
                    "seed_policy_export": str(cell["seed_export"]),
                    "collection_data": str(cell["collection_data"]),
                    "collection_data_sha256": cell["collection_data_sha256"],
                    "collection_seed": pool_seed(domain, seed),
                    "pool_root": str(cell["pool"]),
                    "collection_job_id": int(collect_id),
                    "audit_job_id": int(audit_id),
                    "held_collection_scheduler_record": collect_record,
                    "held_audit_scheduler_record": audit_record,
                }
            )

        graph_audit_id = str(smoke_cell["audit_id"])
        smoke_id = e98.submit_held(
            gpu_command(
                smoke_cell["run"],
                smoke_vars,
                name="e100-rlep-smoke",
                dependency=graph_audit_id,
            )
        )
        submitted.append(smoke_id)
        smoke_record = e98.audit_held(
            smoke_id,
            name="e100-rlep-smoke",
            expected=(
                f"Dependency=afterok:{graph_audit_id}",
                "OAT_ZERO_VARIANT=rlep",
                "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
                "OAT_ZERO_MAX_TRAIN=32",
            ),
        )
        smoke_audit_id = e98.submit_held(
            smoke_audit_command(snapshot=snapshot, smoke_id=smoke_id, target=smoke_target)
        )
        submitted.append(smoke_audit_id)
        smoke_audit_record = e98.audit_held(
            smoke_audit_id,
            name="e100-rlep-smoke-audit",
            expected=(f"Dependency=afterok:{smoke_id}", str(smoke_target), "--expected-terminal-step"),
        )

        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            name = f"e100-{e79.DOMAIN_TAGS[domain][:6]}-rlep-s{seed}"
            dependency = f"{cell['audit_id']}:{smoke_audit_id}"
            job_id = e98.submit_held(
                gpu_command(run, cell["train_env"], name=name, dependency=dependency)
            )
            submitted.append(job_id)
            held = e98.audit_held(
                job_id,
                name=name,
                expected=(
                    "Dependency=afterok:",
                    str(cell["audit_id"]),
                    smoke_audit_id,
                    f"ReqNodeList={e79.placement(domain, seed)[0]}",
                    f"OAT_ZERO_SEED={seed}",
                    "OAT_ZERO_VARIANT=rlep",
                    f"OAT_ZERO_RLEP_EXPERIENCE_ROOT={cell['pool']}",
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
                    "node": e79.placement(domain, seed)[0],
                    "gpu": e79.placement(domain, seed)[1],
                    "run_stamp": run_stamp(domain, seed),
                    "run_dir": str(cell["target"]),
                    "job_id": int(job_id),
                    "pool_audit_dependency_job_id": int(cell["audit_id"]),
                    "smoke_audit_dependency_job_id": int(smoke_audit_id),
                    "pool_root": str(cell["pool"]),
                    "paired_e79_control": {
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e100_sparse_rlep_dr_falcon1b_jobs_v1",
            "cohort": "e100",
            "released": False,
            "model": "Falcon3-1B-Instruct",
            "model_revision": e79.MODEL_REVISION,
            "protocol": str(PROTOCOL),
            "protocol_sha256": e81.digest(PROTOCOL),
            "launcher_sha256": e81.digest(Path(__file__)),
            "source_manifest": str(SOURCE_MANIFEST),
            "source_manifest_sha256": e81.digest(SOURCE_MANIFEST),
            "pair_ledger": str(PAIR_LEDGER),
            "pair_ledger_sha256": e81.digest(PAIR_LEDGER),
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "domains": list(DOMAINS),
            "seeds": list(SEEDS),
            "arms": [ARM],
            "inherited_arms": ["control"],
            "passes": PASSES,
            "train_rows": TRAIN_ROWS,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "variant": VARIANT,
            "rlep_replay_count_on_eligible_prompt": REPLAY_COUNT,
            "ineligible_prompt_fallback": "unchanged_16_row_drgrpo_update",
            "collection": {
                "candidates_per_prompt": COLLECTION_K * COLLECTION_DRAWS,
                "k": COLLECTION_K,
                "draws": COLLECTION_DRAWS,
                "temperature": COLLECTION_TEMPERATURE,
                "top_p": COLLECTION_TOP_P,
                "minimum_verified_on_eligible_prompt": 2,
                "preserve_trajectory_frequency": True,
                "records": pool_records,
            },
            "objective": "sparse_prompt_matched_RLEP_16_plus_2_else_E79_DrGRPO_16",
            "scientific_difference": "against E79 control: two frozen prior-policy successes on eligible prompts only",
            "smoke": {
                "scientific": False,
                "job_id": int(smoke_id),
                "audit_job_id": int(smoke_audit_id),
                "run_dir": str(smoke_target),
                "target_steps": 32,
                "pool_audit_dependency_job_id": int(graph_audit_id),
                "held_scheduler_record": smoke_record,
                "held_audit_scheduler_record": smoke_audit_record,
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
        f"[e100] released 25 pools, 25 audits, smoke {smoke_id}, "
        f"smoke audit {smoke_audit_id}, and {len(run_records)} science cells"
    )
    print(f"[e100] ledger {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
