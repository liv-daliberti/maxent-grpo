#!/usr/bin/env python3
"""Submit E98 RLEP-Dr collection, audit, smoke, and scientific jobs.

Each scientific cell is protected by two scheduler dependencies: its own
seed-specific experience pool must pass the fail-closed CPU audit, and the
Graph/s43 RLEP learner smoke must finish successfully. All jobs are first held,
their immutable exports are audited, and only then are they released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_decoding_frontier as e72  # noqa: E402
import launch_e78_verified_replay_only_05b as e78  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


DOMAINS = ("graph_coloring", "python_factors", "pantry_plan")
SEEDS = e81.SEEDS
PASSES = e81.PASSES
TRAIN_ROWS = e81.TRAIN_ROWS
CHECKPOINT_INTERVAL = e81.CHECKPOINT_INTERVAL
TARGET_STEPS = e81.TARGET_STEPS
ARM = "rlep_dr"
VARIANT = "rlep"
REPLAY_COUNT = 2
COLLECTION_K = 16
COLLECTION_DRAWS = 4
COLLECTION_TEMPERATURE = 0.7
COLLECTION_TOP_P = 0.95
LEDGER = "var/artifacts/e98_rlep_dr_05b_jobs.json"
PROTOCOL = "paper/preregistration/e98_rlep_dr_05b_20260812.md"
PAIR_LEDGER = e81.PAIR_LEDGER
SOURCE_MANIFEST = e81.SOURCE_MANIFEST
SNAPSHOT_PREFIX = "e98_rlep_dr"
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "src/oat_drgrpo/learner/grpo.py",
    "src/oat_drgrpo/rlep.py",
    "src/oat_drgrpo/ucpo.py",  # imported unconditionally by grpo.py
    "ops/run_experiment.sh",
    "ops/train.sh",
    "ops/exp_scaling/audit_e98_rlep_pool.py",
)
SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/args.py", "rlep_experience_root:"),
    ("src/oat_drgrpo/learner/grpo.py", "rlep_mixed_advantages"),
    ("src/oat_drgrpo/rlep.py", "class RLEPExperiencePool"),
    ("ops/run_experiment.sh", "rlep)"),
    ("ops/train.sh", "--rlep-experience-root"),
    ("ops/exp_scaling/audit_e98_rlep_pool.py", "RLEP_POOL_COMPLETE.json"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def selected_references(root: Path) -> list[dict[str, Any]]:
    runs = [r for r in e81.references(root) if str(r["domain"]) in DOMAINS]
    expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    found = {(str(r["domain"]), int(r["seed"])) for r in runs}
    if found != expected or len(runs) != len(expected):
        raise SystemExit("E98 source manifest does not cover exactly 15 cells")
    return runs


def selected_pairs(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    pairs = e81.pair_index(e81.pair_ledger(root))
    return {key: value for key, value in pairs.items() if key[0] in DOMAINS}


def verify_snapshot(snapshot: Path) -> None:
    missing = [
        f"{name}: {needle!r}"
        for name, needle in SNAPSHOT_REQUIREMENTS
        if needle not in (snapshot / name).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit("E98 snapshot is not RLEP-capable:\n  " + "\n  ".join(missing))


def dataset_identity(dataset: Any) -> str:
    identity = hashlib.sha256()
    for row in dataset:
        identity.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        identity.update(b"\n")
    return identity.hexdigest()


def materialize_collection_data(root: Path, run: dict[str, Any]) -> tuple[Path, str]:
    """Expose the exact training rows under the evaluator's registered split."""

    domain = str(run["domain"])
    source = Path(run["inherited_eval_config"]["prompt_data"])
    target = root / "var/data/e98_rlep_collection_prompts" / domain
    source_data = load_from_disk(str(source))["train"]
    if len(source_data) != TRAIN_ROWS:
        raise SystemExit(f"{domain}: expected {TRAIN_ROWS} source training prompts")
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{domain}.", dir=target.parent))
        try:
            DatasetDict({"multi_answer": source_data}).save_to_disk(str(temporary))
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    target_dict = load_from_disk(str(target))
    if set(target_dict) != {"multi_answer"} or len(target_dict["multi_answer"]) != TRAIN_ROWS:
        raise SystemExit(f"{domain}: invalid E98 collection view")
    source_hash = dataset_identity(source_data)
    target_hash = dataset_identity(target_dict["multi_answer"])
    if source_hash != target_hash:
        raise SystemExit(f"{domain}: collection view changes training-row content")
    return target, source_hash


def terminal_control_export(control: dict[str, Any]) -> Path:
    run_dir = Path(str(control["run_dir"])).resolve()
    receipt = run_dir / "TRAINING_COMPLETE.json"
    if not receipt.is_file():
        raise SystemExit(f"RLEP seed policy has no completion receipt: {receipt}")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    export = Path(str(payload.get("terminal_export", ""))).resolve()
    if run_dir not in export.parents:
        raise SystemExit(f"RLEP seed-policy export escapes its E78 run: {export}")
    for required in ("config.json", "model.safetensors", "tokenizer_config.json"):
        if not (export / required).is_file():
            raise SystemExit(f"incomplete RLEP seed-policy export: {export / required}")
    if int(payload.get("terminal_step", -1)) < TARGET_STEPS:
        raise SystemExit(f"RLEP seed policy is not terminal: {receipt}")
    return export


def pool_seed(domain: str, seed: int) -> int:
    return 980000 + 100 * DOMAINS.index(domain) + int(seed)


def pool_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data/e98_rlep_pools" / domain / f"s{seed}"


def run_stamp(domain: str, seed: int) -> str:
    return f"e98_rlep_dr_{e81.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{e81.MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def collection_env(
    root: Path,
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
    env = e72.build_export_vars(root, job, target)
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
            "RUN_STAMP": f"e98_rlep_pool_{e81.DOMAIN_TAGS[domain]}_s{seed}",
        }
    )
    return env


def training_objective(pool: Path) -> dict[str, str]:
    result = dict(e78.fixed_objective("control"))
    result.update(
        {
            "OAT_ZERO_VARIANT": VARIANT,
            "OAT_ZERO_UCPO_TAU": "0.0",
            "OAT_ZERO_RLEP_EXPERIENCE_ROOT": str(pool),
            "OAT_ZERO_RLEP_REPLAY_COUNT": str(REPLAY_COUNT),
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "0",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "0",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
            "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING": "0",
        }
    )
    return result


def training_env(
    root: Path,
    run: dict[str, Any],
    *,
    snapshot: Path,
    pool: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = e81.build_env(root, run, snapshot)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(training_objective(pool))
    return env, target


def smoke_env(
    root: Path, run: dict[str, Any], *, snapshot: Path, pool: Path
) -> tuple[dict[str, str], Path]:
    env, _ = training_env(root, run, snapshot=snapshot, pool=pool)
    target = root / "var/data/e98_rlep_smoke_graph_s43"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": "e98_rlep_smoke_graph_s43",
            "OAT_ZERO_MAX_TRAIN": "32",
            "OAT_ZERO_MAX_QUERIES": "32",
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
    root: Path,
    run: dict[str, Any],
    env: dict[str, str],
    *,
    name: str,
    dependency: str = "",
) -> list[str]:
    node = str(run["source_node"])
    place = e81.placement(node)
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{','.join(f'{k}={v}' for k, v in env.items())}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={node}",
        f"--gres={place['gres']}",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=1-12:00:00",
        "--nice=100",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(root / "ops/slurm/train_node302.slurm"))
    return command


def audit_command(
    root: Path,
    run: dict[str, Any],
    *,
    snapshot: Path,
    pool: Path,
    dependency: str,
) -> list[str]:
    domain, seed = str(run["domain"]), int(run["seed"])
    node = str(run["source_node"])
    place = e81.placement(node)
    script = snapshot / "ops/exp_scaling/audit_e98_rlep_pool.py"
    python = root / "var/seed_paper_eval/paper310/bin/python"
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name=e98-{e81.DOMAIN_TAGS[domain][:6]}-audit-s{seed}",
        f"--dependency=afterok:{dependency}",
        f"--export=ALL,PYTHONPATH={snapshot / 'src'}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={node}",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:15:00",
        "--nice=100",
        f"--output={root / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={root / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        f"{python} {script} --pool-root {pool} --expected-prompts {TRAIN_ROWS}",
    ]


def submit_held(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch response: {result.stdout!r}")
    return job_id


def audit_held(job_id: str, *, name: str, expected: tuple[str, ...]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E98 job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = ("JobState=PENDING", "Reason=JobHeldUser", f"JobName={name}") + expected
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held E98 job {job_id} lacks {missing}")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    protocol, ledger = root / PROTOCOL, root / LEDGER
    for required in (protocol, root / SOURCE_MANIFEST, root / PAIR_LEDGER):
        if not required.is_file():
            raise SystemExit(f"required E98 input is absent: {required}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E98 submission: {ledger}")

    pairs = selected_pairs(root)
    runs = selected_references(root)
    base_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(e81.pair_ledger(root)["snapshot_root"])
    )
    snapshot, patched = e81.ensure_paired_snapshot(
        root,
        base_snapshot,
        prefix=SNAPSHOT_PREFIX,
        patched_files=PATCHED_FILES,
    )
    verify_snapshot(snapshot)

    collection_views: dict[str, tuple[Path, str]] = {}
    for domain in DOMAINS:
        template = next(r for r in runs if str(r["domain"]) == domain)
        collection_views[domain] = materialize_collection_data(root, template)

    cells: list[dict[str, Any]] = []
    for run in runs:
        domain, seed = str(run["domain"]), int(run["seed"])
        control = pairs[(domain, seed)]["control"]
        if str(control["source_node"]) != str(run["source_node"]):
            raise SystemExit(f"{domain}/s{seed}: E78 placement drift")
        seed_export = terminal_control_export(control)
        pool = pool_path(root, domain, seed)
        target = save_path(root, domain, seed)
        if pool.exists() or target.exists():
            raise SystemExit(f"refusing to overwrite E98 artifacts for {domain}/s{seed}")
        collection_data, data_hash = collection_views[domain]
        collect = collection_env(
            root,
            run,
            snapshot=snapshot,
            seed_export=seed_export,
            collection_data=collection_data,
            target=pool,
        )
        train, _ = training_env(root, run, snapshot=snapshot, pool=pool)
        cells.append(
            {
                "run": run,
                "control": control,
                "seed_export": seed_export,
                "pool": pool,
                "target": target,
                "collection_data": collection_data,
                "collection_data_sha256": data_hash,
                "collect_env": collect,
                "train_env": train,
            }
        )

    smoke_cell = next(
        cell
        for cell in cells
        if cell["run"]["domain"] == "graph_coloring" and cell["run"]["seed"] == 43
    )
    smoke_vars, smoke_target = smoke_env(
        root, smoke_cell["run"], snapshot=snapshot, pool=smoke_cell["pool"]
    )
    if smoke_target.exists():
        raise SystemExit(f"refusing to overwrite E98 smoke: {smoke_target}")

    if args.dry_run or not args.submit:
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            collect_name = f"e98-{e81.DOMAIN_TAGS[domain][:6]}-pool-s{seed}"
            print(shlex.join(gpu_command(root, run, cell["collect_env"], name=collect_name)))
            print(shlex.join(audit_command(root, run, snapshot=snapshot, pool=cell["pool"], dependency="POOL_JOB_ID")))
        print(shlex.join(gpu_command(root, smoke_cell["run"], smoke_vars, name="e98-rlep-smoke", dependency="GRAPH43_AUDIT_ID")))
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            name = f"e98-{e81.DOMAIN_TAGS[domain][:6]}-rlep-s{seed}"
            print(shlex.join(gpu_command(root, run, cell["train_env"], name=name, dependency="POOL_AUDIT_ID:SMOKE_JOB_ID")))
        print(f"[e98] pools=15 audits=15 smoke=1 scientific=15 snapshot={snapshot} patched={patched}")
        return 0

    submitted: list[str] = []
    pool_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    try:
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            collect_name = f"e98-{e81.DOMAIN_TAGS[domain][:6]}-pool-s{seed}"
            collect_id = submit_held(
                gpu_command(root, run, cell["collect_env"], name=collect_name)
            )
            submitted.append(collect_id)
            collect_record = audit_held(
                collect_id,
                name=collect_name,
                expected=(
                    f"ReqNodeList={run['source_node']}",
                    f"OAT_ZERO_PRETRAIN={cell['seed_export']}",
                    f"OAT_ZERO_EVAL_DATA={cell['collection_data']}",
                    "OAT_ZERO_EVAL_ONLY=1",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_K=16",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=0.7",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P=0.95",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
                    "OAT_ZERO_TEST_SPLIT=multi_answer",
                ),
            )
            audit_name = f"e98-{e81.DOMAIN_TAGS[domain][:6]}-audit-s{seed}"
            audit_id = submit_held(
                audit_command(
                    root,
                    run,
                    snapshot=snapshot,
                    pool=cell["pool"],
                    dependency=collect_id,
                )
            )
            submitted.append(audit_id)
            audit_record = audit_held(
                audit_id,
                name=audit_name,
                expected=(f"Dependency=afterok:{collect_id}", str(cell["pool"])),
            )
            cell["collect_id"] = collect_id
            cell["audit_id"] = audit_id
            pool_records.append(
                {
                    "domain": domain,
                    "seed": seed,
                    "source_node": str(run["source_node"]),
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
        smoke_id = submit_held(
            gpu_command(
                root,
                smoke_cell["run"],
                smoke_vars,
                name="e98-rlep-smoke",
                dependency=graph_audit_id,
            )
        )
        submitted.append(smoke_id)
        smoke_record = audit_held(
            smoke_id,
            name="e98-rlep-smoke",
            expected=(
                f"Dependency=afterok:{graph_audit_id}",
                "OAT_ZERO_VARIANT=rlep",
                f"OAT_ZERO_RLEP_EXPERIENCE_ROOT={smoke_cell['pool']}",
                "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                "OAT_ZERO_MAX_QUERIES=32",
            ),
        )

        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            name = f"e98-{e81.DOMAIN_TAGS[domain][:6]}-rlep-s{seed}"
            dependency = f"{cell['audit_id']}:{smoke_id}"
            job_id = submit_held(
                gpu_command(root, run, cell["train_env"], name=name, dependency=dependency)
            )
            submitted.append(job_id)
            held = audit_held(
                job_id,
                name=name,
                expected=(
                    "Dependency=afterok:",
                    str(cell["audit_id"]),
                    smoke_id,
                    f"ReqNodeList={run['source_node']}",
                    f"OAT_ZERO_SEED={seed}",
                    "OAT_ZERO_VARIANT=rlep",
                    f"OAT_ZERO_RLEP_EXPERIENCE_ROOT={cell['pool']}",
                    "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY=0",
                    "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=0",
                    "OAT_ZERO_NUM_PROMPT_EPOCH=8",
                    "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
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
                    "pool_audit_dependency_job_id": int(cell["audit_id"]),
                    "smoke_dependency_job_id": int(smoke_id),
                    "paired_e78_control": {
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e98_rlep_dr_05b_jobs_v1",
            "cohort": "e98",
            "released": False,
            "model": "Qwen2.5-0.5B-Instruct",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "source_manifest": str(root / SOURCE_MANIFEST),
            "source_manifest_sha256": e81.digest(root / SOURCE_MANIFEST),
            "pair_ledger": str(root / PAIR_LEDGER),
            "pair_ledger_sha256": e81.digest(root / PAIR_LEDGER),
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
            "registered_passes": [i / 2 for i in range(2 * PASSES + 1)],
            "variant": VARIANT,
            "rlep_replay_count": REPLAY_COUNT,
            "collection": {
                "candidates_per_prompt": COLLECTION_K * COLLECTION_DRAWS,
                "k": COLLECTION_K,
                "draws": COLLECTION_DRAWS,
                "temperature": COLLECTION_TEMPERATURE,
                "top_p": COLLECTION_TOP_P,
                "minimum_verified_per_prompt": 2,
                "preserve_trajectory_frequency": True,
                "records": pool_records,
            },
            "objective": "RLEP_16_fresh_plus_2_offline_successes_with_DrGRPO_baseline",
            "scientific_difference": "against E78 control: offline frequency-preserving success replay and its two extra rows",
            "smoke": {
                "scientific": False,
                "job_id": int(smoke_id),
                "run_dir": str(smoke_target),
                "max_queries": 32,
                "pool_audit_dependency_job_id": int(graph_audit_id),
                "held_scheduler_record": smoke_record,
            },
            "runs": run_records,
        }
        e81.atomic_json(ledger, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        raise
    print(
        f"[e98] released 15 pools, 15 audits, smoke {smoke_id}, "
        f"and {len(run_records)} dependent scientific cells"
    )
    print(f"[e98] ledger {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
