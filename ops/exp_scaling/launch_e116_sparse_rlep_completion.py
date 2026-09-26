#!/usr/bin/env python3
"""Submit E116 sparse RLEP-Dr completion at Qwen 0.5B and 3B."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_comparator_completion as shared  # noqa: E402
import launch_e97_ucpo_05b as e97  # noqa: E402
import launch_e98_rlep_05b as e98  # noqa: E402


ROOT = shared.ROOT
PROTOCOL = ROOT / "paper/preregistration/e116_sparse_rlep_dr_completion_20260819.md"
RLEP_REFERENCE = ROOT / "var/artifacts/e98r1_pantry_memory_recovery_jobs.json"
VARIANT = "rlep"
ARM = "rlep_dr_sparse"
REPLAY_COUNT = 2
COLLECTION_K = 16
COLLECTION_DRAWS = 4
COLLECTION_TEMPERATURE = 0.7
COLLECTION_TOP_P = 0.95
PATCHED_FILES = tuple(
    sorted(
        set(e97.PATCHED_FILES)
        | {
            "src/oat_drgrpo/canonical_actions.py",
            "src/oat_drgrpo/pantry_support_action.py",
            "ops/exp_scaling/audit_e98_rlep_pool.py",
            "ops/exp_scaling/audit_e98r1_sparse_rlep_smoke.py",
        }
    )
)
SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/args.py", "rlep_sparse_fallback: bool"),
    ("src/oat_drgrpo/learner/grpo.py", "rlep_replay_eligible"),
    ("src/oat_drgrpo/rlep.py", "allow_sparse"),
    ("src/oat_drgrpo/pantry_support_action.py", "pantry_support_mask_from_allocation"),
    ("src/oat_drgrpo/canonical_actions.py", "canonical_action_code_from_verified_response"),
    ("ops/run_experiment.sh", "OAT_ZERO_RLEP_SPARSE_FALLBACK"),
    ("ops/train.sh", "--rlep-sparse-fallback"),
    ("ops/exp_scaling/audit_e98_rlep_pool.py", "--allow-sparse"),
    ("ops/exp_scaling/audit_e98r1_sparse_rlep_smoke.py", "replay_dose_pairs"),
)


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    model_revision: str
    parent: Path
    ledger: Path
    domains: tuple[str, ...]
    seeds: tuple[int, ...]
    model_tag: str


FAMILIES = (
    Family(
        "qwen05b",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
        ROOT / "var/artifacts/e116_sparse_rlep_qwen05b_domain_extension_jobs.json",
        ("countdown", "mathir"),
        (43, 44, 45, 46, 47),
        "qwen25_05b_instruct",
    ),
    Family(
        "qwen3b",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
        ROOT / "var/artifacts/e116_sparse_rlep_qwen3b_jobs.json",
        ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"),
        (70, 71, 72, 73, 74),
        "qwen25_3b_instruct",
    ),
)


def domain_tag(domain: str) -> str:
    return {
        "graph_coloring": "graph",
        "countdown": "count",
        "python_factors": "python",
        "mathir": "mathir",
        "pantry_plan": "pantry",
    }[domain]


def verify_snapshot(snapshot: Path) -> None:
    missing = []
    for relative, needle in SNAPSHOT_REQUIREMENTS:
        path = snapshot / relative
        if not path.is_file() or needle not in path.read_text(encoding="utf-8"):
            missing.append(f"{relative}: {needle!r}")
    if missing:
        raise SystemExit("E116 snapshot lacks contracts:\n  " + "\n  ".join(missing))


def pool_seed(family: Family, domain: str, seed: int) -> int:
    family_offset = 0 if family.key == "qwen05b" else 10_000
    all_domains = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
    return 1_160_000 + family_offset + 100 * all_domains.index(domain) + seed


def pool_path(family: Family, domain: str, seed: int) -> Path:
    return ROOT / "var/data/e116_rlep_pools" / family.key / domain / f"s{seed}"


def run_stamp(family: Family, domain: str, seed: int) -> str:
    return f"e116_{family.key}_{domain}_sparse_rlep_s{seed}"


def output_path(family: Family, domain: str, seed: int) -> Path:
    return ROOT / "var/data" / f"xdr_{family.model_tag}_{run_stamp(family, domain, seed)}"


def terminal_export(control: dict[str, Any]) -> tuple[Path, bool]:
    receipt = Path(str(control["run_dir"])) / "TRAINING_COMPLETE.json"
    if receipt.is_file():
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        run_dir = Path(str(control["run_dir"])).resolve()
        export = Path(str(payload.get("terminal_export", ""))).resolve()
        if run_dir not in export.parents:
            raise SystemExit(f"RLEP seed-policy export escapes its run: {export}")
        for required in ("config.json", "tokenizer_config.json"):
            if not (export / required).is_file():
                raise SystemExit(f"incomplete RLEP seed-policy export: {export / required}")
        single = export / "model.safetensors"
        index = export / "model.safetensors.index.json"
        if not single.is_file():
            if not index.is_file():
                raise SystemExit(f"RLEP seed-policy export has no model weights: {export}")
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            shards = {str(name) for name in weight_map.values()}
            missing = sorted(name for name in shards if not (export / name).is_file())
            if not shards or missing:
                raise SystemExit(
                    f"incomplete sharded RLEP seed-policy export: missing={missing}"
                )
        if int(payload.get("terminal_step", -1)) < 3072:
            raise SystemExit(f"RLEP seed policy is not terminal: {receipt}")
        return export, True
    # E78/E80-R1 write the terminal export at update 3073. The collection job
    # depends on this exact parent job, so the path is consumed only after the
    # immutable control reports success.
    expected = (
        Path(str(control["run_dir"]))
        / f"debug_job{int(control['job_id'])}"
        / "saved_models/step_03073"
    )
    return expected, False


def collection_overrides(
    family: Family,
    run: dict[str, Any],
    *,
    snapshot: Path,
    seed_export: Path,
    collection_data: Path,
    pool: Path,
) -> dict[str, str]:
    domain, seed = str(run["domain"]), int(run["seed"])
    return {
        "OAT_ZERO_PRETRAIN": str(seed_export),
        "SAVE_PATH": str(pool),
        "RUN_STAMP": f"e116_{family.key}_{domain}_pool_s{seed}",
        "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        "OAT_ZERO_EVAL_ONLY": "1",
        "OAT_ZERO_EVAL_DATA": str(collection_data),
        "OAT_ZERO_TEST_SPLIT": "multi_answer",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K": str(COLLECTION_K),
        "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE": repr(COLLECTION_TEMPERATURE),
        "OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P": repr(COLLECTION_TOP_P),
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(COLLECTION_DRAWS),
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": str(pool_seed(family, domain, seed)),
        "OAT_ZERO_AUTO_RESUME": "0",
        "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def training_overrides(
    family: Family,
    run: dict[str, Any],
    *,
    snapshot: Path,
    pool: Path,
    target: Path,
) -> dict[str, str]:
    domain, seed = str(run["domain"]), int(run["seed"])
    return {
        "OAT_ZERO_VARIANT": VARIANT,
        "OAT_ZERO_UCPO_TAU": "0.0",
        "OAT_ZERO_RLEP_EXPERIENCE_ROOT": str(pool),
        "OAT_ZERO_RLEP_REPLAY_COUNT": str(REPLAY_COUNT),
        "OAT_ZERO_RLEP_SPARSE_FALLBACK": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
        "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING": "0",
        "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        "SAVE_PATH": str(target),
        "RUN_STAMP": run_stamp(family, domain, seed),
    }


def audit_pool_command(*, family: Family, domain: str, seed: int, snapshot: Path, pool: Path, dependency: str) -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = snapshot / "ops/exp_scaling/audit_e98_rlep_pool.py"
    command = f"{python} {script} --pool-root {pool} --expected-prompts 384 --allow-sparse"
    return shared.cpu_audit_command(
        name=f"e116-{family.key}-{domain_tag(domain)}-audit-s{seed}",
        dependency=dependency,
        snapshot=snapshot,
        command=command,
    )


def smoke_overrides(base: dict[str, str], family: Family) -> tuple[dict[str, str], Path]:
    target = ROOT / "var/data" / f"e116_sparse_rlep_{family.key}_smoke"
    values = dict(base)
    values.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": f"e116_sparse_rlep_{family.key}_smoke",
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
    return values, target


def smoke_audit_command(*, family: Family, snapshot: Path, target: Path, dependency: str) -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = snapshot / "ops/exp_scaling/audit_e98r1_sparse_rlep_smoke.py"
    command = f"{python} {script} --run-root {target} --expected-terminal-step 32"
    return shared.cpu_audit_command(
        name=f"e116-{family.key}-smoke-audit",
        dependency=dependency,
        snapshot=snapshot,
        command=command,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--family", choices=("all", *(family.key for family in FAMILIES)), default="all")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    selected = list(FAMILIES) if args.family == "all" else [next(f for f in FAMILIES if f.key == args.family)]
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing E116 protocol: {PROTOCOL}")
    reference = shared.read_ledger(RLEP_REFERENCE)
    patch_source = Path(str(reference["snapshot_root"])).resolve()
    verify_snapshot(patch_source)

    plans: dict[str, dict[str, Any]] = {}
    for family in selected:
        if args.submit and family.ledger.exists():
            raise SystemExit(f"refusing duplicate E116 submission: {family.ledger}")
        parent = shared.read_ledger(family.parent)
        runs = shared.controls(parent, domains=family.domains, seeds=family.seeds)
        bases = {shared.base_snapshot(run) for run in runs}
        if len(bases) != 1:
            raise RuntimeError(f"{family.key}: parent controls span runtimes")
        snapshot, patched = shared.ensure_overlay_snapshot(
            base=bases.pop(),
            patch_source=patch_source,
            prefix=f"e116_sparse_rlep_{family.key}",
            patched_files=PATCHED_FILES,
        )
        verify_snapshot(snapshot)
        views: dict[str, tuple[Path, str]] = {}
        for domain in family.domains:
            template = next(run for run in runs if str(run["domain"]) == domain)
            submitted_env = shared.export_pairs(shared.submit_line(template))
            prompt_data = submitted_env.get("OAT_ZERO_PROMPT_DATA", "")
            if not prompt_data:
                raise SystemExit(f"{family.key}/{domain}: parent lacks prompt data")
            source_view = {
                "domain": domain,
                "inherited_eval_config": {"prompt_data": prompt_data},
            }
            views[domain] = e98.materialize_collection_data(ROOT, source_view)

        cells: list[dict[str, Any]] = []
        for run in runs:
            domain, seed = str(run["domain"]), int(run["seed"])
            pool = pool_path(family, domain, seed)
            target = output_path(family, domain, seed)
            if pool.exists() or target.exists():
                raise SystemExit(f"refusing existing E116 artifact for {family.key}/{domain}/s{seed}")
            seed_export, seed_export_ready = terminal_export(run)
            collection_data, data_hash = views[domain]
            collect = shared.clone_command(
                run,
                name=f"e116-{family.key}-{domain_tag(domain)}-pool-s{seed}",
                overrides=collection_overrides(
                    family,
                    run,
                    snapshot=snapshot,
                    seed_export=seed_export,
                    collection_data=collection_data,
                    pool=pool,
                ),
                dependency="" if seed_export_ready else str(run["job_id"]),
            )
            train_values = training_overrides(
                family, run, snapshot=snapshot, pool=pool, target=target
            )
            cells.append(
                {
                    "run": run,
                    "domain": domain,
                    "seed": seed,
                    "pool": pool,
                    "target": target,
                    "seed_export": seed_export,
                    "seed_export_ready": seed_export_ready,
                    "collection_data": collection_data,
                    "collection_data_sha256": data_hash,
                    "collect_argv": collect,
                    "train_overrides": train_values,
                }
            )
        smoke_cell = cells[0]
        smoke_values, smoke_target = smoke_overrides(
            smoke_cell["train_overrides"], family
        )
        if smoke_target.exists():
            raise SystemExit(f"refusing existing E116 smoke: {smoke_target}")
        plans[family.key] = {
            "family": family,
            "parent": parent,
            "snapshot": snapshot,
            "patched": patched,
            "cells": cells,
            "smoke_cell": smoke_cell,
            "smoke_values": smoke_values,
            "smoke_target": smoke_target,
        }

    if args.dry_run or not args.submit:
        for plan in plans.values():
            family: Family = plan["family"]
            first = plan["cells"][0]
            print(shlex.join(first["collect_argv"]))
            print(shlex.join(audit_pool_command(family=family, domain=first["domain"], seed=first["seed"], snapshot=plan["snapshot"], pool=first["pool"], dependency="POOL_JOB_ID")))
            smoke = shared.clone_command(first["run"], name=f"e116-{family.key}-smoke", overrides=plan["smoke_values"], dependency="POOL_AUDIT_ID", time_limit="03:00:00")
            print(shlex.join(smoke))
            science = shared.clone_command(first["run"], name=f"e116-{family.key}-{domain_tag(first['domain'])}-s{first['seed']}", overrides=first["train_overrides"], dependency="POOL_AUDIT_ID:SMOKE_AUDIT_ID")
            print(shlex.join(science))
            ready = sum(bool(cell["seed_export_ready"]) for cell in plan["cells"])
            print(f"[e116] {family.key}: pools={len(plan['cells'])} audits={len(plan['cells'])} smoke=1 smoke_audit=1 science={len(plan['cells'])} terminal_seed_exports={ready}/{len(plan['cells'])}")
        return 0

    submitted: list[str] = []
    payloads: list[tuple[Path, dict[str, Any]]] = []
    try:
        for plan in plans.values():
            family: Family = plan["family"]
            snapshot: Path = plan["snapshot"]
            pool_records: list[dict[str, Any]] = []
            for cell in plan["cells"]:
                run, domain, seed = cell["run"], cell["domain"], cell["seed"]
                collect_name = f"e116-{family.key}-{domain_tag(domain)}-pool-s{seed}"
                collect_id = shared.submit_held(cell["collect_argv"])
                submitted.append(collect_id)
                collect_expected = [
                    f"OAT_ZERO_PRETRAIN={cell['seed_export']}",
                    f"OAT_ZERO_EVAL_DATA={cell['collection_data']}",
                    "OAT_ZERO_EVAL_ONLY=1",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_K=16",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=0.7",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P=0.95",
                    "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
                ]
                if not cell["seed_export_ready"]:
                    collect_expected.insert(
                        0, f"Dependency=afterok:{int(run['job_id'])}"
                    )
                collect_held = shared.audit_held(
                    collect_id,
                    name=collect_name,
                    expected=collect_expected,
                )
                audit_argv = audit_pool_command(
                    family=family,
                    domain=domain,
                    seed=seed,
                    snapshot=snapshot,
                    pool=cell["pool"],
                    dependency=collect_id,
                )
                audit_name = f"e116-{family.key}-{domain_tag(domain)}-audit-s{seed}"
                audit_id = shared.submit_held(audit_argv)
                submitted.append(audit_id)
                audit_held = shared.audit_held(
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
                        "seed_policy_job_id": int(run["job_id"]),
                        "seed_policy_export": str(cell["seed_export"]),
                        "seed_policy_terminal_at_submission": bool(cell["seed_export_ready"]),
                        "collection_data": str(cell["collection_data"]),
                        "collection_data_sha256": cell["collection_data_sha256"],
                        "collection_seed": pool_seed(family, domain, seed),
                        "pool_root": str(cell["pool"]),
                        "collection_job_id": int(collect_id),
                        "audit_job_id": int(audit_id),
                        "held_collection_scheduler_record": collect_held,
                        "held_audit_scheduler_record": audit_held,
                    }
                )

            smoke_cell = plan["smoke_cell"]
            smoke_id = shared.submit_held(
                shared.clone_command(
                    smoke_cell["run"],
                    name=f"e116-{family.key}-smoke",
                    overrides=plan["smoke_values"],
                    dependency=str(smoke_cell["audit_id"]),
                    time_limit="03:00:00",
                )
            )
            submitted.append(smoke_id)
            smoke_held = shared.audit_held(
                smoke_id,
                name=f"e116-{family.key}-smoke",
                expected=(
                    f"Dependency=afterok:{smoke_cell['audit_id']}",
                    "OAT_ZERO_VARIANT=rlep",
                    "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                    "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
                    "OAT_ZERO_MAX_TRAIN=32",
                ),
            )
            smoke_audit_argv = smoke_audit_command(
                family=family,
                snapshot=snapshot,
                target=plan["smoke_target"],
                dependency=smoke_id,
            )
            smoke_audit_id = shared.submit_held(smoke_audit_argv)
            submitted.append(smoke_audit_id)
            smoke_audit_held = shared.audit_held(
                smoke_audit_id,
                name=f"e116-{family.key}-smoke-audit",
                expected=(f"Dependency=afterok:{smoke_id}", str(plan["smoke_target"]), "--expected-terminal-step"),
            )

            run_records: list[dict[str, Any]] = []
            for cell in plan["cells"]:
                domain, seed, run = cell["domain"], cell["seed"], cell["run"]
                dependency = f"{cell['audit_id']}:{smoke_audit_id}"
                name = f"e116-{family.key}-{domain_tag(domain)}-s{seed}"
                job_id = shared.submit_held(
                    shared.clone_command(
                        run,
                        name=name,
                        overrides=cell["train_overrides"],
                        dependency=dependency,
                    )
                )
                submitted.append(job_id)
                held = shared.audit_held(
                    job_id,
                    name=name,
                    expected=(
                        "Dependency=afterok:",
                        str(cell["audit_id"]),
                        smoke_audit_id,
                        f"OAT_ZERO_SEED={seed}",
                        "OAT_ZERO_VARIANT=rlep",
                        f"OAT_ZERO_RLEP_EXPERIENCE_ROOT={cell['pool']}",
                        "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                        "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
                        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=0",
                        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
                    ),
                )
                run_records.append(
                    {
                        "domain": domain,
                        "arm": ARM,
                        "seed": seed,
                        "job_id": int(job_id),
                        "run_stamp": run_stamp(family, domain, seed),
                        "run_dir": str(cell["target"]),
                        "pool_root": str(cell["pool"]),
                        "pool_audit_dependency_job_id": int(cell["audit_id"]),
                        "smoke_audit_dependency_job_id": int(smoke_audit_id),
                        "paired_control": {
                            "job_id": int(run["job_id"]),
                            "run_stamp": str(run["run_stamp"]),
                            "run_dir": str(run["run_dir"]),
                        },
                        "held_scheduler_record": held,
                    }
                )
            payload = {
                "schema": "e116_sparse_rlep_completion_jobs_v1",
                "cohort": f"e116_{family.key}",
                "released": False,
                "model": family.label,
                "model_revision": family.model_revision,
                "protocol": str(PROTOCOL),
                "protocol_sha256": shared.digest(PROTOCOL),
                "launcher_sha256": shared.digest(Path(__file__)),
                "parent_ledger": str(family.parent),
                "parent_ledger_sha256": shared.digest(family.parent),
                "rlep_reference_ledger": str(RLEP_REFERENCE),
                "rlep_reference_ledger_sha256": shared.digest(RLEP_REFERENCE),
                "snapshot_root": str(snapshot),
                "snapshot_patched_files": plan["patched"],
                "domains": list(family.domains),
                "seeds": list(family.seeds),
                "passes": 8,
                "train_rows": 384,
                "target_steps": 3072,
                "checkpoint_interval_steps": 192,
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
                "objective": "sparse_prompt_matched_RLEP_16_plus_2_else_DrGRPO_16",
                "scientific_difference": "against paired Dr.GRPO: two frozen prior-policy successes on eligible prompts only",
                "smoke": {
                    "scientific": False,
                    "job_id": int(smoke_id),
                    "audit_job_id": int(smoke_audit_id),
                    "run_dir": str(plan["smoke_target"]),
                    "target_steps": 32,
                    "pool_audit_dependency_job_id": int(smoke_cell["audit_id"]),
                    "held_scheduler_record": smoke_held,
                    "held_audit_scheduler_record": smoke_audit_held,
                },
                "runs": run_records,
            }
            shared.atomic_json(family.ledger, payload)
            payloads.append((family.ledger, payload))
        shared.release(submitted)
        for path, payload in payloads:
            payload["released"] = True
            shared.atomic_json(path, payload)
    except Exception:
        shared.cancel(submitted)
        raise
    print(f"[e116] released {len(submitted)} staged jobs for {sum(len(plan['cells']) for plan in plans.values())} scientific cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

