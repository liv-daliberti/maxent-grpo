#!/usr/bin/env python3
"""Submit E115 UCPO completion at Qwen2.5-0.5B and Qwen2.5-3B."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_comparator_completion as shared  # noqa: E402
import launch_e97_ucpo_05b as e97  # noqa: E402


ROOT = shared.ROOT
TAU = 0.2
VARIANT = "ucpo"


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
        ROOT / "var/artifacts/e115_ucpo_qwen05b_domain_extension_jobs.json",
        ("countdown", "mathir"),
        (43, 44, 45, 46, 47),
        "qwen25_05b_instruct",
    ),
    Family(
        "qwen3b",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
        ROOT / "var/artifacts/e115_ucpo_qwen3b_jobs.json",
        ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"),
        (70, 71, 72, 73, 74),
        "qwen25_3b_instruct",
    ),
)
PROTOCOL = ROOT / "paper/preregistration/e115_ucpo_direct_comparator_completion_20260819.md"
UCPO_REFERENCE = ROOT / "var/artifacts/e97_ucpo_05b_jobs.json"


def domain_tag(domain: str) -> str:
    return {
        "graph_coloring": "graph",
        "countdown": "count",
        "python_factors": "python",
        "mathir": "mathir",
        "pantry_plan": "pantry",
    }[domain]


def run_stamp(family: Family, domain: str, seed: int) -> str:
    return f"e115_{family.key}_{domain}_ucpo_s{seed}"


def output_path(family: Family, domain: str, seed: int) -> Path:
    return ROOT / "var/data" / f"xdr_{family.model_tag}_{run_stamp(family, domain, seed)}"


def objective() -> dict[str, str]:
    return {
        "OAT_ZERO_VARIANT": VARIANT,
        "OAT_ZERO_UCPO_TAU": repr(TAU),
        "OAT_ZERO_RLEP_EXPERIENCE_ROOT": "",
        "OAT_ZERO_RLEP_REPLAY_COUNT": "0",
    }


def science_command(family: Family, run: dict[str, Any], snapshot: Path, dependency: str) -> list[str]:
    domain, seed = str(run["domain"]), int(run["seed"])
    overrides = objective()
    overrides.update(
        {
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "SAVE_PATH": str(output_path(family, domain, seed)),
            "RUN_STAMP": run_stamp(family, domain, seed),
        }
    )
    return shared.clone_command(
        run,
        name=f"e115-{family.key}-{domain_tag(domain)}-s{seed}",
        overrides=overrides,
        dependency=dependency,
    )


def smoke_command(family: Family, run: dict[str, Any], snapshot: Path) -> tuple[list[str], Path]:
    domain, seed = str(run["domain"]), int(run["seed"])
    target = ROOT / "var/data" / f"e115_ucpo_{family.key}_smoke"
    overrides = objective()
    overrides.update(
        {
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "SAVE_PATH": str(target),
            "RUN_STAMP": f"e115_ucpo_{family.key}_smoke",
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
    return (
        shared.clone_command(
            run,
            name=f"e115-{family.key}-smoke",
            overrides=overrides,
            time_limit="03:00:00",
        ),
        target,
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
        raise SystemExit(f"missing E115 protocol: {PROTOCOL}")
    reference = shared.read_ledger(UCPO_REFERENCE)
    patch_source = Path(str(reference["snapshot_root"])).resolve()
    e97.verify_snapshot(patch_source)

    plans: dict[str, dict[str, Any]] = {}
    for family in selected:
        if args.submit and family.ledger.exists():
            raise SystemExit(f"refusing duplicate E115 submission: {family.ledger}")
        parent = shared.read_ledger(family.parent)
        runs = shared.controls(parent, domains=family.domains, seeds=family.seeds)
        bases = {shared.base_snapshot(run) for run in runs}
        if len(bases) != 1:
            raise RuntimeError(f"{family.key}: parent controls span runtimes")
        snapshot, patched = shared.ensure_overlay_snapshot(
            base=bases.pop(),
            patch_source=patch_source,
            prefix=f"e115_ucpo_{family.key}",
            patched_files=e97.PATCHED_FILES,
        )
        e97.verify_snapshot(snapshot)
        smoke_argv, smoke_target = smoke_command(family, runs[0], snapshot)
        if smoke_target.exists():
            raise SystemExit(f"refusing existing E115 smoke: {smoke_target}")
        for run in runs:
            target = output_path(family, str(run["domain"]), int(run["seed"]))
            if target.exists():
                raise SystemExit(f"refusing existing E115 output: {target}")
        plans[family.key] = {
            "family": family,
            "parent": parent,
            "runs": runs,
            "snapshot": snapshot,
            "patched": patched,
            "smoke_argv": smoke_argv,
            "smoke_target": smoke_target,
        }

    if args.dry_run or not args.submit:
        for plan in plans.values():
            family = plan["family"]
            print(shlex.join(plan["smoke_argv"]))
            first = science_command(family, plan["runs"][0], plan["snapshot"], "SMOKE_JOB_ID")
            print(shlex.join(first))
            print(f"[e115] {family.key}: smoke=1 scientific={len(plan['runs'])} snapshot={plan['snapshot']}")
        return 0

    submitted: list[str] = []
    payloads: list[tuple[Path, dict[str, Any]]] = []
    try:
        for plan in plans.values():
            family: Family = plan["family"]
            snapshot: Path = plan["snapshot"]
            smoke_id = shared.submit_held(plan["smoke_argv"])
            submitted.append(smoke_id)
            smoke_held = shared.audit_held(
                smoke_id,
                name=f"e115-{family.key}-smoke",
                expected=(
                    "OAT_ZERO_VARIANT=ucpo",
                    "OAT_ZERO_UCPO_TAU=0.2",
                    "OAT_ZERO_MAX_TRAIN=32",
                    "OAT_ZERO_MAX_QUERIES=32",
                ),
            )
            records: list[dict[str, Any]] = []
            for run in plan["runs"]:
                domain, seed = str(run["domain"]), int(run["seed"])
                name = f"e115-{family.key}-{domain_tag(domain)}-s{seed}"
                argv = science_command(family, run, snapshot, smoke_id)
                job_id = shared.submit_held(argv)
                submitted.append(job_id)
                held = shared.audit_held(
                    job_id,
                    name=name,
                    expected=(
                        f"Dependency=afterok:{smoke_id}",
                        f"OAT_ZERO_SEED={seed}",
                        "OAT_ZERO_VARIANT=ucpo",
                        "OAT_ZERO_UCPO_TAU=0.2",
                        "OAT_ZERO_RLEP_REPLAY_COUNT=0",
                        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
                        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
                    ),
                )
                records.append(
                    {
                        "domain": domain,
                        "arm": "ucpo",
                        "seed": seed,
                        "job_id": int(job_id),
                        "run_stamp": run_stamp(family, domain, seed),
                        "run_dir": str(output_path(family, domain, seed)),
                        "smoke_dependency_job_id": int(smoke_id),
                        "paired_control": {
                            "job_id": int(run["job_id"]),
                            "run_stamp": str(run["run_stamp"]),
                            "run_dir": str(run["run_dir"]),
                        },
                        "held_scheduler_record": held,
                    }
                )
            payload = {
                "schema": "e115_ucpo_completion_jobs_v1",
                "cohort": f"e115_{family.key}",
                "released": False,
                "model": family.label,
                "model_revision": family.model_revision,
                "protocol": str(PROTOCOL),
                "protocol_sha256": shared.digest(PROTOCOL),
                "launcher_sha256": shared.digest(Path(__file__)),
                "parent_ledger": str(family.parent),
                "parent_ledger_sha256": shared.digest(family.parent),
                "ucpo_reference_ledger": str(UCPO_REFERENCE),
                "ucpo_reference_ledger_sha256": shared.digest(UCPO_REFERENCE),
                "snapshot_root": str(snapshot),
                "snapshot_patched_files": plan["patched"],
                "domains": list(family.domains),
                "seeds": list(family.seeds),
                "passes": 8,
                "train_rows": 384,
                "target_steps": 3072,
                "checkpoint_interval_steps": 192,
                "variant": VARIANT,
                "ucpo_tau": TAU,
                "objective": "DrGRPO_with_uniform_correct_advantage_redistribution",
                "scientific_difference": "against paired Dr.GRPO: UCPO tau=.2 only",
                "smoke": {
                    "scientific": False,
                    "job_id": int(smoke_id),
                    "run_dir": str(plan["smoke_target"]),
                    "target_steps": 32,
                    "held_scheduler_record": smoke_held,
                },
                "runs": records,
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
    print(f"[e115] released {len(submitted)} jobs including {len(selected)} smokes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

