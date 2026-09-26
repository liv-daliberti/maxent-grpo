#!/usr/bin/env python3
"""Submit E114: Qwen2.5-3B plain-GRPO seeds 71--74."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_comparator_completion as shared  # noqa: E402
import launch_e95_plain_grpo_control as e95  # noqa: E402


ROOT = shared.ROOT
DOMAINS = e95.DOMAINS
SEEDS = (71, 72, 73, 74)
FAMILY = "Qwen2.5-3B"
PARENT = ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
REFERENCE = ROOT / "var/artifacts/e95_plain_grpo_Qwen25-3B_jobs.json"
LEDGER = ROOT / "var/artifacts/e114_plain_grpo_qwen3b_extension_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e114_plain_grpo_qwen3b_seed_extension_20260819.md"
VARIANT = "grpo_plain_control"


def tag(domain: str) -> str:
    return {
        "graph_coloring": "graph",
        "countdown": "count",
        "python_factors": "python",
        "mathir": "mathir",
        "pantry_plan": "pantry",
    }[domain]


def run_stamp(domain: str, seed: int) -> str:
    return f"e114_qwen3b_{domain}_plain_grpo_s{seed}"


def output_path(domain: str, seed: int) -> Path:
    return ROOT / "var/data" / f"xdr_qwen25_3b_instruct_{run_stamp(domain, seed)}"


def command(run: dict[str, Any], snapshot: Path) -> list[str]:
    domain, seed = str(run["domain"]), int(run["seed"])
    return shared.clone_command(
        run,
        name=f"e114-q3-{tag(domain)}-s{seed}",
        overrides={
            "OAT_ZERO_VARIANT": VARIANT,
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "SAVE_PATH": str(output_path(domain, seed)),
            "RUN_STAMP": run_stamp(domain, seed),
        },
    )


def preflight(run: dict[str, Any], snapshot: Path) -> None:
    env = shared.export_pairs(command(run, snapshot))
    with tempfile.TemporaryDirectory(prefix="e114-preflight-", dir="/tmp") as temporary:
        env.update(
            {
                "SAVE_PATH": str(Path(temporary) / "run"),
                "RUN_STAMP": "e114_plain_grpo_preflight",
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
        raise RuntimeError(f"E114 plain-GRPO preflight failed:\n{combined[-4000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    for required in (PARENT, REFERENCE, PROTOCOL):
        if not required.is_file():
            raise SystemExit(f"missing E114 input: {required}")
    if args.submit and LEDGER.exists():
        raise SystemExit(f"refusing duplicate E114 submission: {LEDGER}")

    parent = shared.read_ledger(PARENT)
    runs = shared.controls(parent, domains=DOMAINS, seeds=SEEDS)
    reference = shared.read_ledger(REFERENCE)
    if str(reference.get("family")) != FAMILY:
        raise RuntimeError("E114 reference is not the Qwen2.5-3B E95 ledger")
    snapshot = Path(str(reference["snapshot_root"])).resolve()
    patched = list(reference.get("snapshot_patched_files", []))
    missing = [
        f"{relative}: {needle!r}"
        for relative, needle in e95.SNAPSHOT_REQUIREMENTS
        if not (snapshot / relative).is_file()
        or needle not in (snapshot / relative).read_text(encoding="utf-8")
    ]
    if missing:
        raise RuntimeError("E114 immutable E95 snapshot drift: " + "; ".join(missing))
    preflight(runs[0], snapshot)
    commands = [command(run, snapshot) for run in runs]
    for run in runs:
        target = output_path(str(run["domain"]), int(run["seed"]))
        if target.exists():
            raise SystemExit(f"refusing existing E114 output: {target}")

    if args.dry_run or not args.submit:
        print(shlex.join(commands[0]))
        print(f"[e114] scientific={len(commands)} seeds={list(SEEDS)} snapshot={snapshot}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for run, argv in zip(runs, commands):
            domain, seed = str(run["domain"]), int(run["seed"])
            name = f"e114-q3-{tag(domain)}-s{seed}"
            job_id = shared.submit_held(argv)
            submitted.append(job_id)
            held = shared.audit_held(
                job_id,
                name=name,
                expected=(
                    f"OAT_ZERO_SEED={seed}",
                    f"OAT_ZERO_VARIANT={VARIANT}",
                    f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
                    "OAT_ZERO_NUM_PROMPT_EPOCH=8",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
                ),
            )
            records.append(
                {
                    "arm": VARIANT,
                    "domain": domain,
                    "seed": seed,
                    "job_id": int(job_id),
                    "run_stamp": run_stamp(domain, seed),
                    "run_dir": str(output_path(domain, seed)),
                    "parent_control_job_id": int(run["job_id"]),
                    "held_scheduler_record": held,
                }
            )
        payload = {
            "schema": "e114_plain_grpo_qwen3b_extension_jobs_v1",
            "cohort": "e114",
            "released": False,
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "protocol": str(PROTOCOL),
            "protocol_sha256": shared.digest(PROTOCOL),
            "launcher_sha256": shared.digest(Path(__file__)),
            "parent_ledger": str(PARENT),
            "parent_ledger_sha256": shared.digest(PARENT),
            "plain_grpo_reference_ledger": str(REFERENCE),
            "plain_grpo_reference_ledger_sha256": shared.digest(REFERENCE),
            "extends": "E95 Qwen2.5-3B seed 70 without rerunning it",
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "domains": list(DOMAINS),
            "seeds": list(SEEDS),
            "passes": 8,
            "train_rows": 384,
            "target_steps": 3072,
            "checkpoint_interval_steps": 192,
            "variant": VARIANT,
            "objective": "plain_GRPO_reward_only_control",
            "scientific_difference": "against E80-R1 Dr.GRPO: critic_type=grpo only",
            "runs": records,
        }
        shared.atomic_json(LEDGER, payload)
        shared.release(submitted)
        payload["released"] = True
        shared.atomic_json(LEDGER, payload)
    except Exception:
        shared.cancel(submitted)
        raise
    print(f"[e114] released {len(records)} scientific cells; ledger={LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

