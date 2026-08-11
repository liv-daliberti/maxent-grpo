#!/usr/bin/env python3
"""Submit E83: fixed semantic MaxEnt with the replay derivative off.

E83 closes the 2x2 over the two applied derivatives. Its cell definitions and
objective come from the E81 launcher; the only differences are the variant
branch and the replay compute-only switch, so the arm is the published
compute-matched control plus semantic MaxEnt and nothing else. E78 and E81 are
inherited unchanged and never re-run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


DOMAINS = e81.DOMAINS
ARM = "semantic_only"
SEEDS = e81.SEEDS
PASSES = e81.PASSES
TRAIN_ROWS = e81.TRAIN_ROWS
CHECKPOINT_INTERVAL = e81.CHECKPOINT_INTERVAL
TARGET_STEPS = e81.TARGET_STEPS
REPLAY_WEIGHT = e81.REPLAY_WEIGHT
SEMANTIC_COEF = e81.SEMANTIC_COEF
VARIANT = "compute_matched_semantic_maxent"
MODEL_TAG = e81.MODEL_TAG

LEDGER = "var/artifacts/e83_semantic_maxent_without_replay_05b_jobs.json"
PROTOCOL = (
    "paper/preregistration/e83_semantic_maxent_without_replay_05b_20260807.md"
)
SOURCE_MANIFEST = e81.SOURCE_MANIFEST
PAIR_LEDGER = e81.PAIR_LEDGER
E81_LEDGER = "var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json"
SNAPSHOT_PREFIX = "e83_semantic_maxent_no_replay"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(domain: str, seed: int) -> str:
    return f"e83_semantic_only_{e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective() -> dict[str, str]:
    """E81's objective with the replay derivative switched off.

    Exactly two keys move. Everything else -- the semantic coefficient, clip,
    pseudocount, gating, bank capacity, scheduler, and every disabled actuator
    -- is inherited from E81 so the factorial varies one thing at a time.
    """

    objective = dict(e81.fixed_objective())
    objective["OAT_ZERO_VARIANT"] = VARIANT
    objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] = "1"
    return objective


def build_env(
    root: Path, run: dict[str, Any], snapshot_root: Path
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    env, _ = e81.build_env(root, run, snapshot_root)
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(fixed_objective())
    return env, target


def sbatch_command(root: Path, run: dict[str, Any], env: dict[str, str]) -> list[str]:
    node = str(run["source_node"])
    place = e81.placement(node)
    name = f"e83-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-sonly-s{run['seed']}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={node}",
        f"--gres={place['gres']}",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=1-12:00:00",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot inspect held E83 job {job_id}: {result.stderr.strip()}"
        )
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e83-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-sonly-s{run['seed']}",
        f"ReqNodeList={run['source_node']}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        # the defining switch: replay is traversed for compute, never applied
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA=0.0",
    )
    missing = [text for text in required if text not in record]
    if missing:
        raise RuntimeError(f"held E83 job {job_id} lacks {missing}")
    return record


def factorial_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Map (domain, seed) to the three inherited arms of the 2x2."""

    index = e81.pair_index(e81.pair_ledger(root))
    e81_path = root / E81_LEDGER
    if e81_path.is_file():
        payload = json.loads(e81_path.read_text(encoding="utf-8"))
        for run in payload["runs"]:
            key = (str(run["domain"]), int(run["seed"]))
            if key in index:
                index[key]["semantic"] = run
    return index


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol = root / PROTOCOL
    source_manifest = root / SOURCE_MANIFEST
    for required in (protocol, source_manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen E83 input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E83 submission: {ledger}")

    pairs = factorial_index(root)
    source_runs = e81.references(root)
    e78_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(e81.pair_ledger(root)["snapshot_root"])
    )
    snapshot_root, patched = e81.ensure_paired_snapshot(
        root, e78_snapshot, prefix=SNAPSHOT_PREFIX
    )

    planned: list[dict[str, Any]] = []
    for run in source_runs:
        domain = str(run["domain"])
        seed = int(run["seed"])
        pair = pairs[(domain, seed)]
        if str(pair["control"]["source_node"]) != str(run["source_node"]):
            raise SystemExit(
                f"{domain}/s{seed}: E78 control ran on "
                f"{pair['control']['source_node']}, manifest pins "
                f"{run['source_node']}"
            )
        env, target = build_env(root, run, snapshot_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E83 run: {target}")
        planned.append(
            {
                "domain": domain,
                "arm": ARM,
                "seed": seed,
                "source_node": str(run["source_node"]),
                "run_stamp": run_stamp(domain, seed),
                "run_dir": str(target),
                "paired_runs": {
                    arm: {
                        "run_stamp": str(record["run_stamp"]),
                        "run_dir": str(record["run_dir"]),
                        "job_id": int(record["job_id"]),
                    }
                    for arm, record in pair.items()
                },
                "command": sbatch_command(root, run, env),
                "template": run,
            }
        )

    if len(planned) != 25:
        raise SystemExit(f"E83 expected 25 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e83] dry_run=True cells={len(planned)} snapshot={snapshot_root} "
            f"patched={patched}"
        )
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(
                    f"invalid job id for {cell['run_stamp']}: {result.stdout!r}"
                )
            submitted.append(job_id)
            held_record = held_job_audit(job_id, cell["template"])
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "domain",
                        "arm",
                        "seed",
                        "source_node",
                        "run_stamp",
                        "run_dir",
                        "paired_runs",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held_record}
            )

        payload = {
            "schema": "e83_semantic_maxent_without_replay_05b_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": e81.digest(source_manifest),
            "pair_ledger": str(root / PAIR_LEDGER),
            "pair_ledger_sha256": e81.digest(root / PAIR_LEDGER),
            "e78_snapshot_root": str(e78_snapshot),
            "snapshot_root": str(snapshot_root),
            "snapshot_patched_files": patched,
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(DOMAINS),
            "arms": [ARM],
            "inherited_arms": ["control", "replay", "semantic"],
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_weight": REPLAY_WEIGHT,
            "replay_derivative_applied": False,
            "semantic_coefficient": SEMANTIC_COEF,
            "semantic_surprisal_clip": e81.SEMANTIC_SURPRISAL_CLIP,
            "semantic_pseudocount": e81.SEMANTIC_PSEUDOCOUNT,
            "objective": "fixed_open_set_semantic_maxent_only",
            "scientific_difference": (
                "against E78 control: the applied semantic advantage only; "
                "against E81: the applied replay derivative only"
            ),
            "completes_factorial": "e78_control x e78_replay x e81_semantic",
            "runs": records,
            "released": False,
        }
        e81.atomic_json(ledger, payload)
        for job_id in submitted:
            release = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if release.returncode != 0:
                raise RuntimeError(
                    f"release failed for {job_id}: {release.stderr.strip()}"
                )
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        e81.cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e83] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
