#!/usr/bin/env python3
"""Submit E103: E102 plus a bounded explorer-starvation fallback at 0.5B."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e102_full_open_bank_maxent_replay_05b as e102  # noqa: E402


ARM = "starvation_fallback"
VARIANT = "open_bank_starvation_fallback_maxent_replay"
DOMAINS = e102.DOMAINS
SEEDS = e102.SEEDS
PASSES = e102.PASSES
TRAIN_ROWS = e102.TRAIN_ROWS
CHECKPOINT_INTERVAL = e102.CHECKPOINT_INTERVAL
TARGET_STEPS = e102.TARGET_STEPS
REPLAY_MASS_WEIGHT = e102.REPLAY_MASS_WEIGHT
BANK_BALANCE_WEIGHT = e102.BANK_BALANCE_WEIGHT
PROPOSAL_TEMPERATURE = e102.PROPOSAL_TEMPERATURE
PROPOSAL_MAX_ATTEMPTS = e102.PROPOSAL_MAX_ATTEMPTS
PRIORITY_VISITS = e102.PRIORITY_VISITS
PRIORITY_MULTIPLIER = e102.PRIORITY_MULTIPLIER

STARVATION_PATIENCE_UPDATES = 64
STARVATION_FALLBACK_MAX_ATTEMPTS = 4
STARVATION_BURST_UPDATES = 16
STARVATION_COOLDOWN_UPDATES = 48

SMOKE_TRAIN_ROWS = 4
SMOKE_PASSES = 8
SMOKE_TARGET_STEPS = SMOKE_TRAIN_ROWS * SMOKE_PASSES
SMOKE_PATIENCE_UPDATES = 2
SMOKE_BURST_UPDATES = 2
SMOKE_COOLDOWN_UPDATES = 2
SMOKE_SUFFIX = "smoke"

LEDGER = "var/artifacts/e103_starvation_fallback_maxent_replay_05b_jobs.json"
PROTOCOL = "paper/preregistration/e103_starvation_fallback_maxent_replay_05b_20260817.md"
E102_LEDGER = e102.LEDGER
E78_LEDGER = e102.E78_LEDGER
SMOKE_AUDIT = "var/artifacts/e103_starvation_fallback_smoke_gate.json"
SOURCE_MANIFEST = e102.SOURCE_MANIFEST
MODEL_TAG = e102.MODEL_TAG

# Start with the healthy 24--48 GB pool learned from E102 operations. node007
# had a full /tmp, node206 had contaminated GPU state, node302 is congested,
# and node105 is kept excluded from this paired family.
NODES = "node[020-025,101,103-104,202-205,207-208,403,805]"
PARTITION = "all"
ACCOUNT = "mltheory"
GRES = "gpu:1"
MEMORY = "36G"
TIME_LIMIT = "2-00:00:00"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_e102_comparator(root: Path) -> Path:
    path = root / E102_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("released") is not True:
        raise SystemExit("E103 requires the released E102 comparator ledger")
    for key, expected in (
        ("train_rows", TRAIN_ROWS),
        ("passes", PASSES),
        ("target_steps", TARGET_STEPS),
        ("checkpoint_interval_steps", CHECKPOINT_INTERVAL),
    ):
        if int(payload.get(key, -1)) != expected:
            raise SystemExit(f"E102 comparator {key} drifted from {expected}")
    runs = payload.get("runs", [])
    if Counter(str(run.get("arm")) for run in runs) != {"full_open_bank": 25}:
        raise SystemExit("E102 comparator ledger must contain 25 full-open-bank cells")
    expected_pairs = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    pairs = {
        (str(run.get("domain")), int(run.get("seed", -1))) for run in runs
    }
    if pairs != expected_pairs:
        raise SystemExit("E102 comparator domain/seed cells do not match E103")
    incomplete = [
        str(run.get("run_dir"))
        for run in runs
        if not (Path(str(run.get("run_dir"))) / "TRAINING_COMPLETE.json").is_file()
    ]
    if incomplete:
        raise SystemExit(f"E102 comparator has incomplete cells: {incomplete}")
    return path


def check_smoke_gate(root: Path, snapshot_root: Path) -> Path:
    path = root / SMOKE_AUDIT
    if not path.is_file():
        raise SystemExit(f"E103 smoke gate audit is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "e103-starvation-fallback-smoke-audit-v1":
        raise SystemExit("E103 smoke gate has an unknown schema")
    if payload.get("passed") is not True:
        raise SystemExit("E103 full release requires a passing smoke gate")
    if Path(str(payload.get("snapshot_root"))) != snapshot_root:
        raise SystemExit("E103 smoke gate was produced from a different source snapshot")
    reports = payload.get("reports", {})
    if set(reports) != set(DOMAINS):
        raise SystemExit("E103 smoke gate must cover all five registered domains")
    for domain, report in reports.items():
        if int(report.get("last_step", -1)) < SMOKE_TARGET_STEPS:
            raise SystemExit(f"E103 {domain} smoke did not reach its target")
        if float(report.get("fallback_activations", 0.0)) <= 0.0:
            raise SystemExit(f"E103 {domain} smoke never activated the fallback")
        if float(report.get("fallback_extra_groups", 0.0)) <= 0.0:
            raise SystemExit(f"E103 {domain} smoke generated no fallback groups")
        if float(report.get("proposal_rows_to_ppo_max", 1.0)) != 0.0:
            raise SystemExit(f"E103 {domain} smoke leaked proposal rows into PPO")
        if float(report.get("objective_outcome_delta_max", 1.0)) != 0.0:
            raise SystemExit(f"E103 {domain} smoke changed PPO objective support")
        if float(report.get("applied_positive_gradient_max", 1.0)) > 1e-7:
            raise SystemExit(f"E103 {domain} smoke violated retention safety")
    return path


def run_stamp(domain: str, seed: int) -> str:
    return f"e103_starvation_fallback_{e102.e78.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective(*, smoke: bool = False) -> dict[str, str]:
    objective = e102.fixed_objective()
    objective.update(
        {
            "OAT_ZERO_VARIANT": VARIANT,
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK": "1",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES": str(
                SMOKE_PATIENCE_UPDATES if smoke else STARVATION_PATIENCE_UPDATES
            ),
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK_MAX_ATTEMPTS": str(
                STARVATION_FALLBACK_MAX_ATTEMPTS
            ),
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_BURST_UPDATES": str(
                SMOKE_BURST_UPDATES if smoke else STARVATION_BURST_UPDATES
            ),
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_COOLDOWN_UPDATES": str(
                SMOKE_COOLDOWN_UPDATES if smoke else STARVATION_COOLDOWN_UPDATES
            ),
        }
    )
    return objective


def build_env(
    root: Path,
    run: dict[str, Any],
    snapshot_root: Path,
    *,
    smoke: bool = False,
) -> tuple[dict[str, str], Path]:
    env, _ = e102.build_env(root, run, snapshot_root, smoke=smoke)
    domain = str(run["domain"])
    seed = int(run["seed"])
    target = save_path(root, domain, seed)
    stamp = run_stamp(domain, seed)
    if smoke:
        target = target.with_name(target.name + f"_{SMOKE_SUFFIX}")
        stamp += f"_{SMOKE_SUFFIX}"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": stamp,
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(SMOKE_PASSES if smoke else PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(SMOKE_PASSES if smoke else PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(
                SMOKE_TARGET_STEPS if smoke else CHECKPOINT_INTERVAL
            ),
        }
    )
    if smoke:
        env["OAT_ZERO_MAX_TRAIN"] = str(SMOKE_TRAIN_ROWS)
    env.update(fixed_objective(smoke=smoke))
    return env, target


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    env: dict[str, str],
    *,
    smoke: bool = False,
) -> list[str]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    name = f"e103{'m' if smoke else ''}-{e102.e78.DOMAIN_TAGS[domain][:6]}-s{seed}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--nodelist={NODES}",
        f"--gres={GRES}",
        "--cpus-per-task=8",
        f"--mem={MEMORY}",
        f"--time={'01:00:00' if smoke else TIME_LIMIT}",
        "--nice=0",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any], *, smoke: bool = False) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"Account={ACCOUNT}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        f"OAT_ZERO_MAX_TRAIN={SMOKE_TRAIN_ROWS if smoke else TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={SMOKE_PASSES if smoke else PASSES}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK=1",
        f"OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES={SMOKE_PATIENCE_UPDATES if smoke else STARVATION_PATIENCE_UPDATES}",
        f"OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK_MAX_ATTEMPTS={STARVATION_FALLBACK_MAX_ATTEMPTS}",
        f"OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_BURST_UPDATES={SMOKE_BURST_UPDATES if smoke else STARVATION_BURST_UPDATES}",
        f"OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_COOLDOWN_UPDATES={SMOKE_COOLDOWN_UPDATES if smoke else STARVATION_COOLDOWN_UPDATES}",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=4",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER=4.0",
    )
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held job {job_id} lacks {missing}")
    # Slurm keeps short jobs on the requested `all` meta-partition, but routes
    # two-day jobs to its effective account partition (`mltheory`). The submit
    # line remains `--partition=all`; accept only these two audited forms.
    if not any(
        f"Partition={partition}" in record
        for partition in (PARTITION, ACCOUNT)
    ):
        raise RuntimeError(
            f"held E103 job {job_id} has an unexpected effective partition"
        )
    node_match = re.search(r"(?:^| )ReqNodeList=([^ ]+)", record)
    if node_match is None:
        raise RuntimeError(f"held E103 job {job_id} lacks ReqNodeList")
    requested_nodes = node_match.group(1)
    for excluded in ("node007", "node105", "node206", "node302"):
        if excluded in requested_nodes:
            raise RuntimeError(f"held E103 job {job_id} permits {excluded}")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def release(job_ids: list[str]) -> None:
    for job_id in job_ids:
        result = subprocess.run(
            ["scontrol", "release", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"release failed for {job_id}: {result.stderr.strip()}")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-domain", choices=DOMAINS, default="graph_coloring")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    if args.smoke and not args.submit and not args.dry_run:
        args.dry_run = True

    protocol = root / PROTOCOL
    source_manifest = root / SOURCE_MANIFEST
    for required in (protocol, source_manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    e102_ledger = check_e102_comparator(root)
    e78_ledger = e102.check_e78_comparators(root)
    ledger = root / LEDGER
    if args.submit and not args.smoke and ledger.exists():
        raise SystemExit(f"refusing duplicate E103 submission: {ledger}")

    source_runs = e102.e78.references(root)
    if args.smoke:
        source_runs = [
            run
            for run in source_runs
            if str(run["domain"]) == args.smoke_domain and int(run["seed"]) == 43
        ]
    snapshot_root = e102.snapshot_util.ensure_snapshot(root, args.snapshot_root)
    smoke_audit = (
        check_smoke_gate(root, snapshot_root)
        if args.submit and not args.smoke
        else None
    )

    planned: list[dict[str, Any]] = []
    for run in source_runs:
        env, target = build_env(root, run, snapshot_root, smoke=args.smoke)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E103 run: {target}")
        planned.append(
            {
                "arm": ARM,
                "domain": str(run["domain"]),
                "seed": int(run["seed"]),
                "run_stamp": run_stamp(str(run["domain"]), int(run["seed"]))
                + (f"_{SMOKE_SUFFIX}" if args.smoke else ""),
                "run_dir": str(target),
                "command": sbatch_command(root, run, env, smoke=args.smoke),
                "template": run,
                "objective": fixed_objective(smoke=args.smoke),
            }
        )
    expected_cells = 1 if args.smoke else 25
    if len(planned) != expected_cells:
        raise SystemExit(f"E103 expected {expected_cells} cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e103] dry_run=True smoke={args.smoke} cells={len(planned)} "
            f"snapshot={snapshot_root}"
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
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held_record = held_job_audit(job_id, cell["template"], smoke=args.smoke)
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "arm",
                        "domain",
                        "seed",
                        "run_stamp",
                        "run_dir",
                        "objective",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held_record}
            )

        if args.smoke:
            release(submitted)
            print(
                f"[e103] smoke_job={submitted[0]} domain={args.smoke_domain} "
                f"run_dir={records[0]['run_dir']} snapshot={snapshot_root}"
            )
            return 0

        payload = {
            "schema": "e103_starvation_fallback_maxent_replay_05b_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e102.e78.digest(protocol),
            "launcher_sha256": e102.e78.digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": e102.e78.digest(source_manifest),
            "primary_comparator_ledger": str(e102_ledger),
            "primary_comparator_ledger_sha256": e102.e78.digest(e102_ledger),
            "secondary_comparator_ledger": str(e78_ledger),
            "secondary_comparator_ledger_sha256": e102.e78.digest(e78_ledger),
            "smoke_gate_audit": str(smoke_audit),
            "smoke_gate_audit_sha256": e102.e78.digest(smoke_audit),
            "snapshot_root": str(snapshot_root),
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(DOMAINS),
            "arms": [ARM],
            "comparators": ["e102/full_open_bank", "e78/replay", "e78/control"],
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_mass_weight": REPLAY_MASS_WEIGHT,
            "bank_balance_weight_requested": BANK_BALANCE_WEIGHT,
            "retention_safe_balance": True,
            "proposal_temperature": PROPOSAL_TEMPERATURE,
            "proposal_base_max_attempts": PROPOSAL_MAX_ATTEMPTS,
            "starvation_patience_eligible_updates": STARVATION_PATIENCE_UPDATES,
            "starvation_fallback_max_attempts": STARVATION_FALLBACK_MAX_ATTEMPTS,
            "starvation_burst_eligible_updates": STARVATION_BURST_UPDATES,
            "starvation_cooldown_eligible_updates": STARVATION_COOLDOWN_UPDATES,
            "priority_visits": PRIORITY_VISITS,
            "priority_multiplier": PRIORITY_MULTIPLIER,
            "placement": {
                "partition": PARTITION,
                "allowed_effective_partitions": [PARTITION, ACCOUNT],
                "account": ACCOUNT,
                "nodes": NODES,
                "excluded_nodes": ["node007", "node105", "node206", "node302"],
                "gres": GRES,
                "memory": MEMORY,
                "nice": 0,
            },
            "scientific_difference": (
                "E102 unchanged except for a target-free bounded increase in "
                "original-prompt proposal attempts after verified admissions stall"
            ),
            "runs": records,
            "released": False,
        }
        e102.e78.atomic_json(ledger, payload)
        release(submitted)
        payload["released"] = True
        e102.e78.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        if not args.smoke and ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e103] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
