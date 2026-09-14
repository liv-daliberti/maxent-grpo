#!/usr/bin/env python3
"""Submit the five-seed Qwen2.5-3B E118 MaxRL/Re:MaxRL extension."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import launch_e76_tuned_scale as snapshot_util
import launch_e80r1_qwen3b_aligned_verified_replay as e80
import launch_e118_maxrl_verified_replay_factorial as e118
import status_e78 as status


DOMAINS = e80.DOMAINS
SEEDS = e80.SEEDS
ARMS = ("maxrl", "replay_maxrl")
VARIANTS = e118.VARIANTS
PASSES = e80.PASSES
TRAIN_ROWS = e80.TRAIN_ROWS
TARGET_STEPS = e80.TARGET_STEPS
CHECKPOINT_INTERVAL = e80.CHECKPOINT_INTERVAL
LEDGER = "var/artifacts/e118q3_maxrl_verified_replay_extension_jobs.json"
PROTOCOL = "paper/preregistration/e118q3_qwen3b_five_seed_extension_20260901.md"
COMPARATOR_LEDGER = "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
NODES = "node103,node104,node205,node206,node207,node208,node302"


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def completed_comparators(repo: Path) -> list[dict[str, Any]]:
    payload = json.loads((repo / COMPARATOR_LEDGER).read_text(encoding="utf-8"))
    selected = [run for run in payload["runs"] if int(run["seed"]) in SEEDS and str(run["domain"]) in DOMAINS and str(run["arm"]) in {"control", "replay"}]
    if len(selected) != 50:
        raise SystemExit("E118-Q3 requires exactly 50 E80-R1 comparator cells")
    for run in selected:
        run_dir = Path(str(run["run_dir"]))
        step = max(status.run_step(run_dir), status.receipt_step(run_dir))
        if not status.is_complete(run_dir, step, TARGET_STEPS):
            raise SystemExit(f"nonterminal comparator: {run['domain']}/{run['arm']}/s{run['seed']}")
    return selected


def run_stamp(domain: str, arm: str, seed: int) -> str:
    return f"e118q3_{e80.DOMAIN_TAGS[domain]}_{arm}_s{seed}"


def save_path(repo: Path, domain: str, arm: str, seed: int) -> Path:
    return repo / "var/data" / f"xdr_{e80.MODEL_TAG}_{VARIANTS[arm]}_{run_stamp(domain, arm, seed)}"


def environment(repo: Path, run: dict[str, Any], arm: str, snapshot: Path, model: Path) -> tuple[dict[str, str], Path]:
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(repo, domain, arm, seed)
    env = e80.base.build_export_vars(repo, run, target, "b1b")
    env.update({
        "OAT_ZERO_PRETRAIN": str(model), "SAVE_PATH": str(target),
        "RUN_STAMP": run_stamp(domain, arm, seed),
        "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES), "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
        "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
        "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL), "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
        "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL), "OAT_ZERO_MAX_SAVE_NUM": "1",
        "OAT_ZERO_MAX_RESUME_NUM": "1", "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "1",
        "OAT_ZERO_AUTO_RESUME": "1", "OAT_ZERO_WATCHDOG_REQUEUE": "1", "OAT_ZERO_USE_WB": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_USE_V1": "0",
    })
    env.update(e80.optimizer_env())
    env.update(e80.memory_env())
    env.update(e118.objective(arm))
    return env, target


def command(repo: Path, run: dict[str, Any], arm: str, env: dict[str, str]) -> list[str]:
    name = f"e118q3-{e80.DOMAIN_TAGS[str(run['domain'])][:6]}-{'m' if arm == 'maxrl' else 'rm'}-s{run['seed']}"
    exports = ",".join(f"{key}={value}" for key, value in env.items())
    script = Path(env["OAT_ZERO_OPS_SNAPSHOT_ROOT"]) / "slurm/train_node302.slurm"
    return ["sbatch", "--parsable", "--hold", f"--job-name={name}", f"--export=ALL,{exports}",
        "--partition=all", "--account=allcs", f"--nodelist={NODES}", "--gres=gpu:1",
        "--cpus-per-task=16", "--mem=128G", "--time=12:00:00", "--nice=0", "--requeue",
        "--chdir=" + str(repo), str(script)]


def audit(job_id: str, run: dict[str, Any], arm: str, model: Path) -> str:
    result = subprocess.run(["scontrol", "show", "job", "-dd", "-o", job_id], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    record = result.stdout
    required = ("JobState=PENDING", "Reason=JobHeldUser", "Account=allcs",
        "TresPerNode=gres/gpu:1", f"OAT_ZERO_PRETRAIN={model}",
        "OAT_ZERO_MAXRL_TASK_OBJECTIVE=1", "OAT_ZERO_LEARNING_RATE=1e-07", "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_ADAM_OFFLOAD=1", "OAT_ZERO_ACTIVATION_OFFLOADING=1", "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=" + ("1" if arm == "maxrl" else "0"),
        f"OAT_ZERO_SEED={run['seed']}")
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held E118-Q3 job {job_id} lacks {missing}")
    return record


def main() -> int:
    repo = root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run")
    protocol, ledger = repo / PROTOCOL, repo / LEDGER
    for required in (protocol, repo / COMPARATOR_LEDGER):
        if not required.is_file():
            raise SystemExit(f"required input absent: {required}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate submission: {ledger}")
    comparators = completed_comparators(repo)
    templates = e80.references(repo)
    model = e80.model_root(repo)
    snapshot = snapshot_util.ensure_snapshot(repo, args.snapshot_root)
    planned: list[dict[str, Any]] = []
    for run in templates:
        for arm in ARMS:
            env, target = environment(repo, run, arm, snapshot, model)
            if target.exists():
                raise SystemExit(f"refusing existing run directory: {target}")
            planned.append({"domain": str(run["domain"]), "arm": arm, "seed": int(run["seed"]),
                "run_stamp": run_stamp(str(run["domain"]), arm, int(run["seed"])), "run_dir": str(target),
                "template": run, "command": command(repo, run, arm, env)})
    if len(planned) != 50:
        raise SystemExit(f"expected 50 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(f"[e118q3] dry_run=True cells=50 snapshot={snapshot}")
        return 0
    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(cell["command"], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held = audit(job_id, cell["template"], str(cell["arm"]), model)
            records.append({key: cell[key] for key in ("domain", "arm", "seed", "run_stamp", "run_dir")} | {"job_id": int(job_id), "held_scheduler_record": held})
        payload = {"schema": "e118q3_maxrl_verified_replay_extension_jobs_v1", "protocol": str(protocol),
            "protocol_sha256": e80.digest(protocol), "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": e80.digest(Path(__file__)), "comparator_ledger": str(repo / COMPARATOR_LEDGER),
            "comparator_ledger_sha256": e80.digest(repo / COMPARATOR_LEDGER), "snapshot_root": str(snapshot),
            "model": "Qwen/Qwen2.5-3B-Instruct", "model_revision": e80.MODEL_REVISION,
            "domains": list(DOMAINS), "new_arms": list(ARMS), "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS, "passes": PASSES, "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL, "replay_weight": e80.REPLAY_WEIGHT,
            "comparators": comparators, "runs": records, "released": False, "outcomes_inspected_before_release": False}
        e80.atomic_json(ledger, payload)
        for job_id in submitted:
            result = subprocess.run(["scontrol", "release", job_id], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"release failed for {job_id}: {result.stderr.strip()}")
        payload["released"] = True
        e80.atomic_json(ledger, payload)
    except Exception:
        e80.cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise
    print(f"[e118q3] cells=50 released=50 snapshot={snapshot} ledger={ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
