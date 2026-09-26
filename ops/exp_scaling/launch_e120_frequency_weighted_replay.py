#!/usr/bin/env python3
"""Submit E120 fresh-frequency replay ablation behind a non-PVL smoke gate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e78_verified_replay_only_05b as e78  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e80r1_qwen3b_aligned_verified_replay as e80  # noqa: E402
import status_e78 as status  # noqa: E402


PROTOCOL = "paper/preregistration/e120_frequency_weighted_replay_ablation_20260902.md"
AMENDMENT = "paper/preregistration/e120r1_shell_forwarding_repair_20260903.md"
LEDGER = "var/artifacts/e120r1_frequency_weighted_replay_jobs.json"
SMOKE_DIR = "var/data/e120r1_frequency_weighted_replay_smoke_graph_s43"
DOMAINS_PRIMARY = e78.DOMAINS
DOMAINS_SCALE = ("graph_coloring", "pantry_plan")
KEY_WEIGHTING = "fresh_frequency"
PASSES = 8
TRAIN_ROWS = 384
TARGET_STEPS = PASSES * TRAIN_ROWS
CHECKPOINT_INTERVAL = 192
NON_PVL_A5000_NODES = ("node202", "node203", "node204")

MODEL_SPECS = {
    "qwen05b": {
        "model": "Qwen2.5-0.5B-Instruct",
        "domains": DOMAINS_PRIMARY,
        "seeds": e78.SEEDS,
        "comparator_ledger": e78.LEDGER,
    },
    "falcon1b": {
        "model": "tiiuae/Falcon3-1B-Instruct",
        "domains": DOMAINS_SCALE,
        "seeds": e79.SEEDS,
        "comparator_ledger": e79.LEDGER,
    },
    "qwen3b": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "domains": DOMAINS_SCALE,
        "seeds": e80.SEEDS,
        "comparator_ledger": e80.LEDGER,
    },
}


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def refuse_pvl(value: object, *, label: str) -> None:
    if "pvl" in str(value).lower():
        raise RuntimeError(f"E120 refuses PVL in {label}: {value}")


def source_gate(repo: Path) -> dict[str, Any]:
    python = repo / "var/seed_paper_eval/paper310/bin/python"
    library = repo / "var/seed_paper_eval/paper310/lib"
    sources = [
        "src/oat_drgrpo/canonical_replay.py",
        "src/oat_drgrpo/online_canonical_bank.py",
        "src/oat_drgrpo/args.py",
        "src/oat_drgrpo/learner/init.py",
        "src/oat_drgrpo/learner/grpo.py",
        "tests/test_canonical_replay.py",
        "tests/test_online_canonical_bank.py",
        "tests/test_args.py",
        "tests/test_e120_frequency_weighted_replay.py",
    ]
    compile_result = subprocess.run(
        [str(python), "-m", "py_compile", *sources],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if compile_result.returncode:
        raise RuntimeError(f"E120 compile gate failed: {compile_result.stderr}")
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["LD_LIBRARY_PATH"] = str(library)
    test_result = subprocess.run(
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/test_canonical_replay.py",
            "tests/test_online_canonical_bank.py",
            "tests/test_args.py",
            "tests/test_e120_frequency_weighted_replay.py",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if test_result.returncode:
        raise RuntimeError(
            "E120 focused regression gate failed:\n"
            + test_result.stdout
            + test_result.stderr
        )
    return {
        "compile": "passed",
        "pytest": "passed",
        "pytest_tail": test_result.stdout.strip().splitlines()[-1],
        "outcomes_inspected": False,
    }


def templates(repo: Path) -> list[dict[str, Any]]:
    q05 = {
        (str(run["domain"]), int(run["seed"])): run
        for run in e78.references(repo)
    }
    falcon = {
        (str(run["domain"]), int(run["seed"])): run
        for run in e79.references(repo)
        if str(run["domain"]) in DOMAINS_SCALE
    }
    qwen3 = {
        (str(run["domain"]), int(run["seed"])): run
        for run in e80.references(repo)
        if str(run["domain"]) in DOMAINS_SCALE
    }
    result: list[dict[str, Any]] = []
    for model_key, mapping in (
        ("qwen05b", q05),
        ("falcon1b", falcon),
        ("qwen3b", qwen3),
    ):
        spec = MODEL_SPECS[model_key]
        expected = {
            (domain, seed)
            for domain in spec["domains"]
            for seed in spec["seeds"]
        }
        if set(mapping) != expected:
            raise RuntimeError(f"E120 {model_key} template grid mismatch")
        for domain, seed in sorted(expected):
            result.append(
                {
                    "model_key": model_key,
                    "domain": domain,
                    "seed": seed,
                    "template": mapping[(domain, seed)],
                }
            )
    if len(result) != 45:
        raise RuntimeError(f"E120 expected 45 cells, found {len(result)}")
    return result


def terminal_comparators(
    repo: Path, cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ledgers: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    for cell in cells:
        model_key = str(cell["model_key"])
        ledger_name = str(MODEL_SPECS[model_key]["comparator_ledger"])
        payload = ledgers.setdefault(
            ledger_name,
            json.loads((repo / ledger_name).read_text(encoding="utf-8")),
        )
        matches = [
            run
            for run in payload["runs"]
            if str(run["domain"]) == cell["domain"]
            and int(run["seed"]) == cell["seed"]
            and str(run["arm"]) == "replay"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"E120 comparator resolution failed for {cell}")
        run = matches[0]
        run_dir = Path(str(run["run_dir"]))
        target = int(payload["target_steps"])
        observed_step = max(
            status.run_step(run_dir),
            status.receipt_step(run_dir),
        )
        if not status.is_complete(run_dir, observed_step, target):
            raise RuntimeError(
                f"E120 comparator is not terminal: {model_key}/"
                f"{cell['domain']}/s{cell['seed']}"
            )
        selected.append(
            {
                "model_key": model_key,
                "domain": cell["domain"],
                "seed": cell["seed"],
                "run_dir": str(run_dir),
                "target_steps": target,
                "observed_step": observed_step,
                "ledger": ledger_name,
                "ledger_sha256": e78.digest(repo / ledger_name),
            }
        )
    return selected


def run_stamp(model_key: str, domain: str, seed: int) -> str:
    return (
        f"e120r1_{model_key}_{e78.DOMAIN_TAGS[domain]}_"
        f"fresh_frequency_s{seed}"
    )


def save_path(repo: Path, model_key: str, domain: str, seed: int) -> Path:
    return repo / "var/data" / run_stamp(model_key, domain, seed)


def frequency_environment(
    repo: Path,
    cell: dict[str, Any],
    snapshot: Path,
    *,
    smoke: bool = False,
) -> tuple[dict[str, str], Path]:
    model_key = str(cell["model_key"])
    domain = str(cell["domain"])
    seed = int(cell["seed"])
    template = cell["template"]
    if model_key == "qwen05b":
        env, _ = e78.build_env(repo, template, "replay", snapshot)
    elif model_key == "falcon1b":
        env, _ = e79.build_env(
            repo, template, "replay", snapshot, e79.model_root(repo)
        )
    elif model_key == "qwen3b":
        env, _ = e80.build_env(
            repo, template, "replay", snapshot, e80.model_root(repo)
        )
    else:
        raise RuntimeError(f"unknown E120 model: {model_key}")

    target = (
        repo / SMOKE_DIR
        if smoke
        else save_path(repo, model_key, domain, seed)
    )
    stamp = (
        "e120r1_smoke_qwen05b_graph_fresh_frequency_s43"
        if smoke
        else run_stamp(model_key, domain, seed)
    )
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": stamp,
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING": KEY_WEIGHTING,
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED": "0",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS": "0",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT": "0",
            "E120_SMOKE": "1" if smoke else "0",
        }
    )
    if smoke:
        env.update(
            {
                "OAT_ZERO_MAX_TRAIN": "32",
                "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
                "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
                "OAT_ZERO_EVAL_PROMPT_INTERVAL": "32",
                "OAT_ZERO_SAVE_STEPS": "32",
                "OAT_ZERO_SAVE_FROM": "32",
                "OAT_ZERO_RESUME_STEPS": "32",
                "OAT_ZERO_MAX_SAVE_NUM": "1",
                "OAT_ZERO_MAX_RESUME_NUM": "1",
                "OAT_ZERO_AUTO_RESUME": "0",
                "OAT_ZERO_WATCHDOG_REQUEUE": "0",
            }
        )
    return env, target


def placement(cell: dict[str, Any]) -> dict[str, str]:
    model_key = str(cell["model_key"])
    domain = str(cell["domain"])
    seed = int(cell["seed"])
    if model_key == "qwen3b":
        return {
            "partition": "mltheory",
            "account": "mltheory",
            "node": "node302",
            "gpu": "a100",
            "cpus": "16",
            "mem": "128G",
            "time": "3-00:00:00",
        }
    if model_key == "falcon1b":
        node, gpu = e79.placement(domain, seed)
        return {
            "partition": "cs",
            "account": "allcs",
            "node": node,
            "gpu": gpu,
            "cpus": "8",
            "mem": "64G",
            "time": "3-00:00:00",
        }
    domain_offset = DOMAINS_PRIMARY.index(domain)
    node = NON_PVL_A5000_NODES[
        (domain_offset + e78.SEEDS.index(seed)) % len(NON_PVL_A5000_NODES)
    ]
    return {
        "partition": "cs",
        "account": "allcs",
        "node": node,
        "gpu": "a5000",
        "cpus": "8",
        "mem": "64G",
        "time": "1-12:00:00",
    }


def command(
    repo: Path,
    cell: dict[str, Any],
    env: dict[str, str],
    snapshot: Path,
    *,
    dependency: str | None = None,
    smoke: bool = False,
) -> list[str]:
    place = placement(cell)
    name = (
        "e120r1-smoke"
        if smoke
        else f"e120r1-{cell['model_key']}-{e78.DOMAIN_TAGS[cell['domain']][:5]}-s{cell['seed']}"
    )
    exports = ",".join(f"{key}={value}" for key, value in env.items())
    result = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{exports}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={place['node']}",
        f"--gres=gpu:{place['gpu']}:1",
        f"--cpus-per-task={place['cpus']}",
        f"--mem={place['mem']}",
        f"--time={place['time']}",
        "--nice=100",
        "--requeue" if not smoke else "--no-requeue",
        "--chdir=" + str(repo),
    ]
    if dependency is not None:
        result.append(f"--dependency=afterok:{dependency}")
    result.append(str(snapshot / "ops/slurm/e120_resource_fenced_train.slurm"))
    refuse_pvl(" ".join(result), label="submission command")
    return result


def submit(command_parts: list[str]) -> str:
    result = subprocess.run(
        command_parts, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid Slurm job id: {result.stdout!r}")
    return job_id


def audit_held(
    job_id: str,
    cell: dict[str, Any],
    snapshot: Path,
    *,
    dependency: str | None = None,
    smoke: bool = False,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    record = result.stdout
    refuse_pvl(record, label=f"held scheduler record {job_id}")
    place = placement(cell)
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"Account={place['account']}",
        f"Partition={place['partition']}",
        f"ReqNodeList={place['node']}",
        f"gres/gpu:{place['gpu']}:1",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING=fresh_frequency",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        f"E120_SMOKE={1 if smoke else 0}",
    ]
    if dependency is not None:
        required.append(f"Dependency=afterok:{dependency}")
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(f"held E120 job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run")

    repo = root()
    protocol = repo / PROTOCOL
    amendment = repo / AMENDMENT
    ledger = repo / LEDGER
    if not protocol.is_file():
        raise SystemExit(f"missing E120 preregistration: {protocol}")
    if not amendment.is_file():
        raise SystemExit(f"missing E120-R1 amendment: {amendment}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E120 submission: {ledger}")

    cells = templates(repo)
    comparators = terminal_comparators(repo, cells)
    gates = source_gate(repo)
    snapshot = e78.snapshot_util.ensure_snapshot(repo, args.snapshot_root)
    for required in (
        snapshot / "src/oat_drgrpo/canonical_replay.py",
        snapshot / "ops/slurm/e120_resource_fenced_train.slurm",
        snapshot / "ops/exp_scaling/audit_e120_frequency_smoke.py",
    ):
        if not required.is_file():
            raise RuntimeError(f"E120 snapshot lacks {required}")

    smoke_cell = next(
        cell
        for cell in cells
        if cell["model_key"] == "qwen05b"
        and cell["domain"] == "graph_coloring"
        and cell["seed"] == 43
    )
    smoke_env, smoke_target = frequency_environment(
        repo, smoke_cell, snapshot, smoke=True
    )
    if smoke_target.exists():
        raise SystemExit(f"refusing existing E120 smoke directory: {smoke_target}")
    smoke_command = command(
        repo, smoke_cell, smoke_env, snapshot, smoke=True
    )

    planned: list[dict[str, Any]] = []
    for cell in cells:
        env, target = frequency_environment(repo, cell, snapshot)
        if target.exists():
            raise SystemExit(f"refusing existing E120 run directory: {target}")
        planned.append(
            {
                **{key: cell[key] for key in ("model_key", "domain", "seed")},
                "model": MODEL_SPECS[cell["model_key"]]["model"],
                "run_stamp": run_stamp(
                    str(cell["model_key"]), str(cell["domain"]), int(cell["seed"])
                ),
                "run_dir": str(target),
                "environment": env,
                "placement": placement(cell),
                "cell": cell,
            }
        )
    if len(planned) != 45:
        raise RuntimeError("E120 manifest is not exactly 45 cells")

    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(part) for part in smoke_command))
        for item in planned:
            preview = command(
                repo,
                item["cell"],
                item["environment"],
                snapshot,
                dependency="<smoke_job_id>",
            )
            print(" ".join(shlex.quote(part) for part in preview))
        print(
            f"[e120] dry_run=True cells=45 comparators={len(comparators)} "
            f"snapshot={snapshot} pvl=forbidden"
        )
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        smoke_job_id = submit(smoke_command)
        submitted.append(smoke_job_id)
        smoke_record = audit_held(
            smoke_job_id, smoke_cell, snapshot, smoke=True
        )
        for item in planned:
            cell = item["cell"]
            science_command = command(
                repo,
                cell,
                item["environment"],
                snapshot,
                dependency=smoke_job_id,
            )
            job_id = submit(science_command)
            submitted.append(job_id)
            held = audit_held(
                job_id,
                cell,
                snapshot,
                dependency=smoke_job_id,
            )
            records.append(
                {
                    key: item[key]
                    for key in (
                        "model_key",
                        "model",
                        "domain",
                        "seed",
                        "run_stamp",
                        "run_dir",
                        "placement",
                    )
                }
                | {
                    "job_id": int(job_id),
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e120r1_frequency_weighted_replay_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e78.digest(protocol),
            "amendment": str(amendment),
            "amendment_sha256": e78.digest(amendment),
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": e78.digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_identity_sha256": json.loads(
                (snapshot / "SNAPSHOT_IDENTITY.json").read_text(encoding="utf-8")
            )["sha256"],
            "key_weighting": KEY_WEIGHTING,
            "sole_scientific_difference": "within_bank_target_weight_vector",
            "primary_block": {
                "model": MODEL_SPECS["qwen05b"]["model"],
                "domains": list(DOMAINS_PRIMARY),
                "seeds": list(e78.SEEDS),
                "runs": 25,
            },
            "confirmatory_scale_block": {
                "models": [
                    MODEL_SPECS["falcon1b"]["model"],
                    MODEL_SPECS["qwen3b"]["model"],
                ],
                "domains": list(DOMAINS_SCALE),
                "seeds": {
                    "falcon1b": list(e79.SEEDS),
                    "qwen3b": list(e80.SEEDS),
                },
                "runs": 20,
            },
            "passes": PASSES,
            "train_rows": TRAIN_ROWS,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "source_gate": gates,
            "comparators": comparators,
            "pvl_compute_allowed": False,
            "scheduler_record_pvl_substring_audit": "passed",
            "smoke": {
                "job_id": int(smoke_job_id),
                "run_dir": str(smoke_target),
                "held_scheduler_record": smoke_record,
                "science_dependency": f"afterok:{smoke_job_id}",
            },
            "runs": records,
            "released": False,
            "outcomes_inspected_before_release": False,
        }
        e78.atomic_json(ledger, payload)
        for job_id in submitted:
            result = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"failed to release E120 job {job_id}: {result.stderr.strip()}"
                )
        payload["released"] = True
        e78.atomic_json(ledger, payload)
    except Exception:
        e78.cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e120] smoke={smoke_job_id} science={len(records)} "
        f"released={len(submitted)} snapshot={snapshot} ledger={ledger} "
        "pvl=forbidden"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

