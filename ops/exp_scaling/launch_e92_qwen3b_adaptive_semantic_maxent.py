#!/usr/bin/env python3
"""Submit E92: adaptive semantic MaxEnt on top of verified replay, Qwen2.5-3B.

The third-family replication of E89, at seed 70 to pair with E87. Everything about the arm -- controller,
bound, gain, EMA decay, refusal rule, and the registered target ratio
`rho = .015` -- is inherited from E89 unchanged; only the model family and its
cells change. Reusing one target across families is what makes "uniform
semantic pressure" a single treatment rather than three tuned ones.

`rho = .015` was checked for reachability on this family before submission, not
assumed. At the fixed `eta = .10` the realized ratio in E87 is .0121 (MathIR,
the binding domain), .0187 (Graph coloring), .0274 (Countdown) and .0307
(PantryPlan). The applied semantic advantage is exactly proportional to `eta`,
so the `eta <= .40` ceiling reaches about 4x those values; the binding domain
tops out near .0484 and `rho = .015` sits at 31% of it. Python factors is
excluded from the binding calculation: at 3B its task-advantage RMS is
near zero for most updates, so its ratio (.2658) measures a vanishing
denominator rather than headroom.

Cells, seed, schedule, decoding, placement and code are inherited from E87,
which differs from this arm only in how `eta` is set. E80-R1 and E87 are never
re-run.
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
import launch_e80r1_qwen3b_aligned_verified_replay as e80r1  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e87_qwen3b_semantic_maxent_seed70 as e87  # noqa: E402
import launch_e88_adaptive_semantic_maxent_05b as e88  # noqa: E402


DOMAINS = e87.DOMAINS
ARM = "adaptive_semantic_reachable"
SEED = e87.SEED
PASSES = e87.PASSES
TRAIN_ROWS = e87.TRAIN_ROWS
CHECKPOINT_INTERVAL = e87.CHECKPOINT_INTERVAL
TARGET_STEPS = e87.TARGET_STEPS
VARIANT = "adaptive_semantic_maxent_replay"
MODEL_TAG = e87.MODEL_TAG

# Inherited from E89 verbatim. Changing any of these would make E92 a different
# treatment from the arm it replicates.
TARGET_RATIO = 0.015
MAX_COEFFICIENT = e88.MAX_COEFFICIENT

# Measured on this family before submission; see the module docstring.
BINDING_DOMAIN = "mathir"
BINDING_RATIO_AT_FIXED_ETA = 0.0121

LEDGER = "var/artifacts/e92_qwen3b_adaptive_semantic_maxent_jobs.json"
PROTOCOL = "paper/preregistration/e92_qwen3b_adaptive_semantic_maxent_20260811.md"
SOURCE_MANIFEST = e81.SOURCE_MANIFEST
PAIR_LEDGER = e87.PAIR_LEDGER
E87_LEDGER = e87.LEDGER
SNAPSHOT_PREFIX = "e92_qwen3b_adaptive_semantic"

# E82's own patch set predates the controller; inheriting it would produce cells
# that silently run at the fixed coefficient. E90's first submission failed
# exactly that way, so the adaptive patch set is taken from E88, which carries
# the controller module and the learner that drives it.
PATCHED_FILES = e88.PATCHED_FILES

SNAPSHOT_REQUIREMENTS = (
    ("src/oat_drgrpo/semantic_rms_controller.py", "class SemanticRmsController"),
    ("src/oat_drgrpo/args.py", "semantic_rms_control"),
    ("src/oat_drgrpo/learner/grpo.py", "semantic_rms"),
    ("ops/run_experiment.sh", "adaptive_semantic_maxent_replay)"),
    ("ops/train.sh", "SEMANTIC_RMS_TARGET_RATIO"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_snapshot(snapshot_root: Path) -> None:
    """Fail before submission if the frozen runtime cannot adapt the coefficient."""

    missing = []
    for name, needle in SNAPSHOT_REQUIREMENTS:
        path = snapshot_root / name
        if not path.is_file() or needle not in path.read_text(encoding="utf-8"):
            missing.append(f"{name}: {needle!r}")
    if missing:
        raise SystemExit(
            "frozen snapshot cannot run the RMS controller, so these cells "
            "would silently run as the fixed-coefficient arm:\n  "
            + "\n  ".join(missing)
        )


def run_stamp(domain: str, seed: int) -> str:
    return f"e92_qwen3b_adaptive_{e80r1.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective() -> dict[str, str]:
    """E82's objective with the coefficient control as the only change.

    Built from the comparator's own launcher and E88's controller block, so the
    inheritance is checkable. The assertion pins exactly which keys move.
    """

    baseline = dict(e87.fixed_objective())
    objective = dict(baseline)
    controller = {
        key: value
        for key, value in e88.fixed_objective().items()
        if "SEMANTIC_RMS" in key
    }
    objective.update(controller)
    objective["OAT_ZERO_SEMANTIC_RMS_TARGET_RATIO"] = repr(TARGET_RATIO)
    objective["OAT_ZERO_VARIANT"] = VARIANT

    moved = {
        key
        for key in set(objective) | set(baseline)
        if objective.get(key) != baseline.get(key)
    }
    expected = set(controller) | {"OAT_ZERO_VARIANT"}
    if moved != expected:
        raise RuntimeError(
            f"E92 differs from E82 beyond the coefficient control: "
            f"{sorted(moved ^ expected)}"
        )
    if TARGET_RATIO >= 0.88 * 4.0 * BINDING_RATIO_AT_FIXED_ETA:
        raise RuntimeError(
            f"rho={TARGET_RATIO} is not reachable on {BINDING_DOMAIN}: the "
            f"eta<={MAX_COEFFICIENT} ceiling tops out near "
            f"{4.0 * BINDING_RATIO_AT_FIXED_ETA:.4f}"
        )
    return objective


def build_env(
    root: Path, run: dict[str, Any], snapshot_root: Path, qwen_root: Path
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    env, _ = e87.build_env(root, run, snapshot_root, qwen_root)
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(fixed_objective())
    return env, target


def sbatch_command(
    root: Path, run: dict[str, Any], env: dict[str, str], *, nice: int
) -> list[str]:
    base = e87.sbatch_command(root, run, env)
    name = f"e92-{e80r1.DOMAIN_TAGS[str(run['domain'])][:6]}-adp-s{run['seed']}"
    out: list[str] = []
    for token in base:
        if token.startswith("--job-name="):
            out.append(f"--job-name={name}")
        elif token.startswith("--nice="):
            out.append(f"--nice={nice}")
        else:
            out.append(token)
    if not any(t.startswith("--nice=") for t in out):
        out.insert(-1, f"--nice={nice}")
    return out


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument(
        "--nice",
        type=int,
        default=0,
        help="scheduler nice; lower runs sooner (existing cohorts use 100)",
    )
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol = root / PROTOCOL
    for required in (protocol, root / SOURCE_MANIFEST):
        if not required.is_file():
            raise SystemExit(f"required frozen E92 input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E92 submission: {ledger}")

    qwen_root = e80r1.model_root(root)
    source_runs = e80r1.references(root)
    base_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(json.loads((root / PAIR_LEDGER).read_text(encoding="utf-8"))["snapshot_root"])
    )
    snapshot_root, patched = e81.ensure_paired_snapshot(
        root, base_snapshot, prefix=SNAPSHOT_PREFIX, patched_files=PATCHED_FILES
    )
    verify_snapshot(snapshot_root)

    planned: list[dict[str, Any]] = []
    for run in source_runs:
        domain = str(run["domain"])
        seed = int(run["seed"])
        if domain not in DOMAINS or seed != SEED:
            continue
        env, target = build_env(root, run, snapshot_root, qwen_root)
        planned.append(
            {
                "arm": ARM,
                "domain": domain,
                "seed": seed,
                "run_dir": str(target),
                "run_stamp": run_stamp(domain, seed),
                "env": env,
                "source": run,
            }
        )

    if len(planned) != len(DOMAINS):
        raise SystemExit(
            f"expected {len(DOMAINS)} E92 cells, planned {len(planned)}"
        )

    if args.dry_run or not args.submit:
        for cell in planned:
            command = sbatch_command(root, cell["source"], cell["env"], nice=args.nice)
            print(f"{cell['domain']:<16} s{cell['seed']}  {shlex.join(command[:6])}")
        print(f"\n{len(planned)} cells planned at nice={args.nice}; "
              f"rho={TARGET_RATIO} (reachable: binding domain {BINDING_DOMAIN} "
              f"tops out near {4 * BINDING_RATIO_AT_FIXED_ETA:.4f})")
        print(f"snapshot: {snapshot_root}  patched files: {len(patched)}")
        return 0

    submitted: list[dict[str, Any]] = []
    job_ids: list[str] = []
    try:
        for cell in planned:
            command = sbatch_command(root, cell["source"], cell["env"], nice=args.nice)
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip()
            job_ids.append(job_id)
            submitted.append(
                {
                    **{k: v for k, v in cell.items() if k not in ("env", "source")},
                    "job_id": job_id,
                    "source_node": str(cell["source"].get("source_node", "")),
                }
            )
    except Exception:
        if job_ids:
            subprocess.run(["scancel", *job_ids], check=False)
        raise

    payload = {
        "schema": "e92_qwen3b_adaptive_semantic_maxent_jobs_v1",
        "cohort": "e92",
        "released": True,
        "model": "Qwen2.5-3B-Instruct",
        "protocol": PROTOCOL,
        "protocol_sha256": e80r1.digest(protocol),
        "arms": [ARM],
        "inherited_arms": ["control", "replay"],
        "pair_ledger": str(root / PAIR_LEDGER),
        "comparator_ledger": E87_LEDGER,
        "objective": "verified_likelihood_plus_rms_targeted_semantic_maxent",
        "scientific_difference": (
            "against E82: the coefficient control only, fixed eta -> "
            f"RMS-targeted at rho={TARGET_RATIO}"
        ),
        "controller": {
            k: v for k, v in fixed_objective().items() if "SEMANTIC_RMS" in k
        },
        "target_ratio": TARGET_RATIO,
        "target_ratio_derivation": (
            f"binding domain {BINDING_DOMAIN} realizes "
            f"{BINDING_RATIO_AT_FIXED_ETA} at eta=.10; the eta<={MAX_COEFFICIENT} "
            f"ceiling reaches ~{4 * BINDING_RATIO_AT_FIXED_ETA:.4f}"
        ),
        "snapshot_root": str(snapshot_root),
        "snapshot_patched_files": len(patched),
        "variant": VARIANT,
        "domains": list(DOMAINS),
        "seeds": [SEED],
        "passes": PASSES,
        "registered_passes": [i / 2 for i in range(2 * PASSES + 1)],
        "train_rows": TRAIN_ROWS,
        "target_steps": TARGET_STEPS,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
        "nice": args.nice,
        "runs": submitted,
    }
    ledger.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"submitted {len(submitted)} E92 cells at nice={args.nice}; ledger {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
