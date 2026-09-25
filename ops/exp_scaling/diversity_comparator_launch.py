#!/usr/bin/env python3
"""Shared submission machinery for the E126/E127/E128 diversity comparators.

GAPO and SetPO answer the same reviewer question --- how does Re:Max compare
against recent diversity-preserving RLVR methods --- on the same panel, with the
same schedule, against the same controls. E128 supplies those controls. All
three cohorts share one audited submission path, one runtime snapshot, and one
placement, and differ only by objective.

Two departures from every earlier Qwen2.5-0.5B comparator are deliberate, and
together they are why the control arm is re-run rather than inherited.

Placement. E97/E115 inherited the paired E78 control's node cell by cell, which
means node302 (A100) or node105 (A5000), both reachable only through the
mltheory account. These cohorts run on the ``cs`` partition, pinned to its
A5000 nodes. Read against the E78 controls that would put a GPU-model term
inside the paired effect, since 36 of those 50 cells ran on an A100 and this
campaign has already observed that terminal values cluster by GPU pool.

Runtime. The E78 runtime no longer exists and cannot be rebuilt (see
``BASE_SNAPSHOT``), so a new arm cannot inherit it at all, and the size of the
difference cannot even be measured.

Neither confound survives the E128 control: every cohort here trains on the
same GPU model, under the same snapshot, on the same schedule, and the paired
difference is taken within this set. The E78 record keeps its own pairings for
the comparators that were trained against it.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e78_verified_replay_only_05b as e78  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


#: The full Level-1 panel. E97 covered three domains and E115 completed the
#: other two; these cohorts register all five at once.
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
SEEDS = e81.SEEDS
PASSES = e81.PASSES
TRAIN_ROWS = e81.TRAIN_ROWS
CHECKPOINT_INTERVAL = e81.CHECKPOINT_INTERVAL
TARGET_STEPS = e81.TARGET_STEPS
PAIR_LEDGER = e81.PAIR_LEDGER
SOURCE_MANIFEST = e81.SOURCE_MANIFEST

#: The ``cs`` partition admits the ``allcs`` account only, which removes the
#: silent-landing hazard a partition/account mismatch causes elsewhere: an
#: mltheory submission to cs is refused rather than rerouted. Verified against
#: ``scontrol show partition cs`` (AllowAccounts=allcs) before submission.
PARTITION = "cs"
ACCOUNT = "allcs"
GRES = "gpu:a5000:1"
#: node202 is the third A5000 in ``cs`` and is draining; it is excluded so a
#: held job cannot be admitted onto a node that will not run it.
NODELIST = "node203,node204"
CPUS_PER_TASK = 8
MEMORY = "64G"
TIME_LIMIT = "1-12:00:00"
NICE = "100"
SLURM_SCRIPT = "ops/slurm/train_node302.slurm"

#: The base runtime these cohorts derive from, and the reason it is not E78's.
#:
#: E78's ledger pins ``e76_tuned_scale_96f68ebb47757af8``. The user-approved,
#: irreversible cleanup of 2026-09-04 retired 528 source snapshots "not
#: referenced by live queue", including that one and the E97 and E81 runtimes;
#: no commit in the repository's history reproduces the tree, so it was built
#: from a dirty working tree and cannot be rebuilt. New cells therefore cannot
#: inherit the runtime the E78 controls trained under, which is precisely why
#: E128 re-runs matched controls instead of reusing them.
#:
#: This base is the one surviving ``e76_tuned_scale`` snapshot that still
#: hashes to its own recorded identity --- the other three have been modified
#: in place since creation --- and it is the runtime E121 ran on.
BASE_SNAPSHOT = "var/artifacts/source_snapshots/e76_tuned_scale_970e16dc21f47834"

#: One snapshot serves all three cohorts, so "same runtime" is checkable by
#: path equality across the three ledgers rather than by comparing digests.
SNAPSHOT_PREFIX = "diversity_comparators"

#: Files these cohorts are permitted to change relative to the base runtime.
#: The objective additions are inert by default: the new args default off, the
#: learner blocks are unreachable unless a new flag is set, and the two shell
#: branches are unreachable for any other variant string.
#:
#: ``learner/run.py`` is the one non-additive entry and is deliberate. The base
#: runtime predates ``eval_mode_coverage_disjoint_draws``, under which a
#: prompt's consecutive evaluation draws share vLLM child streams and pooled
#: mode-coverage statistics count repeats as independent. All three cohorts
#: take the corrected evaluator, so they are measured alike; the E78 record was
#: measured under the old pooling, which is a further reason its controls are
#: not the comparator here.
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "src/oat_drgrpo/learner/grpo.py",
    "src/oat_drgrpo/learner/run.py",
    "src/oat_drgrpo/gapo.py",
    "src/oat_drgrpo/setpo.py",
    "src/oat_drgrpo/setpo_embedder.py",
    # ``ucpo.py`` and ``rlep.py`` are imported unconditionally by the learner
    # and are deliberately *not* declared: they are byte-identical to the base
    # runtime, and the snapshot audit requires the declared set to equal the
    # observed divergence exactly, so naming an unchanged file would fail the
    # audit rather than strengthen it.
    # The capacity floor lives in two files. ``args.py`` carries the
    # objective-dependent minimum and ``online_canonical_bank.py`` the
    # constructor guard; E130 needs both to run a one-slot bank. Declaring only
    # the first is what let a half-patched snapshot pass verify_snapshot on
    # 2026-09-22 and killed all 25 E130 cells behind a failed smoke job. The
    # relaxed guard is inert for every arm at capacity >= 2.
    "src/oat_drgrpo/online_canonical_bank.py",
    "ops/run_experiment.sh",
    "ops/train.sh",
)


def default_common_exports(root: Path) -> tuple[str, ...]:
    """Keys every E126/E127/E128 held job must carry.

    These three cohorts differ from the control only in advantage shaping, so
    the replay traversal stays inert and the horizon stays at eight passes. An
    arm that moves either one supplies its own tuple through
    ``Cohort.common_exports`` rather than weakening this default, so the
    scheduler record is still audited against exactly what the arm requested.
    """

    return (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={PASSES}",
        f"OAT_ZERO_MAX_PROMPT_EPOCHS={PASSES}",
    )


@dataclass(frozen=True)
class Cohort:
    """One comparator cohort's identity and objective."""

    tag: str
    arm: str
    variant: str
    title: str
    ledger: str
    protocol: str
    objective_summary: str
    scientific_difference: str
    #: Extra ``OAT_ZERO_*`` keys layered onto the E78 control objective.
    objective_overrides: Callable[[Path], dict[str, str]]
    #: ``(relative path, needle)`` pairs the frozen snapshot must satisfy.
    snapshot_requirements: tuple[tuple[str, str], ...]
    #: Scheduler export keys every held scientific job must carry.
    expected_exports: Callable[[Path], tuple[str, ...]]
    #: Extra fields recorded in the ledger payload.
    ledger_extras: Callable[[Path], dict[str, Any]]
    #: Scheduler keys shared by every scientific job of this cohort. The
    #: default is the inert-replay, eight-pass contract E126/E127/E128 run
    #: under; an arm that changes the replay derivative or the horizon must
    #: state its own, so nothing goes unaudited.
    common_exports: Callable[[Path], tuple[str, ...]] = field(
        default=default_common_exports
    )
    #: Prompt passes this cohort trains for. Only an arm that deliberately
    #: extends the horizon moves it, and it must also export the matching
    #: epoch keys through ``objective_overrides`` so the request and the
    #: recorded schedule cannot disagree.
    passes: int = PASSES
    #: Files this cohort replaces relative to the base runtime. The default is
    #: the shared E126--E133 patch set; an arm that touches another file must
    #: declare it here, because the snapshot audit requires the declared set
    #: to equal the observed divergence exactly.
    patched_files: tuple[str, ...] = PATCHED_FILES
    #: Extra scheduler keys layered onto the smoke job. The default smoke runs
    #: one pass over 32 prompts; an arm whose mechanism only engages when a
    #: prompt is revisited states a longer smoke here.
    smoke_overrides: dict[str, str] = field(default_factory=dict)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def selected_references(root: Path) -> list[dict[str, Any]]:
    runs = [r for r in e81.references(root) if str(r["domain"]) in DOMAINS]
    expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    found = {(str(r["domain"]), int(r["seed"])) for r in runs}
    if found != expected or len(runs) != len(expected):
        raise SystemExit(
            f"source manifest does not cover exactly {len(expected)} cells"
        )
    return sorted(runs, key=lambda r: (str(r["domain"]), int(r["seed"])))


def selected_pairs(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    pairs = e81.pair_index(e81.pair_ledger(root))
    return {key: value for key, value in pairs.items() if key[0] in DOMAINS}


def verify_snapshot(snapshot: Path, cohort: Cohort) -> None:
    missing = [
        f"{name}: {needle!r}"
        for name, needle in cohort.snapshot_requirements
        if needle not in (snapshot / name).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit(
            f"{cohort.tag} snapshot is not {cohort.arm}-capable:\n  "
            + "\n  ".join(missing)
        )


def run_stamp(cohort: Cohort, domain: str, seed: int) -> str:
    return f"{cohort.tag}_{cohort.arm}_{e81.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(root: Path, cohort: Cohort, domain: str, seed: int) -> Path:
    stamp = run_stamp(cohort, domain, seed)
    return root / "var/data" / f"xdr_{e81.MODEL_TAG}_{cohort.variant}_{stamp}"


def objective(root: Path, cohort: Cohort) -> dict[str, str]:
    """Return the E78 control objective with only this arm's keys moved."""

    baseline = dict(e78.fixed_objective("control"))
    result = dict(baseline)
    # Inert guards first: state every comparator off rather than relying on a
    # runtime default, so a scheduler record shows what was never requested.
    # The cohort's own overrides are applied afterwards and are the only way a
    # guard may be lifted.
    guards = {
        "OAT_ZERO_UCPO_TAU": "0.0",
        "OAT_ZERO_RLEP_EXPERIENCE_ROOT": "",
        "OAT_ZERO_RLEP_REPLAY_COUNT": "0",
        "OAT_ZERO_DAPO_ENABLED": "0",
        "OAT_ZERO_GAPO_ENABLED": "0",
        "OAT_ZERO_GAPO_SUPPORT_INDEX": "",
        "OAT_ZERO_SETPO_COEFFICIENT": "0.0",
        "OAT_ZERO_SETPO_EMBEDDER_PATH": "",
    }
    result.update(guards)
    overrides = cohort.objective_overrides(root)
    result.update(overrides)
    moved = {
        key
        for key in set(result) | set(baseline)
        if result.get(key) != baseline.get(key)
    }
    expected = set(overrides) | set(guards)
    unexpected = sorted(moved - expected)
    if unexpected:
        raise RuntimeError(
            f"{cohort.tag} objective drift relative to E78 control: {unexpected}"
        )
    return result


def build_env(
    root: Path,
    cohort: Cohort,
    run: dict[str, Any],
    snapshot: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = e81.build_env(root, run, snapshot)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(root, cohort, domain, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(cohort, domain, seed),
        }
    )
    env.update(objective(root, cohort))
    return env, target


def smoke_env(
    root: Path,
    cohort: Cohort,
    run: dict[str, Any],
    snapshot: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = build_env(root, cohort, run, snapshot)
    stamp = f"{cohort.tag}_{cohort.arm}_smoke_graph_s43"
    target = root / "var/data" / stamp
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": stamp,
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
    env.update(cohort.smoke_overrides)
    return env, target


def job_name(cohort: Cohort, domain: str | None = None, seed: int | None = None) -> str:
    if domain is None:
        return f"{cohort.tag}-{cohort.arm}-smoke"
    tag = e81.DOMAIN_TAGS[domain][:6]
    return f"{cohort.tag}-{tag}-{cohort.arm}-s{seed}"


def sbatch_command(
    root: Path,
    cohort: Cohort,
    env: dict[str, str],
    *,
    name: str,
    dependency: str = "",
) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{','.join(f'{k}={v}' for k, v in env.items())}",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--nodelist={NODELIST}",
        f"--gres={GRES}",
        f"--cpus-per-task={CPUS_PER_TASK}",
        f"--mem={MEMORY}",
        f"--time={TIME_LIMIT}",
        f"--nice={NICE}",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(root / SLURM_SCRIPT))
    return command


def submit_held(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch response: {result.stdout!r}")
    return job_id


def audit_held(job_id: str, *, name: str, expected: tuple[str, ...]) -> str:
    """Read the scheduler's own record back and refuse any divergence.

    Partition and account are audited explicitly. A partition request that the
    scheduler reroutes is exactly the failure this campaign has hit before, and
    the only trustworthy evidence of where a job will run is the record Slurm
    wrote, not the flags that were sent.
    """

    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName={name}",
        f"Partition={PARTITION}",
        f"Account={ACCOUNT}",
        "TresPerNode=gres/gpu:a5000:1",
    ) + expected
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held job {job_id} lacks {missing}")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def main(cohort: Cohort, argv: list[str] | None = None) -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=cohort.title)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args(argv)
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol = root / cohort.protocol
    ledger = root / cohort.ledger
    for required in (protocol, root / SOURCE_MANIFEST, root / PAIR_LEDGER):
        if not required.is_file():
            raise SystemExit(f"required input is absent: {required}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate {cohort.tag} submission: {ledger}")

    pairs = selected_pairs(root)
    runs = selected_references(root)
    base_snapshot = (
        args.snapshot_root.resolve() if args.snapshot_root else root / BASE_SNAPSHOT
    )
    if not (base_snapshot / "src/oat_drgrpo/__init__.py").is_file():
        raise SystemExit(
            f"base runtime is absent: {base_snapshot}\n"
            "the E78 runtime was retired by the 2026-09-04 cleanup and cannot "
            "be rebuilt; see BASE_SNAPSHOT"
        )
    snapshot, patched = e81.ensure_paired_snapshot(
        root,
        base_snapshot,
        prefix=SNAPSHOT_PREFIX,
        patched_files=cohort.patched_files,
    )
    verify_snapshot(snapshot, cohort)

    cells: list[dict[str, Any]] = []
    for run in runs:
        domain, seed = str(run["domain"]), int(run["seed"])
        env, target = build_env(root, cohort, run, snapshot)
        if target.exists():
            raise SystemExit(f"refusing to overwrite output: {target}")
        cells.append(
            {"run": run, "env": env, "target": target, "pair": pairs[(domain, seed)]}
        )
    smoke_run = next(
        r for r in runs if r["domain"] == "graph_coloring" and r["seed"] == 43
    )
    smoke_vars, smoke_target = smoke_env(root, cohort, smoke_run, snapshot)
    if smoke_target.exists():
        raise SystemExit(f"refusing to overwrite smoke output: {smoke_target}")

    expected_exports = cohort.expected_exports(root)
    if args.dry_run or not args.submit:
        print(
            shlex.join(
                sbatch_command(
                    root, cohort, smoke_vars, name=job_name(cohort)
                )
            )
        )
        for cell in cells:
            run = cell["run"]
            print(
                shlex.join(
                    sbatch_command(
                        root,
                        cohort,
                        cell["env"],
                        name=job_name(
                            cohort, str(run["domain"]), int(run["seed"])
                        ),
                        dependency="SMOKE_JOB_ID",
                    )
                )
            )
        print(
            f"[{cohort.tag}] smoke=1 scientific={len(cells)} "
            f"snapshot={snapshot} patched={patched}"
        )
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        smoke_name = job_name(cohort)
        smoke_id = submit_held(
            sbatch_command(root, cohort, smoke_vars, name=smoke_name)
        )
        submitted.append(smoke_id)
        smoke_queries = smoke_vars["OAT_ZERO_MAX_QUERIES"]
        smoke_record = audit_held(
            smoke_id,
            name=smoke_name,
            expected=expected_exports + (f"OAT_ZERO_MAX_QUERIES={smoke_queries}",),
        )
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            name = job_name(cohort, domain, seed)
            job_id = submit_held(
                sbatch_command(
                    root, cohort, cell["env"], name=name, dependency=smoke_id
                )
            )
            submitted.append(job_id)
            held = audit_held(
                job_id,
                name=name,
                expected=(
                    f"Dependency=afterok:{smoke_id}",
                    f"OAT_ZERO_SEED={seed}",
                )
                + cohort.common_exports(root)
                + expected_exports,
            )
            control = cell["pair"]["control"]
            records.append(
                {
                    "domain": domain,
                    "arm": cohort.arm,
                    "seed": seed,
                    "run_stamp": run_stamp(cohort, domain, seed),
                    "run_dir": str(cell["target"]),
                    "job_id": int(job_id),
                    "smoke_dependency_job_id": int(smoke_id),
                    "paired_e78_control": {
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                        "source_node": str(control["source_node"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": f"{cohort.tag}_{cohort.arm}_05b_jobs_v1",
            "cohort": cohort.tag,
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
            "snapshot_base": str(base_snapshot),
            "snapshot_patched_files": patched,
            "snapshot_note": (
                "the E78 runtime e76_tuned_scale_96f68ebb47757af8 was retired "
                "by the irreversible 2026-09-04 cleanup and is not "
                "reconstructible from any commit; E128 supplies matched "
                "controls under this runtime instead"
            ),
            "domains": list(DOMAINS),
            "seeds": list(SEEDS),
            "arms": [cohort.arm],
            "inherited_arms": ["control"],
            "passes": cohort.passes,
            "train_rows": TRAIN_ROWS,
            "target_steps": TRAIN_ROWS * cohort.passes,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [i / 2 for i in range(2 * cohort.passes + 1)],
            "variant": cohort.variant,
            "objective": cohort.objective_summary,
            "scientific_difference": cohort.scientific_difference,
            "placement": {
                "partition": PARTITION,
                "account": ACCOUNT,
                "gres": GRES,
                "nodelist": NODELIST,
                "node_inherited_from_control": False,
                "note": (
                    "cs A5000 placement; the E78 controls span node302 (A100) "
                    "and node105 (A5000), so pairings against node302 "
                    "controls are not GPU-model matched"
                ),
            },
            "smoke": {
                "scientific": False,
                "job_id": int(smoke_id),
                "run_dir": str(smoke_target),
                "max_queries": int(smoke_queries),
                "held_scheduler_record": smoke_record,
            },
            "runs": records,
        }
        payload.update(cohort.ledger_extras(root))
        e81.atomic_json(ledger, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        raise
    print(
        f"[{cohort.tag}] released smoke {smoke_id} and "
        f"{len(records)} dependent scientific cells"
    )
    print(f"[{cohort.tag}] ledger {ledger}")
    return 0
