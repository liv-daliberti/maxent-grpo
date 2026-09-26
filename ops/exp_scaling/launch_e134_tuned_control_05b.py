#!/usr/bin/env python3
"""Submit E134 stage 1: the Dr.GRPO control swept over its own hyperparameters.

The reviewer objection is that the control is fragile --- one 16-rollout group
per update, no KL, no entropy term, an unswept learning rate --- so replay's
gain may be rescue of a degenerate baseline. E134 tunes the control and then
runs Re:Dr at the setting the tuning picks. Stage 1 is the sweep; the selection
rule and stage 2 are frozen in the protocol before any cell runs.

Each setting is one ``shared.Cohort`` so every cell passes through the same
drift guard and held-record audit as E128-E133. What this launcher adds over
``shared.main`` is a seed subset (selection runs on seeds 43 and 44 only),
several settings in one ledger, and one smoke per mechanical path: four prompts
per update is a batching path no earlier cohort on this panel exercised, and
the entropy term is a loss path none of them enabled.

Four prompts per update holds the fresh-rollout budget, not the update count:
the save and resume cadence is counted in updates, so it is divided by four to
keep checkpoints every 192 prompts. The evaluation interval is counted in
prompts and is left alone.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402

e81 = shared.e81

TAG = "e134"
SELECTION_SEEDS = (43, 44)
LEDGER = "var/artifacts/e134_tuned_control_05b_stage1_jobs.json"
PROTOCOL = "paper/preregistration/e134_tuned_control_05b_20260924.md"
BASELINE_LR = "2e-07"


@dataclass(frozen=True)
class Setting:
    arm: str
    lr: str
    prompts_per_update: int = 1
    entropy: str = "0.0"


SETTINGS = (
    Setting("tc_lr5e8_b1", "5e-08"),
    Setting("tc_lr1e7_b1", "1e-07"),
    Setting("tc_lr5e7_b1", "5e-07"),
    Setting("tc_lr1e6_b1", "1e-06"),
    Setting("tc_lr2e7_b4", BASELINE_LR, 4),
    Setting("tc_lr5e7_b4", "5e-07", 4),
    Setting("tc_lr1e6_b4", "1e-06", 4),
    Setting("tc_ent1e3_b1", BASELINE_LR, 1, "0.001"),
)
#: One smoke per mechanical path. Every batch-1 cell waits on the entropy
#: smoke, which also exercises the batch-1 path; every batch-4 cell waits on
#: the batch-4 smoke.
SMOKE_FOR_BATCH = {1: "tc_ent1e3_b1", 4: "tc_lr1e6_b4"}


def overrides(setting: Setting) -> dict[str, str]:
    keys = {"OAT_ZERO_LEARNING_RATE": setting.lr}
    if setting.entropy != "0.0":
        keys["OAT_ZERO_POLICY_ENTROPY_COEF"] = setting.entropy
    if setting.prompts_per_update != 1:
        width = setting.prompts_per_update * 16
        cadence = str(e81.CHECKPOINT_INTERVAL // setting.prompts_per_update)
        keys.update(
            {
                "OAT_ZERO_ROLLOUT_BATCH_SIZE": str(setting.prompts_per_update),
                "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE": str(
                    setting.prompts_per_update
                ),
                "OAT_ZERO_TRAIN_BATCH_SIZE": str(width),
                "OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE": str(width),
                "OAT_ZERO_SAVE_STEPS": cadence,
                "OAT_ZERO_SAVE_FROM": cadence,
                "OAT_ZERO_RESUME_STEPS": cadence,
            }
        )
    return keys


def expected(setting: Setting) -> tuple[str, ...]:
    return tuple(f"{k}={v}" for k, v in overrides(setting).items()) + (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        "OAT_ZERO_BETA=0.0",
        "OAT_ZERO_GAPO_ENABLED=0",
        "OAT_ZERO_SETPO_COEFFICIENT=0.0",
    )


def cohort(setting: Setting) -> shared.Cohort:
    return shared.Cohort(
        tag=TAG,
        arm=setting.arm,
        variant=shared.e78.VARIANTS["control"],
        title=f"E134  Qwen-0.5B  tuned-control sweep  {setting.arm}",
        ledger=LEDGER,
        protocol=PROTOCOL,
        objective_summary="DrGRPO_control_hyperparameter_sweep",
        scientific_difference=(
            f"against E128: learning rate {setting.lr}, "
            f"{setting.prompts_per_update} prompt(s) per update, "
            f"entropy coefficient {setting.entropy}; replay derivative inert"
        ),
        objective_overrides=lambda root, s=setting: overrides(s),
        snapshot_requirements=(
            ("src/oat_drgrpo/learner/grpo.py", 'infos["policy_entropy_loss"]'),
            ("ops/train.sh", '--learning_rate "${OAT_ZERO_LEARNING_RATE'),
            ("src/oat_drgrpo/learner/run.py", "eval_mode_coverage_disjoint_draws"),
        ),
        expected_exports=lambda root, s=setting: expected(s),
        ledger_extras=lambda root: {},
    )


def smoke_env(root: Path, c: shared.Cohort, setting: Setting, run, snapshot):
    """The shared smoke, with update-counted keys rescaled to the batch."""

    env, target = shared.smoke_env(root, c, run, snapshot)
    k = setting.prompts_per_update
    if k != 1:
        steps = str(int(env["OAT_ZERO_MAX_TRAIN"]) // k)
        env.update(
            {
                "OAT_ZERO_SAVE_STEPS": steps,
                "OAT_ZERO_SAVE_FROM": steps,
                "OAT_ZERO_RESUME_STEPS": steps,
            }
        )
    return env, target


def main(argv: list[str] | None = None) -> int:
    root = shared.repo_root()
    parser = argparse.ArgumentParser(description="E134 stage 1")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)

    ledger = root / LEDGER
    protocol = root / PROTOCOL
    if not protocol.is_file():
        raise SystemExit(f"protocol is absent: {protocol}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E134 submission: {ledger}")

    runs = [r for r in shared.selected_references(root)
            if int(r["seed"]) in SELECTION_SEEDS]
    snapshot, patched = e81.ensure_paired_snapshot(
        root,
        root / shared.BASE_SNAPSHOT,
        prefix=shared.SNAPSHOT_PREFIX,
        patched_files=shared.PATCHED_FILES,
    )
    by_arm = {s.arm: (s, cohort(s)) for s in SETTINGS}
    for _, c in by_arm.values():
        shared.verify_snapshot(snapshot, c)

    cells: list[dict[str, Any]] = []
    for setting, c in by_arm.values():
        for run in runs:
            env, target = shared.build_env(root, c, run, snapshot)
            if target.exists():
                raise SystemExit(f"refusing to overwrite output: {target}")
            cells.append({"setting": setting, "cohort": c, "run": run,
                          "env": env, "target": target})
    smoke_run = next(r for r in runs
                     if r["domain"] == "graph_coloring" and r["seed"] == 43)
    smokes = {}
    for batch, arm in SMOKE_FOR_BATCH.items():
        setting, c = by_arm[arm]
        env, target = smoke_env(root, c, setting, smoke_run, snapshot)
        if target.exists():
            raise SystemExit(f"refusing to overwrite smoke output: {target}")
        smokes[batch] = {"setting": setting, "cohort": c, "env": env,
                         "target": target}

    if not args.submit:
        for batch, s in smokes.items():
            print(shlex.join(shared.sbatch_command(
                root, s["cohort"], s["env"], name=shared.job_name(s["cohort"]))))
        for cell in cells:
            run = cell["run"]
            print(shlex.join(shared.sbatch_command(
                root, cell["cohort"], cell["env"],
                name=shared.job_name(cell["cohort"], run["domain"], run["seed"]),
                dependency=f"SMOKE_B{cell['setting'].prompts_per_update}")))
        print(f"[e134] smokes={len(smokes)} scientific={len(cells)} "
              f"snapshot={snapshot} patched={patched}")
        return 0

    submitted: list[str] = []
    try:
        smoke_records = {}
        for batch, s in smokes.items():
            name = shared.job_name(s["cohort"])
            job = shared.submit_held(
                shared.sbatch_command(root, s["cohort"], s["env"], name=name))
            submitted.append(job)
            # The smoke rescales its own cadence, so audit what it carries.
            cadence = ("OAT_ZERO_SAVE_STEPS", "OAT_ZERO_SAVE_FROM",
                       "OAT_ZERO_RESUME_STEPS")
            smoke_expected = tuple(
                e for e in expected(s["setting"])
                if not e.startswith(cadence)
            ) + tuple(f"{k}={s['env'][k]}" for k in cadence) + (
                "OAT_ZERO_MAX_QUERIES=32",)
            record = shared.audit_held(job, name=name, expected=smoke_expected)
            smoke_records[batch] = {"job_id": int(job), "arm": s["setting"].arm,
                                    "run_dir": str(s["target"]),
                                    "held_scheduler_record": record}
        records = []
        for cell in cells:
            setting, c, run = cell["setting"], cell["cohort"], cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            smoke_id = str(smoke_records[setting.prompts_per_update]["job_id"])
            name = shared.job_name(c, domain, seed)
            job = shared.submit_held(shared.sbatch_command(
                root, c, cell["env"], name=name, dependency=smoke_id))
            submitted.append(job)
            held = shared.audit_held(
                job, name=name,
                expected=(f"Dependency=afterok:{smoke_id}",
                          f"OAT_ZERO_SEED={seed}") + expected(setting))
            records.append({
                "domain": domain, "seed": seed, "arm": setting.arm,
                "learning_rate": setting.lr,
                "prompts_per_update": setting.prompts_per_update,
                "policy_entropy_coef": setting.entropy,
                "target_updates": e81.TRAIN_ROWS * e81.PASSES
                                  // setting.prompts_per_update,
                "run_stamp": shared.run_stamp(c, domain, seed),
                "run_dir": str(cell["target"]),
                "job_id": int(job),
                "smoke_dependency_job_id": int(smoke_id),
                "held_scheduler_record": held,
            })
        payload = {
            "schema": "e134_tuned_control_05b_stage1_jobs_v1",
            "cohort": TAG,
            "stage": 1,
            "released": False,
            "model": "Qwen2.5-0.5B-Instruct",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "domains": list(shared.DOMAINS),
            "seeds": list(SELECTION_SEEDS),
            "baseline_setting": {"cohort": "e128", "learning_rate": BASELINE_LR,
                                 "prompts_per_update": 1,
                                 "policy_entropy_coef": "0.0"},
            "settings": [s.__dict__ for s in SETTINGS],
            "fresh_rollout_budget_held": True,
            "placement": {"partition": shared.PARTITION,
                          "account": shared.ACCOUNT, "gres": shared.GRES,
                          "nodelist": shared.NODELIST},
            "smokes": smoke_records,
            "runs": records,
        }
        e81.atomic_json(ledger, payload)
        for job in submitted:
            subprocess.run(["scontrol", "release", job], check=True)
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        shared.cancel(submitted)
        raise
    print(f"[e134] released {len(smokes)} smokes and {len(records)} cells")
    print(f"[e134] ledger {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
