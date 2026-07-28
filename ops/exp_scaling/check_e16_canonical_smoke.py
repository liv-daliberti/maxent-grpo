#!/usr/bin/env python3
"""Validate E16 smoke cells and build the replayable all-six approval.

The validator checks terminal scheduler state, the frozen manifest/identity,
all 32 per-update metric records, exact controller recurrence, and the exact
finite-support endpoint audit.  ``approve`` accepts only six independently
validated cell results with one identical campaign identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

# The checker itself is part of the frozen execution snapshot.  Resolve its
# controller implementation from that snapshot's paired ``src`` tree before
# importing OAT, never from the mutable checkout or ambient PYTHONPATH.
ROOT = Path(__file__).resolve().parents[2]
FROZEN_SOURCE_ROOT = ROOT / "src"
if not FROZEN_SOURCE_ROOT.is_dir():
    raise RuntimeError(f"E16 frozen source root is missing: {FROZEN_SOURCE_ROOT}")
sys.path.insert(0, str(FROZEN_SOURCE_ROOT))

from oat_drgrpo.maxent_controllers import (  # noqa: E402
    MaxEntDualController,
    MaxEntProportionalController,
)

try:  # package import in tests
    from .verify_e16_smoke_approval import ARMS, REQUIRED_CHECKS, TASKS
    from .e16_canonical_plan import campaign_plan, plan_digest
    from .audit_e14_checkpoint import _weight_identity
    from .verify_e14_dataset import (
        EXPECTED_COMBINED_CONTENT_HASH as EXPECTED_GRAPH_CONTENT_HASH,
    )
    from .verify_e16_canonical_datasets import (
        EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH,
    )
except ImportError:  # direct script execution
    from verify_e16_smoke_approval import ARMS, REQUIRED_CHECKS, TASKS
    from e16_canonical_plan import campaign_plan, plan_digest
    from audit_e14_checkpoint import _weight_identity
    from verify_e14_dataset import (
        EXPECTED_COMBINED_CONTENT_HASH as EXPECTED_GRAPH_CONTENT_HASH,
    )
    from verify_e16_canonical_datasets import (
        EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH,
    )


TARGETS = {
    "graph_coloring": 2.7974740052946436,
    "countdown": 3.9741470618167156,
}
GEOMETRY = {
    "graph_coloring": {"leaf_count": 27, "prefix_count": 13, "max_entropy": math.log(27)},
    "countdown": {"leaf_count": 108, "prefix_count": 25, "max_entropy": math.log(108)},
}
PREFIXES = {
    "graph_coloring": "gce16_canonical_maxent_joint_smoke_v3",
    "countdown": "cde16_canonical_maxent_joint_smoke_v3",
}
SMOKE_SEED = 9006
SMOKE_UPDATES = 32
CHECKER_RELATIVE_PATH = "ops/exp_scaling/check_e16_canonical_smoke.py"
AUDITOR_RELATIVE_PATH = "ops/exp_scaling/audit_e16_canonical_endpoint.py"
APPROVAL_VERIFIER_RELATIVE_PATH = (
    "ops/exp_scaling/verify_e16_smoke_approval.py"
)
FATAL_LOG_TEXT = (
    "Traceback (most recent call last):",
    "RuntimeError:",
    "ValueError:",
    "OUT_OF_MEMORY",
    "DUE TO TIME LIMIT",
    "[watchdog] fatal:",
)


class SmokeGateError(ValueError):
    """Raised when a Stage-S cell is not a complete scientific smoke pass."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SmokeGateError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise SmokeGateError(f"{label} is not an object")
    return payload


def _finite(row: dict[str, Any], key: str, step: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise SmokeGateError(f"step {step} lacks finite {key}") from error
    if not math.isfinite(value):
        raise SmokeGateError(f"step {step} has nonfinite {key}")
    return value


def _close(observed: float, expected: float, label: str, tolerance: float = 2e-6) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise SmokeGateError(
            f"{label} mismatch: expected={expected!r} observed={observed!r}"
        )


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _approval_identity(identity: dict[str, Any]) -> dict[str, str]:
    try:
        antecedent = identity["e15_antecedent"]
        approval_identity = {
            "dataset_identity_sha256": str(identity["dataset_identity_sha256"]),
            "e15_outcome_sha256": str(antecedent["e15_outcome_sha256"]),
            "execution_surface_hash": str(identity["execution_surface_hash"]),
            "full_config_digest": str(identity["full_config_digest"]),
            "protocol_sha256": str(identity["protocol_sha256"]),
            "runtime_identity_sha256": str(identity["runtime_identity_sha256"]),
            "smoke_config_digest": str(identity["smoke_config_digest"]),
            "source_hash": str(identity["source_hash"]),
        }
    except (KeyError, TypeError) as error:
        raise SmokeGateError("campaign identity lacks approval bindings") from error
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in approval_identity.values()
    ):
        raise SmokeGateError("campaign approval binding is not a SHA-256 identity")
    return approval_identity


def _validate_identity(
    identity: dict[str, Any], *, task: str, identity_path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    if (
        identity.get("schema") != "e16_canonical_campaign_identity_v1"
        or identity.get("protocol") != "E16"
        or identity.get("stage") != "smoke"
        or identity.get("task") != task
        or identity.get("prefix") != PREFIXES[task]
        or identity.get("auto_resume") is not False
        or identity.get("watchdog_requeue") is not False
    ):
        raise SmokeGateError("campaign identity has the wrong E16 scope")
    snapshot = Path(str(identity.get("source_snapshot", ""))).resolve()
    if not snapshot.is_absolute() or not snapshot.is_dir():
        raise SmokeGateError("campaign source snapshot is missing")
    if identity.get("source_snapshot_hash") != identity.get("source_hash"):
        raise SmokeGateError("campaign source snapshot hash is inconsistent")
    runtime = identity.get("runtime_identity")
    if (
        not isinstance(runtime, dict)
        or runtime.get("controller_entropy_units") != "canonical_action_nats_exact_v1"
        or runtime.get("controller_entropy_metric")
        != "canonical_exact_sequence_entropy"
    ):
        raise SmokeGateError("campaign uses the wrong controller entropy units")
    config = identity.get("config")
    if (
        not isinstance(config, dict)
        or config.get("config_digest") != identity.get("config_digest")
        or identity.get("smoke_config_digest") != identity.get("config_digest")
        or identity.get("smoke_config_digest")
        != plan_digest(campaign_plan("smoke"))
        or identity.get("full_config_digest")
        != plan_digest(campaign_plan("full"))
    ):
        raise SmokeGateError("campaign config digest is inconsistent")
    plan = config.get("plan")
    if not isinstance(plan, dict) or plan.get("stage") != "smoke" or plan.get("seeds") != [9006]:
        raise SmokeGateError("campaign is not the 32-update seed-9006 plan")
    task_plan = plan.get("tasks", {}).get(task)
    if (
        not isinstance(task_plan, dict)
        or task_plan.get("target_optimizer_updates") != SMOKE_UPDATES
        or task_plan.get("target_entropy") != TARGETS[task]
    ):
        raise SmokeGateError("task plan drifted")
    protocol_path = Path(str(identity.get("protocol_path", ""))).resolve()
    protocol_sha256 = str(identity.get("protocol_sha256", ""))
    if (
        not protocol_path.is_file()
        or _sha256_file(protocol_path) != protocol_sha256
    ):
        raise SmokeGateError("campaign protocol path/hash binding is invalid")
    execution_root = Path(str(identity.get("execution_snapshot_root", ""))).resolve()
    execution_identity = identity.get("execution_identity")
    if (
        not execution_root.is_dir()
        or not isinstance(execution_identity, dict)
        or execution_identity.get("schema")
        != "e16_execution_surface_identity_v1"
        or execution_identity.get("sha256")
        != identity.get("execution_surface_hash")
        or not isinstance(execution_identity.get("files"), list)
    ):
        raise SmokeGateError("campaign execution-surface identity is invalid")
    expected_checker = execution_root / CHECKER_RELATIVE_PATH
    if Path(__file__).resolve() != expected_checker.resolve():
        raise SmokeGateError("smoke check was not executed from the frozen surface")
    execution_files: dict[str, str] = {}
    for record in execution_identity["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise SmokeGateError("execution-surface file record is malformed")
        relative = str(record["path"])
        if relative in execution_files:
            raise SmokeGateError("execution-surface identity repeats a file")
        execution_files[relative] = str(record["sha256"])
    required_execution_files = (
        CHECKER_RELATIVE_PATH,
        AUDITOR_RELATIVE_PATH,
        APPROVAL_VERIFIER_RELATIVE_PATH,
    )
    for relative in required_execution_files:
        path = execution_root / relative
        if not path.is_file() or execution_files.get(relative) != _sha256_file(path):
            raise SmokeGateError(f"frozen execution binding drifted for {relative}")
    identity_path = identity_path.resolve()
    identity_binding = {
        "path": str(identity_path),
        "sha256": _sha256_file(identity_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "source_snapshot": str(snapshot),
        "source_hash": str(identity["source_hash"]),
        "execution_snapshot_root": str(execution_root),
        "execution_surface_hash": str(identity["execution_surface_hash"]),
        "checker": {
            "path": str(expected_checker.resolve()),
            "sha256": execution_files[CHECKER_RELATIVE_PATH],
        },
        "auditor": {
            "path": str((execution_root / AUDITOR_RELATIVE_PATH).resolve()),
            "sha256": execution_files[AUDITOR_RELATIVE_PATH],
        },
        "approval_verifier": {
            "path": str(
                (execution_root / APPROVAL_VERIFIER_RELATIVE_PATH).resolve()
            ),
            "sha256": execution_files[APPROVAL_VERIFIER_RELATIVE_PATH],
        },
    }
    return _approval_identity(identity), identity_binding


def _validate_manifest(
    path: Path, *, task: str, arm: str, job_id: str
) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {candidate for candidate in ARMS}
    if len(rows) != 3 or {row.get("arm") for row in rows} != expected:
        raise SmokeGateError("smoke manifest must contain exactly all three arms")
    selected = [row for row in rows if row.get("arm") == arm]
    if len(selected) != 1:
        raise SmokeGateError("manifest lacks a unique requested arm")
    row = selected[0]
    if re.fullmatch(r"[1-9][0-9]*", str(job_id)) is None:
        raise SmokeGateError("smoke job ID must be a positive integer")
    expected_stamp = f"{PREFIXES[task]}_{arm}_s{SMOKE_SEED}"
    if (
        row.get("seed") != "9006"
        or row.get("job_id") != str(job_id)
        or row.get("run_stamp") != expected_stamp
    ):
        raise SmokeGateError("manifest cell identity differs from requested job")
    return expected_stamp


def _metric_rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise SmokeGateError("metrics stream contains an empty record")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SmokeGateError(f"metrics line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise SmokeGateError(f"metrics line {line_number} is not an object")
        rows.append(row)
    if len(rows) != SMOKE_UPDATES + 2:
        raise SmokeGateError(
            "restart-invalid smoke metrics require one initial row, exactly 32 "
            f"ordered update rows, and one terminal row; found {len(rows)}"
        )

    def integral(row: dict[str, Any], key: str, ordinal: int) -> int:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError) as error:
            raise SmokeGateError(
                f"metrics row {ordinal} lacks integral {key}"
            ) from error
        if not math.isfinite(value) or value != int(value):
            raise SmokeGateError(f"metrics row {ordinal} has nonintegral {key}")
        return int(value)

    trainer_steps = [
        integral(row, "trainer/step", ordinal)
        for ordinal, row in enumerate(rows)
    ]
    if trainer_steps != list(range(SMOKE_UPDATES + 2)):
        raise SmokeGateError(
            "metrics trainer/step is not the exact ordered sequence 0..33"
        )
    expected_global = [0, *range(1, SMOKE_UPDATES + 1), SMOKE_UPDATES]
    for key in ("trainer/global_step", "trainer/policy_sgd_step"):
        observed = [integral(row, key, ordinal) for ordinal, row in enumerate(rows)]
        if observed != expected_global:
            raise SmokeGateError(
                f"metrics {key} is not updates 0..32 plus one terminal alias"
            )
    update_rows = rows[1 : SMOKE_UPDATES + 1]
    learning_rounds = [
        integral(row, "train/learning_round", ordinal)
        for ordinal, row in enumerate(update_rows, 1)
    ]
    if learning_rounds != list(range(1, SMOKE_UPDATES + 1)):
        raise SmokeGateError(
            "metrics update stream is not exactly one ordered sequence 1..32"
        )
    terminal_round = integral(
        rows[-1], "train/learning_round", SMOKE_UPDATES + 1
    )
    if terminal_round != SMOKE_UPDATES:
        raise SmokeGateError("metrics terminal row is not the update-32 alias")
    if rows[0].get("train/learning_round") not in (None, 0, 0.0):
        raise SmokeGateError("metrics initial row contains prior learning progress")
    return update_rows


def _validate_run_and_metrics(
    *, run_dir: Path, run_stamp: str, metrics_path: Path
) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    metrics_path = metrics_path.resolve()
    if not run_dir.is_dir() or not run_dir.name.endswith(f"_{run_stamp}"):
        raise SmokeGateError("run directory does not match the exact run stamp")
    candidates = sorted(run_dir.glob("debug_*/train_metrics.jsonl"))
    if len(candidates) != 1:
        raise SmokeGateError(
            "restart-invalid Stage S requires exactly one debug metrics stream; "
            f"found {len(candidates)}"
        )
    expected_metrics = candidates[0].resolve()
    if metrics_path != expected_metrics:
        raise SmokeGateError("provided metrics are not the run's unique stream")
    return run_dir, expected_metrics.parent


def _validate_common_metrics(rows: list[dict[str, Any]], *, task: str) -> None:
    geometry = GEOMETRY[task]
    any_reward = False
    expected_support_max = 3 if task == "graph_coloring" else 6
    for step, row in enumerate(rows, 1):
        reward = _finite(row, "actor/rewards", step)
        if not 0 <= reward <= 1 or not math.isclose(
            reward * 16, round(reward * 16), rel_tol=0.0, abs_tol=1e-7
        ):
            raise SmokeGateError(f"step {step} reward is not a 16-rollout mean")
        any_reward = any_reward or reward > 0
        if _finite(row, "actor/canonical_invalid_count", step) != 0:
            raise SmokeGateError(f"step {step} emitted an invalid canonical action")
        if _finite(row, "actor/canonical_finish_unexpected_count", step) != 0:
            raise SmokeGateError(f"step {step} terminated unexpectedly")
        for key in ("actor/canonical_sampler_learner", "actor/canonical_sampler_fixed_shape"):
            _close(_finite(row, key, step), 1.0, f"step {step} {key}")
        _close(_finite(row, "actor/canonical_action_count", step), 3.0, f"step {step} horizon")
        _close(
            _finite(row, "actor/canonical_finish_length_count", step),
            16.0,
            f"step {step} fixed-length rollout count",
        )
        if _finite(row, "actor/canonical_behavior_q_norm_error_max", step) > 1e-5:
            raise SmokeGateError(f"step {step} actor distribution is not normalized")
        _close(
            _finite(row, "actor/canonical_behavior_q_support_min", step),
            3.0,
            f"step {step} minimum support",
        )
        _close(
            _finite(row, "actor/canonical_behavior_q_support_max", step),
            float(expected_support_max),
            f"step {step} maximum support",
        )
        if _finite(row, "train/canonical_behavior_q_norm_error_max", step) > 1e-5:
            raise SmokeGateError(f"step {step} learner distribution is not normalized")
        if _finite(row, "train/canonical_behavior_ratio_min", step) <= 0:
            raise SmokeGateError(f"step {step} behavior/current overlap vanished")
        if _finite(row, "train/canonical_behavior_sequence_ess_fraction", step) <= 0:
            raise SmokeGateError(f"step {step} sequence ESS vanished")
        if _finite(row, "train/canonical_behavior_prefix_ess_fraction_min", step) <= 0:
            raise SmokeGateError(f"step {step} prefix ESS vanished")
        exact_entropy = _finite(row, "train/canonical_exact_sequence_entropy", step)
        if not 0 <= exact_entropy <= float(geometry["max_entropy"]) + 1e-8:
            raise SmokeGateError(f"step {step} exact entropy left its support")
        _close(
            _finite(row, "train/canonical_exact_leaf_mass", step),
            1.0,
            f"step {step} exact leaf mass",
            tolerance=1e-9,
        )
        _close(
            _finite(row, "train/canonical_exact_leaf_count", step),
            float(geometry["leaf_count"]),
            f"step {step} exact leaf count",
        )
        _close(
            _finite(row, "train/canonical_exact_prefix_row_count", step),
            float(geometry["prefix_count"]),
            f"step {step} exact prefix count",
        )
        _close(
            _finite(row, "train/canonical_exact_post_update", step),
            1.0,
            f"step {step} post-update marker",
        )
        for key in (
            "train/maxent_entropy_loss",
            "train/maxent_entropy_surrogate",
            "train/maxent_prefix_ratio_max",
            "train/maxent_prefix_ratio_mean",
            "train/maxent_sequence_entropy",
        ):
            _finite(row, key, step)
    if not any_reward:
        raise SmokeGateError("smoke produced no positive rollout reward")


def _validate_controller(rows: list[dict[str, Any]], *, task: str, arm: str) -> None:
    target = TARGETS[task]
    if arm == "maxent":
        for step, row in enumerate(rows, 1):
            _close(_finite(row, "train/maxent_alpha_used", step), 0.10, f"step {step} fixed alpha")
            if any(
                key.startswith("train/maxent_control_")
                or key.startswith("train/maxent_dual_")
                for key in row
            ):
                raise SmokeGateError("fixed arm unexpectedly logged a controller")
        return
    if arm == "maxent_control":
        controller: MaxEntProportionalController | MaxEntDualController = (
            MaxEntProportionalController(
                base_alpha=0.075,
                max_alpha=0.10,
                target_ratio=1.0,
                warmup_steps=1,
                ema_decay=0.9,
                gain=4.0,
                configured_target_entropy=target,
                entropy_units="canonical_action_nats_exact_v1",
                observation_metric_key="canonical_exact_sequence_entropy",
            )
        )
        prefix = "maxent_control"
    else:
        controller = MaxEntDualController(
            base_alpha=0.075,
            min_alpha=0.05,
            max_alpha=0.10,
            target_ratio=1.0,
            warmup_steps=1,
            alpha_lr=0.005,
            configured_target_entropy=target,
            entropy_units="canonical_action_nats_exact_v1",
            observation_metric_key="canonical_exact_sequence_entropy",
        )
        prefix = "maxent_dual"
    for step, row in enumerate(rows, 1):
        _close(
            _finite(row, "train/maxent_alpha_used", step),
            float(controller.current_alpha),
            f"step {step} controller handoff",
        )
        entropy = _finite(row, "train/canonical_exact_sequence_entropy", step)
        diagnostics = controller.observe(entropy)
        _close(
            _finite(row, f"train/{prefix}_next_alpha", step),
            float(controller.current_alpha),
            f"step {step} next alpha",
        )
        _close(
            _finite(row, f"train/{prefix}_target_entropy", step),
            target,
            f"step {step} target entropy",
        )
        expected_observations = diagnostics[f"{prefix}_observations"]
        _close(
            _finite(row, f"train/{prefix}_observations", step),
            expected_observations,
            f"step {step} observation count",
        )
        if not 0.05 <= float(controller.current_alpha) <= 0.10:
            raise SmokeGateError(f"step {step} adaptive alpha left frozen bounds")


def _validate_endpoint(
    endpoint: dict[str, Any],
    *,
    task: str,
    arm: str,
    job_id: str,
    run_stamp: str,
    run_dir: Path,
    debug_dir: Path,
    campaign_binding: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    expected_cell = {
        "task": task,
        "arm": arm,
        "seed": SMOKE_SEED,
        "run_stamp": run_stamp,
        "job_id": str(job_id),
    }
    if (
        endpoint.get("schema") != "e16_exact_canonical_endpoint_audit_v1"
        or endpoint.get("status") != "pass"
        or endpoint.get("formulation")
        != "e15_derived_direct_on_policy_canonical_maxent"
        or endpoint.get("task") != task
        or endpoint.get("cell_identity") != expected_cell
        or endpoint.get("run") != {"path": str(run_dir)}
    ):
        raise SmokeGateError("exact endpoint audit has another cell/run identity")
    endpoint_campaign = endpoint.get("campaign_identity")
    if not isinstance(endpoint_campaign, dict):
        raise SmokeGateError("endpoint lacks its campaign identity binding")
    expected_endpoint_campaign = {
        key: campaign_binding[key]
        for key in (
            "path",
            "sha256",
            "protocol_path",
            "protocol_sha256",
            "execution_snapshot_root",
            "execution_surface_hash",
            "auditor",
        )
    }
    if endpoint_campaign != expected_endpoint_campaign:
        raise SmokeGateError("endpoint binds another campaign/auditor identity")
    source = endpoint.get("source")
    if (
        not isinstance(source, dict)
        or Path(str(source.get("root", ""))).resolve()
        != Path(campaign_binding["source_snapshot"]).resolve()
        or source.get("python_source_sha256") != campaign_binding["source_hash"]
    ):
        raise SmokeGateError("endpoint uses another source snapshot")
    expected_data_hash = (
        EXPECTED_GRAPH_CONTENT_HASH
        if task == "graph_coloring"
        else EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH
    )
    data = endpoint.get("data")
    expected_eval_rows = 96 if task == "graph_coloring" else 128
    if (
        not isinstance(data, dict)
        or data.get("combined_content_hash") != expected_data_hash
        or data.get("eval_rows") != expected_eval_rows
        or data.get("split") != "multi_answer"
    ):
        raise SmokeGateError("endpoint uses another frozen evaluation set")
    checkpoint = endpoint.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SmokeGateError("endpoint lacks a checkpoint identity")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    if (
        not checkpoint_path.is_dir()
        or checkpoint_path.parent != debug_dir / "saved_models"
        or checkpoint_path.name != f"step_{SMOKE_UPDATES:05d}"
        or checkpoint.get("oat_step_tag") != SMOKE_UPDATES
        or checkpoint.get("optimizer_updates") != SMOKE_UPDATES
        or checkpoint.get("role") != "scheduled_update_boundary"
    ):
        raise SmokeGateError("endpoint is not the exact run's scheduled step_00032")
    observed_weights_hash, observed_weight_files = _weight_identity(
        sorted(checkpoint_path.glob("*.safetensors"))
    )
    if (
        checkpoint.get("weights_manifest_sha256") != observed_weights_hash
        or checkpoint.get("weight_files") != observed_weight_files
    ):
        raise SmokeGateError("endpoint checkpoint weights changed after audit")
    geometry = GEOMETRY[task]
    policy = endpoint.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("horizon") != 3
        or policy.get("prefix_count") != geometry["prefix_count"]
        or policy.get("leaf_count") != geometry["leaf_count"]
        or not math.isclose(
            float(policy.get("max_action_entropy_nats", float("nan"))),
            float(geometry["max_entropy"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SmokeGateError("endpoint canonical policy geometry drifted")
    aggregate = endpoint.get("aggregate")
    tolerances = endpoint.get("tolerances")
    if not isinstance(aggregate, dict) or not isinstance(tolerances, dict):
        raise SmokeGateError("endpoint audit lacks aggregate tolerances")
    if (
        aggregate.get("prompt_count") != expected_eval_rows
        or aggregate.get("leaf_count")
        != expected_eval_rows * int(geometry["leaf_count"])
        or aggregate.get("codec_bijection_prompt_count") != expected_eval_rows
    ):
        raise SmokeGateError("endpoint did not exhaust the frozen finite support")
    tolerance_pairs = (
        ("probability_sum_max_abs_error", "probability_sum_abs"),
        ("entropy_identity_max_abs_error", "entropy_identity_abs_nats"),
        (
            "teacher_forced_vs_prefix_tree_per_token_max_abs_error",
            "teacher_forced_log_probability_abs",
        ),
        (
            "teacher_forced_vs_prefix_tree_sequence_max_abs_error",
            "teacher_forced_log_probability_abs",
        ),
    )
    for observed_key, tolerance_key in tolerance_pairs:
        observed = float(aggregate[observed_key])
        tolerance = float(tolerances[tolerance_key])
        if not math.isfinite(observed) or not math.isfinite(tolerance) or observed > tolerance:
            raise SmokeGateError(f"endpoint {observed_key} exceeds its tolerance")
    return aggregate, checkpoint_path


def validate_cell(
    *,
    task: str,
    arm: str,
    identity_path: Path,
    manifest_path: Path,
    metrics_path: Path,
    stdout_path: Path,
    endpoint_path: Path,
    run_dir: Path,
    job_id: str,
    slurm_state: str,
    slurm_exit_code: str,
) -> dict[str, Any]:
    if task not in TASKS or arm not in ARMS:
        raise SmokeGateError("unsupported E16 smoke task or arm")
    if slurm_state != "COMPLETED" or slurm_exit_code != "0:0":
        raise SmokeGateError("smoke job was not terminal-successful")
    identity = _json_object(identity_path, "campaign identity")
    approval_identity, campaign_binding = _validate_identity(
        identity, task=task, identity_path=identity_path
    )
    run_stamp = _validate_manifest(
        manifest_path, task=task, arm=arm, job_id=job_id
    )
    run_dir, debug_dir = _validate_run_and_metrics(
        run_dir=run_dir, run_stamp=run_stamp, metrics_path=metrics_path
    )
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    for fatal in FATAL_LOG_TEXT:
        if fatal in stdout:
            raise SmokeGateError(f"stdout contains fatal marker {fatal!r}")
    identity_hash = _sha256_file(identity_path)
    if (
        f"[train] protocol_identity={identity_path.resolve()} sha256={identity_hash}"
        not in stdout
    ):
        raise SmokeGateError("stdout does not bind the exact campaign identity")
    job_lines = re.findall(r"(?m)^\[slurm\] job_id=([^\s]+)\s*$", stdout)
    save_lines = re.findall(r"(?m)^\[experiment\] save_path=(.+)\s*$", stdout)
    if job_lines != [str(job_id)] or save_lines != [str(run_dir)]:
        raise SmokeGateError("stdout does not uniquely bind the job/run directory")
    if f"[train] canonical_action_task={task} " not in stdout:
        raise SmokeGateError("stdout does not bind the canonical task")
    if "[train] target_optimizer_updates=32" not in stdout:
        raise SmokeGateError("stdout does not bind the 32-update smoke budget")
    rows = _metric_rows(metrics_path)
    _validate_common_metrics(rows, task=task)
    _validate_controller(rows, task=task, arm=arm)

    endpoint = _json_object(endpoint_path, "endpoint audit")
    aggregate, checkpoint_path = _validate_endpoint(
        endpoint,
        task=task,
        arm=arm,
        job_id=job_id,
        run_stamp=run_stamp,
        run_dir=run_dir,
        debug_dir=debug_dir,
        campaign_binding=campaign_binding,
    )

    evidence_paths = {
        "endpoint_audit": endpoint_path,
        "identity": identity_path,
        "manifest": manifest_path,
        "metrics": metrics_path,
        "stdout": stdout_path,
        "checkpoint_config": checkpoint_path / "config.json",
    }
    evidence = {
        label: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for label, path in evidence_paths.items()
    }
    checks = {key: True for key in sorted(REQUIRED_CHECKS)}
    return {
        "arm": arm,
        "checks": checks,
        "endpoint_audit": evidence["endpoint_audit"],
        "evidence": evidence,
        "identity": approval_identity,
        "job_id": str(job_id),
        "protocol": "E16",
        "run_stamp": run_stamp,
        "run_dir": str(run_dir),
        "checkpoint": {
            "path": str(checkpoint_path),
            "weights_manifest_sha256": endpoint["checkpoint"][
                "weights_manifest_sha256"
            ],
        },
        "campaign_identity": campaign_binding,
        "schema": "e16_canonical_smoke_cell_result_v1",
        "seed": 9006,
        "slurm": {"exit_code": slurm_exit_code, "state": slurm_state},
        "stage": "smoke",
        "status": "pass",
        "summary": {
            "endpoint": aggregate,
            "positive_reward_updates": sum(
                _finite(row, "actor/rewards", step) > 0
                for step, row in enumerate(rows, 1)
            ),
            "updates": len(rows),
        },
        "task": task,
    }


def build_approval(result_paths: list[Path]) -> dict[str, Any]:
    if len(result_paths) != 6:
        raise SmokeGateError("approval requires exactly six cell-result paths")
    cells = []
    identities = []
    observed: set[tuple[str, str, int]] = set()
    unique_paths: dict[str, set[str]] = {
        "result": set(),
        "endpoint_audit": set(),
        "metrics": set(),
        "stdout": set(),
        "checkpoint_config": set(),
        "run_dir": set(),
        "checkpoint": set(),
    }
    unique_hashes: dict[str, set[str]] = {
        "result": set(),
        "endpoint_audit": set(),
        "stdout": set(),
    }
    job_ids: set[str] = set()
    run_stamps: set[str] = set()
    shared_evidence_tasks: dict[tuple[str, str], str] = {}
    task_shared_bindings: dict[tuple[str, str], tuple[str, str]] = {}
    for path in result_paths:
        resolved_result_path = str(path.resolve())
        if resolved_result_path in unique_paths["result"]:
            raise SmokeGateError("one cell-result path was reused")
        unique_paths["result"].add(resolved_result_path)
        result = _json_object(path, f"cell result {path}")
        if (
            result.get("schema") != "e16_canonical_smoke_cell_result_v1"
            or result.get("protocol") != "E16"
            or result.get("stage") != "smoke"
            or result.get("status") != "pass"
            or result.get("seed") != 9006
        ):
            raise SmokeGateError(f"{path} is not a passing E16 smoke result")
        task, arm = str(result.get("task")), str(result.get("arm"))
        key = (task, arm, 9006)
        if task not in TASKS or arm not in ARMS or key in observed:
            raise SmokeGateError(f"duplicate or invalid result identity {key!r}")
        observed.add(key)
        job_id = str(result.get("job_id"))
        run_stamp = str(result.get("run_stamp"))
        expected_run_stamp = f"{PREFIXES[task]}_{arm}_s{SMOKE_SEED}"
        run_dir = Path(str(result.get("run_dir", ""))).resolve()
        checkpoint_record = result.get("checkpoint")
        campaign_identity = result.get("campaign_identity")
        if (
            re.fullmatch(r"[1-9][0-9]*", job_id) is None
            or job_id in job_ids
            or run_stamp != expected_run_stamp
            or run_stamp in run_stamps
            or not run_dir.is_dir()
            or not isinstance(checkpoint_record, dict)
            or not isinstance(campaign_identity, dict)
        ):
            raise SmokeGateError(f"{path} has a reused or invalid exact cell identity")
        job_ids.add(job_id)
        run_stamps.add(run_stamp)
        checks = result.get("checks")
        if (
            not isinstance(checks, dict)
            or set(checks) != REQUIRED_CHECKS
            or any(value is not True for value in checks.values())
        ):
            raise SmokeGateError(f"{path} lost a smoke check")
        evidence = result.get("evidence")
        expected_evidence = {
            "checkpoint_config",
            "endpoint_audit",
            "identity",
            "manifest",
            "metrics",
            "stdout",
        }
        if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
            raise SmokeGateError(f"{path} lacks immutable evidence")
        for label, record in evidence.items():
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise SmokeGateError(f"{path} evidence {label} is malformed")
            evidence_path = Path(str(record["path"]))
            if not evidence_path.is_file() or _sha256_file(evidence_path) != record["sha256"]:
                raise SmokeGateError(f"{path} evidence {label} changed")
        if result.get("endpoint_audit") != evidence["endpoint_audit"]:
            raise SmokeGateError(f"{path} endpoint binding is inconsistent")
        if (
            campaign_identity.get("path") != evidence["identity"]["path"]
            or campaign_identity.get("sha256") != evidence["identity"]["sha256"]
        ):
            raise SmokeGateError(f"{path} campaign identity binding is inconsistent")
        checkpoint_path = Path(str(checkpoint_record.get("path", ""))).resolve()
        if (
            checkpoint_path / "config.json"
        ).resolve() != Path(evidence["checkpoint_config"]["path"]).resolve():
            raise SmokeGateError(f"{path} checkpoint binding is inconsistent")
        for label in ("endpoint_audit", "metrics", "stdout", "checkpoint_config"):
            evidence_path = str(Path(evidence[label]["path"]).resolve())
            if evidence_path in unique_paths[label]:
                raise SmokeGateError(f"cell-specific {label} evidence was reused")
            unique_paths[label].add(evidence_path)
        for label in ("endpoint_audit", "stdout"):
            digest = str(evidence[label]["sha256"])
            if digest in unique_hashes[label]:
                raise SmokeGateError(f"cell-specific {label} content was reused")
            unique_hashes[label].add(digest)
        for label in ("identity", "manifest"):
            evidence_path = str(Path(evidence[label]["path"]).resolve())
            sharing_key = (label, evidence_path)
            prior_task = shared_evidence_tasks.setdefault(sharing_key, task)
            if prior_task != task:
                raise SmokeGateError(f"{label} evidence was reused across tasks")
            binding = (evidence_path, str(evidence[label]["sha256"]))
            expected_binding = task_shared_bindings.setdefault(
                (task, label), binding
            )
            if binding != expected_binding:
                raise SmokeGateError(
                    f"{task} arms do not share one exact {label} binding"
                )
        for label, value in (("run_dir", str(run_dir)), ("checkpoint", str(checkpoint_path))):
            if value in unique_paths[label]:
                raise SmokeGateError(f"one exact {label} was reused across cells")
            unique_paths[label].add(value)
        slurm = result.get("slurm")
        if not isinstance(slurm, dict):
            raise SmokeGateError(f"{path} lacks terminal scheduler evidence")
        replayed = validate_cell(
            task=task,
            arm=arm,
            identity_path=Path(evidence["identity"]["path"]),
            manifest_path=Path(evidence["manifest"]["path"]),
            metrics_path=Path(evidence["metrics"]["path"]),
            stdout_path=Path(evidence["stdout"]["path"]),
            endpoint_path=Path(evidence["endpoint_audit"]["path"]),
            run_dir=Path(result["run_dir"]),
            job_id=job_id,
            slurm_state=str(slurm.get("state")),
            slurm_exit_code=str(slurm.get("exit_code")),
        )
        if replayed != result:
            raise SmokeGateError(f"{path} does not match replayed raw evidence")
        result_hash = _sha256_file(path)
        if result_hash in unique_hashes["result"]:
            raise SmokeGateError("one cell-result payload was reused")
        unique_hashes["result"].add(result_hash)
        cells.append(
            {
                "arm": arm,
                "campaign_identity": result["campaign_identity"],
                "checkpoint": result["checkpoint"],
                "endpoint_audit": result["endpoint_audit"],
                "job_id": job_id,
                "result": {"path": str(path.resolve()), "sha256": result_hash},
                "run_dir": str(run_dir),
                "run_stamp": run_stamp,
                "seed": 9006,
                "task": task,
            }
        )
        identities.append(result.get("identity"))
    expected = {(task, arm, 9006) for task in TASKS for arm in ARMS}
    if observed != expected:
        raise SmokeGateError("results do not cover the complete 2 x 3 smoke grid")
    if any(identity != identities[0] for identity in identities[1:]):
        raise SmokeGateError("smoke cells do not share one immutable identity")
    return {
        "all_six_passed": True,
        "cells": sorted(cells, key=lambda cell: (cell["task"], cell["arm"])),
        "identity": identities[0],
        "protocol": "E16",
        "schema": "e16_canonical_smoke_approval_v1",
        "seed": 9006,
        "stage": "smoke",
        "status": "approved",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cell = sub.add_parser("validate-cell")
    cell.add_argument("--task", choices=TASKS, required=True)
    cell.add_argument("--arm", choices=ARMS, required=True)
    cell.add_argument("--identity", type=Path, required=True)
    cell.add_argument("--manifest", type=Path, required=True)
    cell.add_argument("--metrics", type=Path, required=True)
    cell.add_argument("--stdout", type=Path, required=True)
    cell.add_argument("--endpoint-audit", type=Path, required=True)
    cell.add_argument("--run-dir", type=Path, required=True)
    cell.add_argument("--job-id", required=True)
    cell.add_argument("--slurm-state", required=True)
    cell.add_argument("--slurm-exit-code", required=True)
    cell.add_argument("--out", type=Path, required=True)
    approve = sub.add_parser("approve")
    approve.add_argument("--result", type=Path, action="append", required=True)
    approve.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate-cell":
            payload = validate_cell(
                task=args.task,
                arm=args.arm,
                identity_path=args.identity,
                manifest_path=args.manifest,
                metrics_path=args.metrics,
                stdout_path=args.stdout,
                endpoint_path=args.endpoint_audit,
                run_dir=args.run_dir,
                job_id=args.job_id,
                slurm_state=args.slurm_state,
                slurm_exit_code=args.slurm_exit_code,
            )
        else:
            payload = build_approval(args.result)
    except (FileNotFoundError, KeyError, SmokeGateError) as error:
        raise SystemExit(f"E16 smoke gate rejected: {error}") from error
    _write(args.out, payload)
    print(f"E16 {args.command} passed: {args.out.resolve()}")
    print(f"sha256={_sha256_file(args.out)}")


if __name__ == "__main__":
    main()
