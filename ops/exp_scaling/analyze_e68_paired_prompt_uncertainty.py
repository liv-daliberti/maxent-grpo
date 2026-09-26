#!/usr/bin/env python3
"""Descriptive crossed paired-bootstrap intervals for E68 versus E66."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METHOD = (
    ROOT
    / "paper/preregistration/"
    "e68_paired_prompt_uncertainty_secondary_20260727.md"
)
IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e68_paired_prompt_uncertainty_secondary_identity.json"
)
OUT = (
    ROOT
    / "var/artifacts/"
    "e68_paired_prompt_uncertainty_secondary_latest.json"
)
E66_IDENTITY = (
    ROOT
    / "var/artifacts/e66_same_plumbing_actuator_ablation_identity.json"
)
E68_IDENTITY = (
    ROOT
    / "var/artifacts/e68_separated_support_actuator_ablation_identity.json"
)
PASSES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
SEEDS = (43, 44, 45)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_NAMESPACE = "20260727"
DOMAINS = {
    "graph_coloring": ("Graph coloring", 192),
    "countdown": ("Countdown", 384),
    "python_factor": ("Python factors", 384),
    "mathir": ("MathIR action menu", 384),
}
METRICS = {
    "greedy": ("greedy", "mean_at_k"),
    "mean8": ("sampled", "mean_at_k"),
    "pass8": ("sampled", "any_correct_at_k"),
    "distinct8": ("sampled", "distinct_correct_modes_at_k"),
}


class IncompleteCheckpoint(Exception):
    """The live checkpoint has not yet written its complete trace surface."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_analysis_identity() -> dict[str, Any]:
    identity = _load_json(IDENTITY)
    violations = []
    expected = {
        "method_sha256": _sha256(METHOD),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    for field, actual in expected.items():
        if identity.get(field) != actual:
            violations.append(
                f"{field} mismatch: identity={identity.get(field)!r}, "
                f"actual={actual!r}"
            )
    if violations:
        raise RuntimeError("; ".join(violations))
    return identity


def _resolve_run(job: dict[str, Any]) -> Path | None:
    pattern = (
        f"var/data/*_{job['run_stamp']}/debug_job{int(job['job_id'])}"
    )
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one run directory for job {job['job_id']}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_traces(path: Path) -> dict[int, list[dict[str, Any]]]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    if not path.is_file():
        return by_step
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: malformed JSON: {exc}"
                ) from exc
            if not isinstance(record, dict) or not isinstance(
                record.get("step"), int
            ):
                raise RuntimeError(
                    f"{path}:{line_number}: missing integer step"
                )
            by_step.setdefault(int(record["step"]), []).append(record)
    return by_step


def _trace_surface(
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    greedy = [
        row
        for row in records
        if row.get("evaluation_kind")
        == "deterministic_greedy_trace_neutral"
    ]
    sampled = {
        row.get("draw_index"): row
        for row in records
        if row.get("evaluation_kind") == "fixed_seed_sampled_k_neutral"
    }
    if len(greedy) != 1 or set(sampled) != {0, 1, 2, 3}:
        raise IncompleteCheckpoint(label)
    if len(
        [
            row
            for row in records
            if row.get("evaluation_kind")
            == "fixed_seed_sampled_k_neutral"
        ]
    ) != 4:
        raise RuntimeError(f"{label}: duplicate sampled draw index")
    if greedy[0].get("sample_count") != 1:
        raise RuntimeError(f"{label}: greedy sample_count is not 1")
    if any(row.get("sample_count") != 8 for row in sampled.values()):
        raise RuntimeError(f"{label}: sampled sample_count is not 8")
    return {"greedy": greedy[0], "sampled": sampled}


def _prompt_identity(prompt: dict[str, Any]) -> tuple[Any, ...]:
    return (
        prompt.get("prompt_index"),
        prompt.get("prompt"),
        prompt.get("reference"),
    )


def _finite_float(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"{label}: expected finite numeric metric")
    return float(value)


def _paired_prompt_vectors(
    control_records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, np.ndarray]:
    surfaces = {
        "control": _trace_surface(control_records, label=f"{label}:E66"),
        "repair": _trace_surface(repair_records, label=f"{label}:E68"),
    }
    for kind in ("greedy", "sampled"):
        if kind == "greedy":
            pairs = [(None, surfaces["control"][kind], surfaces["repair"][kind])]
        else:
            pairs = [
                (
                    draw,
                    surfaces["control"][kind][draw],
                    surfaces["repair"][kind][draw],
                )
                for draw in range(4)
            ]
        for draw, control, repair in pairs:
            if control.get("seed") != repair.get("seed"):
                raise RuntimeError(
                    f"{label}:{kind}:{draw}: evaluation seed mismatch"
                )
            if control.get("draw_index") != repair.get("draw_index"):
                raise RuntimeError(
                    f"{label}:{kind}:{draw}: draw index mismatch"
                )
            control_prompts = control.get("prompts")
            repair_prompts = repair.get("prompts")
            if not isinstance(control_prompts, list) or not isinstance(
                repair_prompts, list
            ):
                raise RuntimeError(f"{label}:{kind}:{draw}: prompts missing")
            if len(control_prompts) != len(repair_prompts):
                raise RuntimeError(
                    f"{label}:{kind}:{draw}: prompt count mismatch"
                )
            for index, (left, right) in enumerate(
                zip(control_prompts, repair_prompts, strict=True)
            ):
                if _prompt_identity(left) != _prompt_identity(right):
                    raise RuntimeError(
                        f"{label}:{kind}:{draw}: prompt identity mismatch "
                        f"at index {index}"
                    )

    canonical = surfaces["control"]["greedy"]["prompts"]
    prompt_count = len(canonical)
    if prompt_count == 0:
        raise RuntimeError(f"{label}: empty prompt surface")
    for arm, surface in surfaces.items():
        all_traces = [surface["greedy"]] + [
            surface["sampled"][draw] for draw in range(4)
        ]
        for trace_name, trace in enumerate(all_traces):
            identities = [
                _prompt_identity(prompt) for prompt in trace["prompts"]
            ]
            if identities != [
                _prompt_identity(prompt) for prompt in canonical
            ]:
                raise RuntimeError(
                    f"{label}:{arm}: prompt order differs across traces "
                    f"at trace {trace_name}"
                )

    vectors: dict[str, np.ndarray] = {}
    for metric, (surface_kind, raw_metric) in METRICS.items():
        arm_vectors = {}
        for arm, surface in surfaces.items():
            if surface_kind == "greedy":
                traces = [surface["greedy"]]
            else:
                traces = [surface["sampled"][draw] for draw in range(4)]
            values = []
            for prompt_index in range(prompt_count):
                draw_values = [
                    _finite_float(
                        trace["prompts"][prompt_index]
                        .get("metrics", {})
                        .get(raw_metric),
                        label=(
                            f"{label}:{arm}:{metric}:prompt{prompt_index}"
                        ),
                    )
                    for trace in traces
                ]
                values.append(float(np.mean(draw_values)))
            arm_vectors[arm] = np.asarray(values, dtype=np.float64)
        vectors[metric] = arm_vectors["repair"] - arm_vectors["control"]
    return vectors


def _bootstrap_seed(domain: str, training_pass: int, metric: str) -> int:
    key = (
        f"{BOOTSTRAP_NAMESPACE}|{domain}|{training_pass}|{metric}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _crossed_bootstrap(
    difference: np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    if difference.ndim != 2 or difference.shape[0] != len(SEEDS):
        raise ValueError("difference must be a 3-by-prompt matrix")
    prompt_count = difference.shape[1]
    generator = np.random.default_rng(seed)
    seed_weights = generator.multinomial(
        len(SEEDS),
        [1 / len(SEEDS)] * len(SEEDS),
        size=replicates,
    )
    prompt_weights = generator.multinomial(
        prompt_count,
        [1 / prompt_count] * prompt_count,
        size=replicates,
    )
    estimates = np.einsum(
        "bs,sp,bp->b",
        seed_weights / len(SEEDS),
        difference,
        prompt_weights / prompt_count,
        optimize=True,
    )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def _analyze_checkpoint(
    paired_traces: dict[int, dict[str, dict[int, list[dict[str, Any]]]]],
    *,
    domain: str,
    training_pass: int,
    step: int,
) -> dict[str, Any]:
    matrices: dict[str, list[np.ndarray]] = {
        metric: [] for metric in METRICS
    }
    for seed in SEEDS:
        control_records = paired_traces[seed]["control"].get(step, [])
        repair_records = paired_traces[seed]["repair"].get(step, [])
        if not control_records or not repair_records:
            raise IncompleteCheckpoint(f"{domain}:pass{training_pass}:s{seed}")
        vectors = _paired_prompt_vectors(
            control_records,
            repair_records,
            label=f"{domain}:pass{training_pass}:s{seed}",
        )
        for metric, vector in vectors.items():
            matrices[metric].append(vector)

    prompt_counts = {
        matrix.shape[0]
        for vectors in matrices.values()
        for matrix in vectors
    }
    if len(prompt_counts) != 1:
        raise RuntimeError(
            f"{domain}:pass{training_pass}: prompt count differs by seed"
        )
    prompt_count = prompt_counts.pop()
    result = {
        "training_pass": training_pass,
        "optimizer_step": step,
        "paired_training_seeds": len(SEEDS),
        "prompts_per_seed": prompt_count,
        "sampled_draws_averaged_per_prompt": 4,
        "metrics": {},
    }
    for metric, vectors in matrices.items():
        difference = np.stack(vectors)
        seed_deltas = difference.mean(axis=1)
        point_estimate = float(seed_deltas.mean())
        lower, upper = _crossed_bootstrap(
            difference,
            seed=_bootstrap_seed(domain, training_pass, metric),
        )
        result["metrics"][metric] = {
            "e68_minus_e66": point_estimate,
            "paired_seed_deltas": {
                str(seed): float(delta)
                for seed, delta in zip(SEEDS, seed_deltas, strict=True)
            },
            "descriptive_crossed_bootstrap_95": [lower, upper],
        }
    return result


def main() -> None:
    frozen_identity = _verify_analysis_identity()
    identities = {
        "control": _load_json(E66_IDENTITY),
        "repair": _load_json(E68_IDENTITY),
    }
    violations = []
    domains: dict[str, Any] = {}
    for domain, (label, steps_per_pass) in DOMAINS.items():
        paired_traces = {}
        materialized = True
        try:
            for seed in SEEDS:
                paired_traces[seed] = {}
                for arm in ("control", "repair"):
                    jobs = {
                        int(job["seed"]): job
                        for job in identities[arm]["jobs"][domain]
                    }
                    run = _resolve_run(jobs[seed])
                    if run is None:
                        materialized = False
                        paired_traces[seed][arm] = {}
                    else:
                        paired_traces[seed][arm] = _load_traces(
                            run / "eval_mode_coverage_draws.jsonl"
                        )
        except RuntimeError as exc:
            violations.append(str(exc))
            materialized = False
        checkpoints = []
        if materialized:
            for training_pass in PASSES:
                try:
                    checkpoints.append(
                        _analyze_checkpoint(
                            paired_traces,
                            domain=domain,
                            training_pass=training_pass,
                            step=training_pass * steps_per_pass,
                        )
                    )
                except IncompleteCheckpoint:
                    continue
                except RuntimeError as exc:
                    violations.append(str(exc))
        domains[domain] = {
            "label": label,
            "registered_checkpoints": len(PASSES),
            "complete_paired_checkpoints": len(checkpoints),
            "checkpoints": checkpoints,
            "latest_complete": checkpoints[-1] if checkpoints else None,
        }

    complete = all(
        row["complete_paired_checkpoints"] == len(PASSES)
        for row in domains.values()
    )
    payload = {
        "schema": "e68_paired_prompt_uncertainty_secondary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "fail" if violations else "complete" if complete else "in_progress"
        ),
        "evidential_role": (
            "post-specified descriptive analysis; never a primary gate"
        ),
        "frozen_identity": frozen_identity,
        "method": {
            "paired_arm_difference": "E68 minus E66",
            "training_seeds": list(SEEDS),
            "registered_passes": list(PASSES),
            "sampled_k": 8,
            "sampled_draws_averaged_before_resampling": 4,
            "bootstrap": "crossed paired seed-by-prompt percentile",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "interval_percentiles": [2.5, 97.5],
            "bootstrap_namespace": BOOTSTRAP_NAMESPACE,
            "p_values": "not computed",
        },
        "summary": {
            "complete_domain_checkpoints": sum(
                row["complete_paired_checkpoints"] for row in domains.values()
            ),
            "expected_domain_checkpoints": len(DOMAINS) * len(PASSES),
            "violation_count": len(violations),
        },
        "domains": domains,
        "violations": violations,
    }
    _atomic_json(OUT, payload)
    print(
        "[e68-paired-prompt] "
        f"status={payload['status']} "
        f"checkpoints={payload['summary']['complete_domain_checkpoints']}/"
        f"{payload['summary']['expected_domain_checkpoints']} "
        f"violations={len(violations)}"
    )
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
