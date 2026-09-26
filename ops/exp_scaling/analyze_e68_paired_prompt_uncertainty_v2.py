#!/usr/bin/env python3
"""Source-aligned descriptive paired uncertainty for E68 versus E66."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = (
    ROOT / "ops/exp_scaling/analyze_e68_paired_prompt_uncertainty.py"
)
METHOD = (
    ROOT
    / "paper/preregistration/"
    "e68_paired_prompt_uncertainty_secondary_v2_20260727.md"
)
IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e68_paired_prompt_uncertainty_secondary_v2_identity.json"
)
OUT = (
    ROOT
    / "var/artifacts/"
    "e68_paired_prompt_uncertainty_secondary_v2_latest.json"
)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "e68_paired_prompt_uncertainty_v1",
        BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v1 analysis routines")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


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


def _verify_identity() -> dict[str, Any]:
    identity = _load_json(IDENTITY)
    actual = {
        "method_sha256": _sha256(METHOD),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "base_script_sha256": _sha256(BASE_SCRIPT),
    }
    mismatches = [
        f"{field} mismatch: identity={identity.get(field)!r}, "
        f"actual={value!r}"
        for field, value in actual.items()
        if identity.get(field) != value
    ]
    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    return identity


def _primary_greedy(
    path: Path,
    *,
    label: str,
) -> tuple[list[tuple[Any, ...]], np.ndarray]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"{label}: primary greedy payload is not a list")
    identities = []
    scores = []
    for expected_index, prompt in enumerate(payload):
        if not isinstance(prompt, dict):
            raise RuntimeError(f"{label}: invalid prompt {expected_index}")
        output_scores = prompt.get("scores")
        if not isinstance(output_scores, list) or len(output_scores) != 1:
            raise RuntimeError(
                f"{label}: expected one primary score at prompt "
                f"{expected_index}"
            )
        score = output_scores[0]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise RuntimeError(
                f"{label}: non-finite primary score at prompt "
                f"{expected_index}"
            )
        identities.append(
            (
                expected_index,
                prompt.get("problem"),
                prompt.get("reference"),
            )
        )
        scores.append(float(score))
    return identities, np.asarray(scores, dtype=np.float64)


def _sidecar_greedy(
    records: list[dict[str, Any]],
    *,
    label: str,
) -> tuple[list[tuple[Any, ...]], np.ndarray]:
    surface = base._trace_surface(records, label=label)
    prompts = surface["greedy"]["prompts"]
    identities = [base._prompt_identity(prompt) for prompt in prompts]
    values = np.asarray(
        [
            base._finite_float(
                prompt.get("metrics", {}).get("mean_at_k"),
                label=f"{label}:prompt{index}",
            )
            for index, prompt in enumerate(prompts)
        ],
        dtype=np.float64,
    )
    return identities, values


def _paired_prompt_vectors_v2(
    control_records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
    *,
    control_primary: Path,
    repair_primary: Path,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    vectors = base._paired_prompt_vectors(
        control_records,
        repair_records,
        label=label,
    )
    primary = {}
    repeated = {}
    for arm, path, records in (
        ("control", control_primary, control_records),
        ("repair", repair_primary, repair_records),
    ):
        primary_ids, primary_values = _primary_greedy(
            path,
            label=f"{label}:{arm}:primary",
        )
        repeated_ids, repeated_values = _sidecar_greedy(
            records,
            label=f"{label}:{arm}:repeated",
        )
        if primary_ids != repeated_ids:
            raise RuntimeError(
                f"{label}:{arm}: primary/repeated greedy prompt mismatch"
            )
        primary[arm] = primary_values
        repeated[arm] = repeated_values
    if primary["control"].shape != primary["repair"].shape:
        raise RuntimeError(f"{label}: primary greedy prompt count mismatch")
    vectors["greedy"] = primary["repair"] - primary["control"]
    sensitivity = {
        arm: {
            "different_prompt_scores": int(
                np.count_nonzero(primary[arm] != repeated[arm])
            ),
            "primary_mean": float(primary[arm].mean()),
            "repeated_mean": float(repeated[arm].mean()),
            "repeated_minus_primary": float(
                repeated[arm].mean() - primary[arm].mean()
            ),
        }
        for arm in ("control", "repair")
    }
    return vectors, sensitivity


def _analyze_checkpoint(
    paired: dict[int, dict[str, dict[str, Any]]],
    *,
    domain: str,
    training_pass: int,
    step: int,
) -> dict[str, Any]:
    matrices: dict[str, list[np.ndarray]] = {
        metric: [] for metric in base.METRICS
    }
    repeatability = {}
    for seed in base.SEEDS:
        control_records = paired[seed]["control"]["traces"].get(step, [])
        repair_records = paired[seed]["repair"]["traces"].get(step, [])
        if not control_records or not repair_records:
            raise base.IncompleteCheckpoint(
                f"{domain}:pass{training_pass}:s{seed}"
            )
        primary_name = f"{step}_multi_answer.json"
        control_primary = (
            paired[seed]["control"]["run"] / "eval_results" / primary_name
        )
        repair_primary = (
            paired[seed]["repair"]["run"] / "eval_results" / primary_name
        )
        if not control_primary.is_file() or not repair_primary.is_file():
            raise base.IncompleteCheckpoint(
                f"{domain}:pass{training_pass}:s{seed}:primary"
            )
        vectors, sensitivity = _paired_prompt_vectors_v2(
            control_records,
            repair_records,
            control_primary=control_primary,
            repair_primary=repair_primary,
            label=f"{domain}:pass{training_pass}:s{seed}",
        )
        repeatability[str(seed)] = sensitivity
        for metric, vector in vectors.items():
            matrices[metric].append(vector)

    prompt_counts = {
        vector.shape[0]
        for vectors in matrices.values()
        for vector in vectors
    }
    if len(prompt_counts) != 1:
        raise RuntimeError(
            f"{domain}:pass{training_pass}: prompt count differs by seed"
        )
    prompt_count = prompt_counts.pop()
    result = {
        "training_pass": training_pass,
        "optimizer_step": step,
        "paired_training_seeds": len(base.SEEDS),
        "prompts_per_seed": prompt_count,
        "sampled_draws_averaged_per_prompt": 4,
        "primary_vs_repeated_greedy_sensitivity": repeatability,
        "metrics": {},
    }
    for metric, vectors in matrices.items():
        difference = np.stack(vectors)
        seed_deltas = difference.mean(axis=1)
        lower, upper = base._crossed_bootstrap(
            difference,
            seed=base._bootstrap_seed(domain, training_pass, metric),
        )
        result["metrics"][metric] = {
            "e68_minus_e66": float(seed_deltas.mean()),
            "paired_seed_deltas": {
                str(seed): float(delta)
                for seed, delta in zip(
                    base.SEEDS,
                    seed_deltas,
                    strict=True,
                )
            },
            "descriptive_crossed_bootstrap_95": [lower, upper],
        }
    return result


def main() -> None:
    frozen_identity = _verify_identity()
    identities = {
        "control": _load_json(base.E66_IDENTITY),
        "repair": _load_json(base.E68_IDENTITY),
    }
    violations = []
    domains: dict[str, Any] = {}
    for domain, (label, steps_per_pass) in base.DOMAINS.items():
        paired = {}
        materialized = True
        try:
            for seed in base.SEEDS:
                paired[seed] = {}
                for arm in ("control", "repair"):
                    jobs = {
                        int(job["seed"]): job
                        for job in identities[arm]["jobs"][domain]
                    }
                    run = base._resolve_run(jobs[seed])
                    if run is None:
                        materialized = False
                        paired[seed][arm] = {"run": None, "traces": {}}
                    else:
                        paired[seed][arm] = {
                            "run": run,
                            "traces": base._load_traces(
                                run / "eval_mode_coverage_draws.jsonl"
                            ),
                        }
        except RuntimeError as exc:
            violations.append(str(exc))
            materialized = False
        checkpoints = []
        if materialized:
            for training_pass in base.PASSES:
                try:
                    checkpoints.append(
                        _analyze_checkpoint(
                            paired,
                            domain=domain,
                            training_pass=training_pass,
                            step=training_pass * steps_per_pass,
                        )
                    )
                except base.IncompleteCheckpoint:
                    continue
                except RuntimeError as exc:
                    violations.append(str(exc))
        domains[domain] = {
            "label": label,
            "registered_checkpoints": len(base.PASSES),
            "complete_paired_checkpoints": len(checkpoints),
            "checkpoints": checkpoints,
            "latest_complete": checkpoints[-1] if checkpoints else None,
        }

    complete = all(
        row["complete_paired_checkpoints"] == len(base.PASSES)
        for row in domains.values()
    )
    payload = {
        "schema": "e68_paired_prompt_uncertainty_secondary_v2",
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
            "greedy_source": "primary eval_results score",
            "sampled_source": "four fixed sampled-K sidecar draws",
            "training_seeds": list(base.SEEDS),
            "registered_passes": list(base.PASSES),
            "sampled_k": 8,
            "sampled_draws_averaged_before_resampling": 4,
            "bootstrap": "crossed paired seed-by-prompt percentile",
            "bootstrap_replicates": base.BOOTSTRAP_REPLICATES,
            "interval_percentiles": [2.5, 97.5],
            "bootstrap_namespace": base.BOOTSTRAP_NAMESPACE,
            "p_values": "not computed",
        },
        "summary": {
            "complete_domain_checkpoints": sum(
                row["complete_paired_checkpoints"]
                for row in domains.values()
            ),
            "expected_domain_checkpoints": (
                len(base.DOMAINS) * len(base.PASSES)
            ),
            "violation_count": len(violations),
        },
        "domains": domains,
        "violations": violations,
    }
    _atomic_json(OUT, payload)
    print(
        "[e68-paired-prompt-v2] "
        f"status={payload['status']} "
        f"checkpoints={payload['summary']['complete_domain_checkpoints']}/"
        f"{payload['summary']['expected_domain_checkpoints']} "
        f"violations={len(violations)}"
    )
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
