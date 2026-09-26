#!/usr/bin/env python3
"""Audit and summarize E121 post-freeze identity-level replay histories."""

from __future__ import annotations

import argparse
import collections
import hashlib
import statistics
import json
import math
from pathlib import Path


FREEZE_STEP = 384
TARGET_STEPS = 3072


def suffix_values(record: dict[str, object], suffix: str) -> list[float]:
    values: list[float] = []
    for key, value in record.items():
        if key.endswith(suffix):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"non-finite {key}={value!r}")
            values.append(number)
    return values


def indexed(record: dict[str, object], marker: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in record.items():
        if marker not in key:
            continue
        index = key.rsplit(marker, 1)[1]
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite {key}={value!r}")
        result[index] = number
    return result


def resolve_metrics(run_dir: Path) -> Path:
    stable = run_dir.resolve()
    direct = stable / "train_metrics.jsonl"
    if direct.is_file():
        return direct
    receipt = stable / "TRAINING_COMPLETE.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    attempt = Path(str(payload["terminal_attempt"])).resolve()
    attempt.relative_to(stable)
    metrics = attempt / "train_metrics.jsonl"
    if not metrics.is_file():
        raise FileNotFoundError(metrics)
    return metrics


def check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def get(record: dict[str, object], name: str) -> float:
    return float(record["train/" + name])


def audit_records(
    records: list[dict[str, object]],
    *,
    freeze_step: int = FREEZE_STEP,
    target_steps: int = TARGET_STEPS,
) -> tuple[dict[str, object], dict[tuple[int, int], list[tuple[int, float, float]]]]:
    """Validate the entire scheduled population and return scored histories.

    Full-bank tracked counts independently determine the population. Complete
    round-robin cycles and explicit per-prompt row membership detect dropped
    prompts or keys. A final carried-forward log record is checked exactly and
    excluded, so it cannot increase an identity's visit count.
    """
    raw = records
    updates = [r for r in raw if 1 <= r["trainer/step"] <= target_steps]
    check(
        [r["trainer/step"] for r in updates] == list(range(1, target_steps + 1)),
        "missing/duplicate updates",
    )
    check(
        all(
            r["trainer/step"]
            == r["trainer/global_step"]
            == r["trainer/policy_sgd_step"]
            == r["misc/global_step"]
            for r in updates
        ),
        "optimizer-step alignment",
    )
    extra = [r for r in raw if r["trainer/step"] > target_steps]
    check(
        len(extra) == 1
        and extra[0]["trainer/step"] == target_steps + 1
        and extra[0]["trainer/global_step"] == target_steps,
        "unexpected terminal logging convention",
    )
    check(
        {k: v for k, v in extra[0].items() if k.startswith("train/")}
        == {k: v for k, v in updates[-1].items() if k.startswith("train/")},
        "terminal replay not a carried-forward duplicate",
    )
    check(
        all(
            get(r, "canonical_replay_bank_freeze_step") == freeze_step for r in updates
        ),
        "freeze config",
    )
    check(
        all(
            get(r, "canonical_replay_bank_membership_frozen")
            == float(r["trainer/step"] >= freeze_step)
            for r in updates
        ),
        "freeze flag boundary",
    )
    post = [r for r in updates if r["trainer/step"] >= freeze_step]
    prompt_sizes = {int(get(r, "online_canonical_tracked_prompts")) for r in post}
    outcome_sizes = {int(get(r, "online_canonical_tracked_outcomes")) for r in post}
    check(len(prompt_sizes) == len(outcome_sizes) == 1, "full-bank counts changed")
    n_prompts = prompt_sizes.pop()
    n_outcomes = outcome_sizes.pop()
    check(
        int(get(updates[freeze_step - 2], "online_canonical_tracked_prompts"))
        == n_prompts,
        "freeze admitted prompt",
    )
    check(
        int(get(updates[freeze_step - 2], "online_canonical_tracked_outcomes"))
        == n_outcomes,
        "freeze admitted outcome",
    )
    members = collections.defaultdict(set)
    outcomes = collections.defaultdict(set)
    counts = collections.defaultdict(set)
    histories = collections.defaultdict(list)
    weights = collections.defaultdict(set)
    prompt_steps = collections.defaultdict(list)
    observed_sets = collections.defaultdict(set)
    sequence = []
    lengths = collections.defaultdict(set)
    totalrows = 0
    max_length_ratio_error = 0.0
    max_token_total_error = 0.0
    group_stems = ["prompt_fingerprint_group_", "membership_fingerprint_group_"]
    row_stems = [
        "prompt_fingerprint_row_",
        "outcome_fingerprint_row_",
        "exemplar_mean_logprob_row_",
        "exemplar_sequence_logprob_row_",
        "fresh_count_row_",
        "target_weight_row_",
    ]
    diagnostics = [
        "alpha_used",
        "objective_scale",
        "backward_scale",
        "applied_positive_gradient_max",
        "applied_score_gradient_sum",
        "applied_score_gradient_l2",
        "realized_response_tokens",
    ]
    for r in post:
        step = int(r["trainer/step"])
        check(
            all(
                not isinstance(v, (float, int)) or math.isfinite(v)
                for k, v in r.items()
                if k.startswith("train/canonical_replay_")
            ),
            "nonfinite telemetry",
        )
        check(
            get(r, "online_canonical_bank_size_before_mean")
            == get(r, "online_canonical_bank_size_after_mean"),
            "postfreeze local bank changed",
        )
        for name in [
            "global_groups_per_step",
            "global_scheduler_active",
            "schedule_used_global",
            "verified_likelihood_active",
            "frequency_count_fresh_only",
        ]:
            check(get(r, "canonical_replay_" + name) == 1, name)
        for name in [
            "key_weighting_frequency",
            "compute_only_configured",
            "global_bootstrap_steps",
            "frequency_count_from_replay",
            "frequency_count_from_proposals",
        ]:
            check(get(r, "canonical_replay_" + name) == 0, name)
        vals = {
            stem: indexed(r, "canonical_replay_" + stem)
            for stem in row_stems + group_stems
        }
        inds = set(vals[row_stems[0]])
        check(inds and all(set(vals[s]) == inds for s in row_stems), "row alignment")
        check(
            inds == {f"{i:02d}" for i in range(len(inds))}, "noncontiguous row indices"
        )
        check(
            all(set(vals[s]) == {"00"} for s in group_stems),
            "one aligned group per step",
        )
        p = int(vals["prompt_fingerprint_group_"]["00"])
        sequence.append(p)
        prompt_steps[p].append(step)
        members[p].add(int(vals["membership_fingerprint_group_"]["00"]))
        row_outcomes = []
        token_total = 0
        for i in sorted(inds):
            check(
                int(vals["prompt_fingerprint_row_"][i]) == p,
                "row/group prompt mismatch",
            )
            o = int(vals["outcome_fingerprint_row_"][i])
            row_outcomes.append(o)
            ident = (p, o)
            m = vals["exemplar_mean_logprob_row_"][i]
            s = vals["exemplar_sequence_logprob_row_"][i]
            check(m < 0 and s < 0, "invalid log probability")
            ratio = s / m
            length = round(ratio)
            lengths[ident].add(length)
            max_length_ratio_error = max(max_length_ratio_error, abs(ratio - length))
            check(
                length > 0 and abs(ratio - length) < 1e-4,
                "nonintegral implied token count",
            )
            token_total += length
            histories[ident].append((step, m, s))
            outcomes[p].add(o)
            counts[ident].add(vals["fresh_count_row_"][i])
            weights[ident].add(vals["target_weight_row_"][i])
        check(len(set(row_outcomes)) == len(inds), "duplicate identity in group")
        observed_sets[p].add(tuple(row_outcomes))
        for name in ["available_modes", "actuator_modes", "banked_modes"]:
            check(
                get(r, "canonical_replay_" + name) == len(inds), "omitted row: " + name
            )
        for name in diagnostics:
            check(
                math.isfinite(get(r, "canonical_replay_" + name)),
                "missing/nonfinite diagnostic " + name,
            )
        max_token_total_error = max(
            max_token_total_error,
            abs(token_total - get(r, "canonical_replay_realized_response_tokens")),
        )
        check(
            token_total == get(r, "canonical_replay_realized_response_tokens"),
            "sequence/mean tokens disagree with replay total",
        )
        check(
            math.isclose(get(r, "canonical_replay_alpha_used"), 0.1, rel_tol=1e-6),
            "coefficient",
        )
        check(get(r, "canonical_replay_objective_scale") == 1 / 16, "objective scale")
        check(get(r, "canonical_replay_backward_scale") == 16, "backward scale")
        check(
            get(r, "canonical_replay_applied_positive_gradient_max") == 0,
            "positive applied replay gradient",
        )
        totalrows += len(inds)
    check(len(prompt_steps) == n_prompts, "missing or colliding prompt identities")
    check(len(histories) == n_outcomes, "missing or colliding outcome identities")
    check(
        len(set(sequence[:n_prompts])) == n_prompts,
        "first round-robin cycle incomplete",
    )
    check(
        all(p == sequence[i % n_prompts] for i, p in enumerate(sequence)),
        "round-robin cadence mismatch",
    )
    check(all(len(v) == 1 for v in members.values()), "membership fingerprint changed")
    check(
        all(len(v) == 1 for v in observed_sets.values()),
        "explicit group membership changed",
    )
    check(all(len(v) == 1 for v in counts.values()), "fresh count changed")
    check(all(v == {1.0} for v in weights.values()), "uniform target weight changed")
    check(all(len(v) == 1 for v in lengths.values()), "exemplar token length changed")
    check(
        all(
            [row[0] for row in rows] == prompt_steps[ident[0]]
            for ident, rows in histories.items()
        ),
        "missing scheduled identity observation",
    )
    check(all(len(rows) >= 2 for rows in histories.values()), "fewer than two visits")
    check(sum(len(rows) for rows in histories.values()) == totalrows, "row accounting")
    return {
        "metrics_records": len(raw),
        "optimizer_updates": len(updates),
        "terminal_carried_forward_records_excluded": len(extra),
        "postfreeze_first_step": freeze_step,
        "postfreeze_final_step": target_steps,
        "postfreeze_records": len(post),
        "prompt_count": n_prompts,
        "identity_count": n_outcomes,
        "identity_observation_count": totalrows,
        "identities_with_at_least_two_visits": len(histories),
        "insufficient_visit_count": 0,
        "missing_identity_count": 0,
        "finite_every_observation_fraction": 1.0,
        "min_visits": min(map(len, histories.values())),
        "max_visits": max(map(len, histories.values())),
        "changing_membership_count": 0,
        "changing_fresh_count_identity_count": 0,
        "complete_round_robin_schedule": True,
        "maximum_inferred_integer_token_count_error": max_length_ratio_error,
        "maximum_group_token_total_error": max_token_total_error,
        "bank_support_max": max(map(len, outcomes.values())),
    }, dict(histories)


def audit_run(run: dict[str, object]) -> dict[str, object]:
    metrics = resolve_metrics(Path(str(run["run_dir"])))
    records = [
        json.loads(line) for line in metrics.read_text().splitlines() if line.strip()
    ]
    summary, histories = audit_records(records)
    deltas = [rows[-1][1] - rows[0][1] for rows in histories.values()]
    return {
        **summary,
        "seed": int(run["seed"]),
        "metrics": str(metrics),
        "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
        "mean_logprob_delta_min": min(deltas),
        "mean_logprob_delta_median": statistics.median(deltas),
        "drop_gt_half_fraction": sum(delta < -0.5 for delta in deltas) / len(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger",
        type=Path,
        nargs="?",
        default=Path("var/artifacts/e121_fixed_bank_survival_telemetry_jobs.json"),
    )
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    rows = [audit_run(run) for run in ledger["runs"]]
    receipt = {
        "schema": "e121_fixed_bank_survival_audit_v2",
        "ledger": str(args.ledger),
        "freeze_step": FREEZE_STEP,
        "runs": rows,
        "passed": all(row["insufficient_visit_count"] == 0 for row in rows),
    }
    output = args.ledger.with_name("e121_fixed_bank_survival_audit.json")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
