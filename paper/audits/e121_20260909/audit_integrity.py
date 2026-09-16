#!/usr/bin/env python3
"""Independent E121 population, schedule, score, and provenance audit.

Observations are actual optimizer updates 384..3072. The final trainer/step
3073 record carries forward update3072's metrics and is not another visit.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LEDGER = ROOT / "var/artifacts/e121_fixed_bank_survival_telemetry_jobs.json"
FREEZE = 384
HORIZON = 3072


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(row, stem):
    prefix = "train/canonical_replay_" + stem
    return {k[len(prefix) :]: v for k, v in row.items() if k.startswith(prefix)}


def get(row, name):
    return row["train/" + name]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def audit(run):
    d = Path(run["run_dir"])
    receipt_path = d / "TRAINING_COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    attempt = Path(receipt["terminal_attempt"])
    attempt.resolve().relative_to(d.resolve())
    path = attempt / "train_metrics.jsonl"
    raw = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    updates = [r for r in raw if 1 <= r["trainer/step"] <= HORIZON]
    check(
        [r["trainer/step"] for r in updates] == list(range(1, HORIZON + 1)),
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
    extra = [r for r in raw if r["trainer/step"] > HORIZON]
    check(
        len(extra) == 1
        and extra[0]["trainer/step"] == 3073
        and extra[0]["trainer/global_step"] == 3072,
        "unexpected terminal logging convention",
    )
    check(
        {k: v for k, v in extra[0].items() if k.startswith("train/")}
        == {k: v for k, v in updates[-1].items() if k.startswith("train/")},
        "terminal replay not a carried-forward duplicate",
    )
    check(receipt["terminal_step"] == 3073, "unexpected receipt step")
    check(
        all(get(r, "canonical_replay_bank_freeze_step") == FREEZE for r in updates),
        "freeze config",
    )
    check(
        all(
            get(r, "canonical_replay_bank_membership_frozen")
            == float(r["trainer/step"] >= FREEZE)
            for r in updates
        ),
        "freeze flag boundary",
    )
    post = [r for r in updates if r["trainer/step"] >= FREEZE]
    prompt_sizes = {int(get(r, "online_canonical_tracked_prompts")) for r in post}
    outcome_sizes = {int(get(r, "online_canonical_tracked_outcomes")) for r in post}
    check(len(prompt_sizes) == len(outcome_sizes) == 1, "full-bank counts changed")
    n_prompts = prompt_sizes.pop()
    n_outcomes = outcome_sizes.pop()
    check(
        int(get(updates[FREEZE - 2], "online_canonical_tracked_prompts")) == n_prompts,
        "freeze admitted prompt",
    )
    check(
        int(get(updates[FREEZE - 2], "online_canonical_tracked_outcomes"))
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
        vals = {stem: indexed(r, stem) for stem in row_stems + group_stems}
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
            histories[ident].append(step)
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
        all(rows == prompt_steps[ident[0]] for ident, rows in histories.items()),
        "missing scheduled identity observation",
    )
    check(all(len(rows) >= 2 for rows in histories.values()), "fewer than two visits")
    check(sum(len(rows) for rows in histories.values()) == totalrows, "row accounting")
    return dict(
        seed=run["seed"],
        job_id=run["job_id"],
        previous_job_ids=run.get("previous_job_ids", []),
        run_dir=str(d),
        metrics=str(path),
        metrics_sha256=sha(path),
        completion_receipt_sha256=sha(receipt_path),
        metrics_records=len(raw),
        optimizer_updates=len(updates),
        terminal_receipt_step=receipt["terminal_step"],
        terminal_carried_forward_records_excluded=len(extra),
        postfreeze_first_step=FREEZE,
        postfreeze_final_step=HORIZON,
        postfreeze_update_count=len(post),
        frozen_prompts=n_prompts,
        frozen_identities=n_outcomes,
        observed_prompts=len(prompt_steps),
        observed_identities=len(histories),
        identity_observations=totalrows,
        min_visits=min(map(len, histories.values())),
        max_visits=max(map(len, histories.values())),
        frozen_count_min=min(next(iter(v)) for v in counts.values()),
        frozen_count_max=max(next(iter(v)) for v in counts.values()),
        bank_support_max=max(map(len, outcomes.values())),
        token_length_min=min(next(iter(v)) for v in lengths.values()),
        token_length_max=max(next(iter(v)) for v in lengths.values()),
        maximum_inferred_integer_token_count_error=max_length_ratio_error,
        maximum_group_token_total_error=max_token_total_error,
        complete_round_robin_schedule=True,
        complete_finite_score_fraction=1.0,
        missing_identity_count=0,
        insufficient_visit_count=0,
        changing_membership_count=0,
        changing_fresh_count_identity_count=0,
        checkpoints_pruned=receipt["resume_checkpoints_pruned"],
        attempt_directories=[p.name for p in d.glob("debug_job*") if p.is_dir()],
        passed=True,
    )


def main():
    ledger = json.loads(LEDGER.read_text())
    rows = [audit(run) for run in ledger["runs"]]
    ids = ",".join(str(r["job_id"]) for r in rows)
    cmd = [
        "sacct",
        "-n",
        "-X",
        "-P",
        "-j",
        ids,
        "--format=JobID,State,ExitCode,Start,End,Elapsed,NodeList",
    ]
    accounting = subprocess.run(
        cmd, text=True, capture_output=True, check=True
    ).stdout.strip()
    check(
        len(accounting.splitlines()) == 5
        and all("|COMPLETED|0:0|" in line for line in accounting.splitlines()),
        "noncompleted scheduler state",
    )
    record = dict(
        schema="e121_independent_integrity_audit_v1",
        ledger=str(LEDGER),
        ledger_sha256=sha(LEDGER),
        preregistration_sha256=sha(Path(ledger["protocol"])),
        snapshot_identity_sha256=ledger["snapshot_identity_sha256"],
        scheduler_accounting=accounting,
        registered_horizon=HORIZON,
        freeze_step=FREEZE,
        runs=rows,
        total_frozen_identities=sum(r["frozen_identities"] for r in rows),
        total_seed_prompt_pairs=sum(r["frozen_prompts"] for r in rows),
        total_identity_observations=sum(r["identity_observations"] for r in rows),
        passed=True,
        caveats=[
            "Terminal trainer/step3073 record duplicates replay telemetry from optimizer update3072 and is excluded.",
            "Successful-run cleanup pruned bank checkpoints. Frozen-population completeness is established by matching identity cardinalities to full-bank tracked counts, complete round-robin cycles, and unchanged explicit memberships.",
            "Fingerprint telemetry is numerically rounded during metric aggregation; cardinality equality with full-bank counts and complete cyclic schedules show no within-run identity collisions in this cohort.",
            "Sequence token counts are reconstructed as sequence_logprob/mean_logprob because telemetry stores total replay tokens rather than row token counts; all ratios are near positive integers, constant by identity, and sum exactly to logged replay-token totals.",
            "E121 dependency barriers were released early on2026-09-08 by a documented user-authorized scheduling amendment; scientific configuration and seed identities were unchanged.",
            "Scores are normalized teacher-forced exemplar log probabilities, not canonical-mode probabilities; there is no no-replay control in E121.",
        ],
    )
    (OUT / "integrity.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ["runs", "caveats"]}, indent=2
        )
    )
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "seed",
                        "frozen_prompts",
                        "frozen_identities",
                        "identity_observations",
                        "min_visits",
                        "max_visits",
                        "passed",
                    ]
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
