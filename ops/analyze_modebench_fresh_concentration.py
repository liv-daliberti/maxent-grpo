#!/usr/bin/env python3
"""Equal-prompt concentration contrasts from authenticated fresh ModeBench draws.

This offline reanalysis never generates responses or reruns a verifier.  The four
public summary/contrast helpers can also analyze newly authenticated collections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile

SCHEMA = "modebench-fresh-concentration-v1"
T95_DF4 = 2.7764451051977987
CONTRASTS = (("initial", "drgrpo"), ("initial", "replay_drgrpo"),
             ("drgrpo", "replay_drgrpo"))
EXPECTED_SEEDS = {"python_factors": (43, 44, 45, 46, 47),
                  "mathir": (43, 44, 45, 46, 47), "pantry_plan": (43, 46)}
OUTCOMES = ("collision", "mean_correct", "pass8_rarefied", "distinct8_rarefied",
            "extra8_rarefied", "pass_all", "distinct_all")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _mean(values):
    return statistics.mean(values) if values else None


def prompt_summary(keys, *, prompt_id, row_sha256=None):
    """Summarize all sampled slots; None is failure, any JSON key is verified.

    Collision is undefined with fewer than two correct draws. Rarefaction at
    min(8,n) uses the observed finite pool, not iid draws from an estimated law.
    """
    keys = list(keys)
    require(keys, "A prompt needs sampled slots, including failures")
    counts = Counter(json.dumps(k, sort_keys=True, separators=(",", ":"), allow_nan=False)
                     for k in keys if k is not None)
    n, r = len(keys), sum(counts.values())
    pairs = math.comb(r, 2)
    collisions = sum(math.comb(c, 2) for c in counts.values())
    k = min(8, n)
    denominator = math.comb(n, k)
    absent = lambda c: math.comb(n - c, k) / denominator if n - c >= k else 0.0
    p8 = 1.0 - absent(r)
    d8 = sum(1.0 - absent(c) for c in counts.values())
    return {"prompt_id": prompt_id, "row_sha256": row_sha256,
            "draws": n, "correct_draws": r, "mode_counts": sorted(counts.values(), reverse=True),
            "correct_pairs": pairs, "colliding_correct_pairs": collisions,
            "collision": collisions / pairs if pairs else None,
            "mean_correct": r / n, "rarefaction_k": k, "pass8_rarefied": p8,
            "distinct8_rarefied": d8, "extra8_rarefied": max(0.0, d8 - p8),
            "pass_all": float(r > 0), "distinct_all": len(counts)}


def per_prompt_contrast(left, right):
    """Return right-minus-left on the same prompt, preserving ineligibility."""
    require(left["prompt_id"] == right["prompt_id"], "Contrast prompt identities differ")
    require(left.get("row_sha256") == right.get("row_sha256"), "Contrast task hashes differ")
    require(left["draws"] == right["draws"], "Contrast draw budgets differ")
    eligible = left["correct_draws"] >= 2 and right["correct_draws"] >= 2
    return {"prompt_id": left["prompt_id"], "row_sha256": left.get("row_sha256"),
            "jointly_eligible": eligible,
            "ineligible_reason": None if eligible else (
                "both_R_lt_2" if left["correct_draws"] < 2 and right["correct_draws"] < 2
                else "left_R_lt_2" if left["correct_draws"] < 2 else "right_R_lt_2"),
            "left": left, "right": right,
            "delta": {m: right[m] - left[m] if m != "collision" or eligible else None
                      for m in OUTCOMES}}


def _unique_prompts(records):
    result = {}
    for record in records:
        require(record["prompt_id"] not in result, "Duplicate prompt identity")
        result[record["prompt_id"]] = record
    return result


def seed_contrast(left_records, right_records, *, training_seed, expected_prompt_ids):
    """Equal-prompt means on the joint R>=2 population; retain all prompts."""
    left, right = _unique_prompts(left_records), _unique_prompts(right_records)
    expected = list(expected_prompt_ids)
    require(len(expected) == len(set(expected)), "Duplicate expected prompt identity")
    require(set(left) == set(right) == set(expected), "Missing or extra comparison prompts")
    prompts = [per_prompt_contrast(left[p], right[p]) for p in sorted(expected)]
    joint = [p for p in prompts if p["jointly_eligible"]]
    populations = {}
    for name, rows in (("joint_R_ge_2", joint), ("all_prompts", prompts)):
        stats = {}
        for side in ("left", "right"):
            # Collision for all prompts is deliberately undefined when any
            # prompt is ineligible; an own-eligible mean is labeled separately.
            stats[side] = {m: _mean([p[side][m] for p in rows])
                           if rows and all(p[side][m] is not None for p in rows) else None
                           for m in OUTCOMES}
        stats["delta"] = {m: stats["right"][m] - stats["left"][m]
                          if stats["left"][m] is not None and stats["right"][m] is not None else None
                          for m in OUTCOMES}
        populations[name] = {"prompts": len(rows), **stats}
    pooled = {}
    own = {}
    for side in ("left", "right"):
        pair_count = sum(p[side]["correct_pairs"] for p in joint)
        collision_count = sum(p[side]["colliding_correct_pairs"] for p in joint)
        pooled[side] = {"correct_pairs": pair_count, "colliding_correct_pairs": collision_count,
                        "collision": collision_count / pair_count if pair_count else None}
        own_rows = [p[side] for p in prompts if p[side]["collision"] is not None]
        own[side] = {"eligible_prompts": len(own_rows),
                     "equal_prompt_collision": _mean([p["collision"] for p in own_rows])}
    a, b = pooled["left"]["collision"], pooled["right"]["collision"]
    pooled["delta"] = b - a if a is not None and b is not None else None
    equal_delta = populations["joint_R_ge_2"]["delta"]["collision"]
    return {"training_seed": training_seed, "expected_prompts": len(expected),
            "jointly_eligible_prompts": len(joint),
            "joint_coverage": len(joint) / len(expected) if expected else None,
            "defined": bool(joint), "undefined_reason": None if joint else "no_joint_R_ge_2_prompts",
            "populations": populations, "own_eligible": own,
            "pooled_pairs_same_joint_population": pooled,
            "pooled_minus_equal_prompt_delta": pooled["delta"] - equal_delta if joint else None,
            "prompts": prompts}


def aggregate_seed_contrasts(records, *, expected_seeds, shared_initial=False):
    """Average defined seed contrasts explicitly; t intervals require all five.

    A shared initial model is one fixed checkpoint/output pool, reused only as
    the reference for trained seeds. Intervals condition on that pool; it is
    never counted as five independent initial runs.
    """
    expected = list(expected_seeds)
    by_seed = {}
    for record in records:
        seed = record["training_seed"]
        require(seed not in by_seed, "Duplicate training seed")
        by_seed[seed] = record
    require(len(expected) == len(set(expected)), "Duplicate expected training seed")
    require(set(by_seed) == set(expected), "Missing or extra registered seed records")
    defined = [by_seed[s] for s in expected if by_seed[s]["defined"]]
    complete_five = len(expected) == 5 and len(defined) == 5
    estimates = {}
    for metric in OUTCOMES:
        values = [r["populations"]["joint_R_ge_2"]["delta"][metric] for r in defined]
        point = _mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else None
        se = sd / math.sqrt(5) if complete_five else None
        estimates[metric] = {"mean": point, "seed_sd": sd,
                             "ci95": [point - T95_DF4 * se, point + T95_DF4 * se]
                             if complete_five else None,
                             "interval_method": "paired_Student_t_df4_nominal" if complete_five else None}
    pooled_values = [r["pooled_pairs_same_joint_population"]["delta"] for r in defined]
    full = {metric: _mean([r["populations"]["all_prompts"]["delta"][metric]
                           for r in records])
            if all(r["populations"]["all_prompts"]["delta"][metric] is not None for r in records) else None
            for metric in OUTCOMES}
    return {"expected_seeds": expected, "defined_seeds": [r["training_seed"] for r in defined],
            "undefined_seeds": [s for s in expected if not by_seed[s]["defined"]],
            "n_expected": len(expected), "n_defined": len(defined),
            "status": "five_seed_nominal_interval" if complete_five else "descriptive_partial_or_two_seed",
            "shared_initial_output_pool": shared_initial,
            "independent_initial_checkpoints": 1 if shared_initial else 0,
            "uncertainty_scope": "trained-seed variation conditional on fixed prompts and the shared initial output pool"
            if shared_initial else "paired trained-seed variation conditional on fixed evaluation prompts",
            "eligible_prompt_counts": {str(s): by_seed[s]["jointly_eligible_prompts"] for s in expected},
            "joint_coverage_mean": _mean([r["joint_coverage"] for r in records]),
            "joint_population_effects": estimates,
            "all_prompt_effects_equal_seed": full,
            "pooled_pairs_delta_equal_seed": _mean(pooled_values),
            "equal_prompt_delta_equal_seed": estimates["collision"]["mean"],
            "pooled_minus_equal_prompt_delta": _mean(pooled_values) - estimates["collision"]["mean"]
            if defined else None,
            "seed_effects": {str(s): by_seed[s]["populations"]["joint_R_ge_2"]["delta"]["collision"]
                             for s in expected}}


def population_reconciliation(contrast):
    """Decompose population/prompt/seed weighting and fix the prompt population.

    Own-population pooled pairs reproduce the original per-checkpoint fresh64
    collision statistic. No pair count is treated as an independent sample size.
    The common sensitivity intersects eligibility over every registered seed.
    """
    seed_rows = contrast["seeds"]
    common = set.intersection(*[{p["prompt_id"] for p in r["prompts"] if p["jointly_eligible"]}
                                for r in seed_rows])
    records, own = [], []
    global_counts = {side: {"correct_pairs": 0, "colliding_correct_pairs": 0}
                     for side in ("left", "right")}
    for r in seed_rows:
        sides = {}
        for side in ("left", "right"):
            rows = [p[side] for p in r["prompts"] if p[side]["correct_draws"] >= 2]
            counts = {name: sum(p[name] for p in rows)
                      for name in ("correct_pairs", "colliding_correct_pairs")}
            for name, value in counts.items():
                global_counts[side][name] += value
            sides[side] = {**counts, "eligible_prompts": len(rows),
                           "collision": counts["colliding_correct_pairs"] / counts["correct_pairs"]
                           if counts["correct_pairs"] else None}
        a, b = sides["left"]["collision"], sides["right"]["collision"]
        own.append({"training_seed": r["training_seed"], **sides,
                    "delta": b-a if a is not None and b is not None else None})
        # Reconstructing the already preserved prompt summaries never revisits
        # a verifier or makes an ineligible prompt appear jointly correct.
        records.append(seed_contrast([p["left"] for p in r["prompts"] if p["prompt_id"] in common],
                                     [p["right"] for p in r["prompts"] if p["prompt_id"] in common],
                                     training_seed=r["training_seed"], expected_prompt_ids=sorted(common)))
    for counts in global_counts.values():
        counts["collision"] = counts["colliding_correct_pairs"] / counts["correct_pairs"] if counts["correct_pairs"] else None
    a, b = global_counts["left"]["collision"], global_counts["right"]["collision"]
    global_counts["delta"] = b-a if a is not None and b is not None else None
    # Empty fixed common populations are kept as empty/null, including all seeds.
    if not common:
        for r in records:
            r["joint_coverage"] = 0.0
    defined = [r for r in own if r["delta"] is not None]
    common_summary = aggregate_seed_contrasts(records, expected_seeds=contrast["summary"]["expected_seeds"],
                                              shared_initial=contrast["summary"]["shared_initial_output_pool"])
    for field in ("initial_weights_shared", "initial_sampling_replicas", "independent_initial_checkpoints",
                  "shared_initial_output_pool", "uncertainty_scope"):
        if field in contrast["summary"]:
            common_summary[field] = contrast["summary"][field]
    return {"identity": _flat_contrast_identity(contrast),
            "primary_equal_prompt_then_equal_seed": contrast["summary"]["equal_prompt_delta_equal_seed"],
            "joint_population_pair_pooled_then_equal_seed": contrast["summary"]["pooled_pairs_delta_equal_seed"],
            "own_population_pair_pooled_then_equal_seed": _mean([r["delta"] for r in defined]),
            "own_population_defined_seeds": [r["training_seed"] for r in defined],
            "own_population_per_seed": own,
            "own_population_all_pairs_pooled_across_seeds": global_counts,
            "fixed_common_prompt_ids": sorted(common), "fixed_common_prompts": len(common),
            "fixed_common_population": common_summary}


def publish_reconciliation(source, output):
    """Publish additional population sensitivities without changing prior output."""
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists(), "Supplement already exists; immutable outputs cannot be overwritten")
    manifest = json.loads((source.parent / "manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        require(file_binding(source.parent / relative)["sha256"] == digest, "Published reanalysis file changed")
    report = json.loads(source.read_text())
    reconciliations = [population_reconciliation(c) for c in report["contrasts"]]
    payload = {"schema": SCHEMA + "-population-reconciliation", "status": "complete",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_report": file_binding(source), "analysis_source": file_binding(__file__),
               "generation_calls": 0, "verifier_calls": 0, "old_artifacts_modified": False,
               "reconciliations": reconciliations,
               "limits": ["These retrospective sensitivities preserve all registered seeds and do not replace the primary equal-prompt estimand.",
                          "Own-eligible populations differ between policies. Pooling all pairs also weights seeds by their pair counts.",
                          "The fixed common prompt set is the intersection across all registered seeds within each contrast, not a new random test population.",
                          "Repeated shared initial counts in the all-pairs summary do not create independent initial observations; no interval is attached to that pooled diagnostic."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fresh-reconciliation-", dir=output.parent) as temp:
        stage = Path(temp)
        json_write(stage / "report.json", payload)
        rows = []
        lines = ["# Fresh64 population and weighting reconciliation", "",
                 "Authenticated supplement to the immutable primary reanalysis. All values are right minus left; negative collision means less concentration.", "",
                 "| Domain | Level | Wording | Contrast | Primary | Joint pooled/equal seed | Own pooled/equal seed | Own pooled/all pairs | Common prompts | Common primary |",
                 "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in reconciliations:
            row = {**r["identity"], "primary": r["primary_equal_prompt_then_equal_seed"],
                   "joint_pooled_equal_seed": r["joint_population_pair_pooled_then_equal_seed"],
                   "own_pooled_equal_seed": r["own_population_pair_pooled_then_equal_seed"],
                   "own_all_pairs": r["own_population_all_pairs_pooled_across_seeds"]["delta"],
                   "common_prompts": r["fixed_common_prompts"],
                   "common_primary": r["fixed_common_population"]["equal_prompt_delta_equal_seed"]}
            rows.append(row)
            if row["grading"] == "strict":
                lines.append(f"| {row['domain']} | {row['level']} | {row['wording']} | {row['contrast']} | {_fmt(row['primary'])} | {_fmt(row['joint_pooled_equal_seed'])} | {_fmt(row['own_pooled_equal_seed'])} | {_fmt(row['own_all_pairs'])} | {row['common_prompts']} | {_fmt(row['common_primary'])} |")
        lines += ["", "## Interpretation", ""] + ["- " + limit for limit in payload["limits"]]
        (stage / "README.md").write_text("\n".join(lines) + "\n")
        _csv(stage / "reconciliation.csv", rows)
        shutil.copy2(__file__, stage / Path(__file__).name)
        json_write(stage / "manifest.json", {"schema": payload["schema"] + "-manifest",
                "files": {str(p.relative_to(stage)): file_binding(p)["sha256"] for p in sorted(stage.rglob("*")) if p.is_file()}})
        require(not output.exists(), "Supplement appeared during execution; refusing overwrite")
        stage.rename(output)
    return output


def _discovery_module():
    path = Path(__file__).with_name("analyze_modebench_discovery_curves.py")
    spec = importlib.util.spec_from_file_location("fresh_concentration_authenticated_discovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_correction_receipts(base, plan, helper):
    """Bind cached corrected grades to the independent confirmation receipts."""
    local = Path(base) / "local"
    summary_path = local / "grading_correction_summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary["status"] == "complete" and summary["generation_calls"] == 0
            and summary["original_data_modified"] is False, "Incomplete correction summary")
    helper.bound_file(local / "plan.json", summary["plan_sha256"])
    helper.bound_file(local / "completion_integrity_audit.json", summary["completion_integrity_audit_sha256"])
    complete_path = local / "measurement_integrity/COMPLETE.json"
    helper.bound_file(complete_path, summary["independent_confirmation_complete_sha256"])
    complete = json.loads(complete_path.read_text())
    require(complete["status"] == "complete" and complete["generation_calls"] == 0
            and complete["original_data_modified"] is False, "Incomplete independent grade confirmations")
    entries = {entry["checkpoint"]: entry for entry in summary["entries"]}
    require(len(entries) == len(summary["entries"]) == len(plan["checkpoints"])
            and set(entries) == {c["label"] for c in plan["checkpoints"]}, "Correction receipt checkpoint inventory differs")
    bindings = [file_binding(summary_path), file_binding(complete_path)]
    for checkpoint in plan["checkpoints"]:
        label = checkpoint["label"]
        path = local / "measurement_integrity" / (label + ".json")
        expected = complete["confirmation_receipt_sha256"][str(path.resolve())]
        require(entries[label]["confirmation_receipt_sha256"] == expected, "Confirmation bindings differ")
        bindings.append(helper.bound_file(path, expected))
        receipt = json.loads(path.read_text())
        require(receipt["checkpoint"] == label and receipt["status"] == "confirmed"
                and receipt["generation_calls"] == 0 and receipt["modified_original_grades"] is False
                and receipt["modified_verifier_or_timeouts"] is False, "Invalid correction confirmation")
        directory = Path(plan["output_root"]) / label
        for name, field in (("responses.jsonl", "raw_responses_sha256"),
                            ("discovery_grades.jsonl", "grading_sidecar_sha256"),
                            ("discovery_grading_audit.json", "grading_audit_sha256")):
            helper.bound_file(directory / name, receipt[field])
        require(receipt["plan_sha256"] == helper.file_sha(local / "plan.json"), "Correction plan mismatch")
    return {"sources": bindings, "records": summary["records"],
            "strict_changed_records": summary["strict_changed_records"],
            "raw_verified": summary["raw_verified"], "strict_verified": summary["strict_verified"],
            "normalized_verified": summary["normalized_verified"],
            "cause_attribution": summary["cause_attribution"]}


def build_report(base):
    helper = _discovery_module()
    base = Path(base).resolve()
    design = helper.authenticate_design(base)
    plan_path = base / "local/plan.json"
    plan = json.loads(plan_path.read_text())
    helper.validate_local_plan(design, plan_path, plan)
    corrections = authenticate_correction_receipts(base, plan, helper)
    models, sources = {}, []
    total_slots = 0
    for checkpoint in plan["checkpoints"]:
        print("Authenticating " + checkpoint["label"], flush=True)
        runs = helper.authenticate_local(design, plan_path, checkpoint)
        grouped = {}
        for wording, run in zip(helper.ARMS, runs):
            for grading in helper.GRADINGS:
                for key, row in sorted(run["rows"].items()):
                    samples = [run[grading][(*key, i)] for i in range(helper.DRAWS)]
                    # The existing loader validates every slot's child seed,
                    # original durable batch receipt, and cached grading hash.
                    keys = [s["canonical_key"] if s["verified"] else None for s in samples]
                    record = prompt_summary(keys, prompt_id="/".join(map(str, key)), row_sha256=helper.sha(row))
                    record.update(level=key[0], domain=key[1], row_index=key[2],
                                  support_reference=design["references"][key])
                    grouped.setdefault((grading, wording, key[0], key[1]), []).append(record)
        total_slots += checkpoint["expected_draws"]
        models[checkpoint["label"]] = {"checkpoint": checkpoint, "groups": grouped}
        sources.append({"checkpoint": checkpoint["label"], "sources": runs[0]["sources"],
                        "grading_audit": runs[0]["grading_audit"]})
    initial = [m for m in models.values() if m["checkpoint"]["training_method"] == "initial"]
    require(len(initial) == 1, "Require exactly one fixed initial checkpoint")
    initial = initial[0]
    inventory = {}
    for m in models.values():
        c = m["checkpoint"]
        if c["training_method"] == "initial":
            continue
        domain = "pantry_plan" if c["domain"] == "pantry" else c["domain"]
        index = domain, c["training_method"], c["training_seed"]
        require(index not in inventory, "Duplicate registered model/domain/seed")
        inventory[index] = m
    require(set(inventory) == {(d, m, s) for d, seeds in EXPECTED_SEEDS.items()
                              for m in ("drgrpo", "replay_drgrpo") for s in seeds},
            "Fresh64 training-seed inventory differs")
    contrasts = []
    for grading in helper.GRADINGS:
        for wording in helper.ARMS:
            for level in helper.LEVELS:
                for domain, seeds in EXPECTED_SEEDS.items():
                    group = grading, wording, level, domain
                    expected_ids = ["/".join(map(str, k)) for k in sorted(design["rows"]) if k[:2] == (level, domain)]
                    for left_method, right_method in CONTRASTS:
                        seed_rows = []
                        for seed in seeds:
                            left_model = initial if left_method == "initial" else inventory[domain, left_method, seed]
                            right_model = inventory[domain, right_method, seed]
                            seed_rows.append(seed_contrast(left_model["groups"][group], right_model["groups"][group],
                                                          training_seed=seed, expected_prompt_ids=expected_ids))
                        contrasts.append({"grading": grading, "wording": wording, "level": level, "domain": domain,
                                          "contrast": right_method + "_minus_" + left_method,
                                          "left_method": left_method, "right_method": right_method,
                                          "summary": aggregate_seed_contrasts(seed_rows, expected_seeds=seeds,
                                                                              shared_initial=left_method == "initial"),
                                          "seeds": seed_rows})
    return {"schema": SCHEMA, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "retrospective_reanalysis": True, "generation_calls": 0, "verifier_calls": 0,
            "old_artifacts_modified": False, "input_campaign": str(base),
            "scope": {"model": "Qwen2.5-0.5B", "trained_on_level": 2, "evaluation_levels": [2, 3],
                      "wordings": list(helper.ARMS), "domains": list(EXPECTED_SEEDS),
                      "grading": {"primary": "strict", "sensitivity": "normalized_secondary"},
                      "prompts_per_cell": 16, "fresh_draws_per_prompt": 64,
                      "unique_checkpoint_cohorts": len(models), "unique_response_slots": total_slots,
                      "initial_checkpoint_count": 1, "contrast_cells": len(contrasts)},
            "estimand": {"primary": "right-minus-left correct-key collision, equal prompts on joint R>=2, then equal defined training seeds",
                         "collision": "sum_c n_c(n_c-1)/(R(R-1))",
                         "undefined_policy": "keep every registered prompt and seed; null when no joint eligible prompt; no interval unless all five seeds defined",
                         "paired_reference": "initial output pool reused as one fixed reference, never independent training-seed replicas",
                         "reconciliation": "pair-pooled collision on each identical joint population, then equal seed means; not substituted for primary"},
            "design_sources": design["sources"], "plan_source": file_binding(plan_path),
            "correction_audit": corrections, "checkpoint_sources": sources,
            "analysis_sources": [file_binding(__file__), file_binding(Path(helper.__file__))],
            "contrasts": contrasts,
            "limitations": [
                "Retrospective contrast specification on existing sampled outcomes; nominal pointwise intervals are not multiplicity adjusted.",
                "Sixteen fixed problems per cell; Student-t intervals describe paired trained-seed variability, not unseen-prompt uncertainty.",
                "Initial comparisons condition on one shared initial checkpoint and output pool; initial sampling uncertainty is not replicated across seeds.",
                "Pantry uses two fixed training seeds (43,46) and remains descriptive, including when both have eligible prompts.",
                "Joint correctness eligibility changes the population across contrasts and seeds; correctness is reported on the same selected prompts.",
                "Within each prompt's 64 outputs child-stream identifiers are distinct; identifiers are reused across checkpoints and wording conditions. This is not proof of iid model samples.",
                "Lower collision is conditional concentration, not proof of larger full support, recovery of particular modes, or diversity-caused accuracy gains.",
                "These are Level-2-trained checkpoints evaluated on Level-2 and Level-3 prompts, not the main Level-1 training population.",
                "Eight-draw metrics are without-replacement rarefactions of the fresh64 pool, not historical eight-draw endpoints.",
                "The existing warmed serial grading cache and independent correction receipts are reused; original raw grades remain unchanged."
            ]}


def _flat_contrast_identity(c):
    return {k: c[k] for k in ("grading", "wording", "level", "domain", "contrast", "model_scale") if k in c}


def _csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    return "undefined" if value is None else f"{value:+.4f}"


def render_markdown(report):
    panel = report.get("scope", {}).get("panel") == "registered_Level1_fresh"
    title = "Registered Level-1 fresh concentration panel" if panel else "Fresh64 concentration reanalysis"
    scope = ("Original Level-1 Graph/Pantry prompts, three model scales, four trained methods, and five initial Monte Carlo replicas of the same weights per model/domain. Cached collector grades are authenticated to committed batches and frozen code." if panel else
             "Strict cached grading is primary; frozen formatting normalization is a sensitivity. Initial comparisons condition on one fixed initial output pool; Pantry's two fixed training seeds are descriptive.")
    lines = ["# " + title, "", "Created: " + report["created_at_utc"], "",
             "No responses were generated or regraded by this analysis. " + scope, "",
             "The primary contrast averages promptwise correct-key collision differences on the joint R≥2 population, then weights defined training seeds equally. Negative values mean less concentration in the right-hand method. Every missing conditional seed remains explicit; only five fully defined seeds receive nominal paired Student-t intervals.", "",
             "## Strict primary contrasts", "",
             "| Model | Domain | Level | Wording | Right minus left | Defined seeds | Joint prompts by seed | Δ collision | Nominal 95% interval | Δ correctness on same prompts | Pair-pooled Δ |",
             "|---|---|---:|---|---|---|---|---:|---|---:|---:|"]
    for c in report["contrasts"]:
        if c["grading"] != "strict":
            continue
        s = c["summary"]; e = s["joint_population_effects"]["collision"]
        ci = "descriptive" if e["ci95"] is None else "[" + ", ".join(_fmt(x) for x in e["ci95"]) + "]"
        denominator = report["scope"]["prompts_per_cell"]
        coverage = ", ".join(f"{seed}:{n}/{denominator}" for seed,n in s["eligible_prompt_counts"].items())
        lines.append(f"| {c.get('model_scale', 'Qwen0.5B')} | {c['domain']} | {c['level']} | {c['wording']} | {c['contrast']} | {s['n_defined']}/{s['n_expected']} | {coverage} | {_fmt(e['mean'])} | {ci} | {_fmt(s['joint_population_effects']['mean_correct']['mean'])} | {_fmt(s['pooled_pairs_delta_equal_seed'])} |")
    if not panel:
        lines += ["", "## Reading the Pantry results", "",
                  "The Pantry rows include every requested initial→Dr.GRPO, initial→Re:Dr.GRPO, and Dr.GRPO→Re:Dr.GRPO comparison under both wordings and both levels. Their collision direction is an observation on the displayed jointly correct prompts, not a five-seed replication."]
    lines += ["", "`seed_contrasts.csv` preserves individual seed effects and same-population correctness; `prompt_contrasts.csv` preserves every eligible and ineligible prompt.", "",
              "## Equal-prompt versus pair-pooled collision", "",
              "The final column pools correct pairs within each seed on exactly the same joint prompt set before averaging seeds. The primary column first weights prompts equally. Their difference is a weighting/estimand difference; a reversal is retained. JSON records numerator and denominator counts. Correctness columns are per-response success rates on the same selected prompts, not full-test pass@8.", "", "## Limits", ""]
    lines += ["- " + x for x in report["limitations"]]
    return "\n".join(lines) + "\n"


def write_artifacts(report, output):
    """Publish a new directory only; never overwrite an existing result."""
    output = Path(output).resolve()
    require(not output.exists(), "Output already exists; immutable reanalyses cannot be overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fresh-concentration-", dir=output.parent) as temp:
        stage = Path(temp)
        prompt_rows, seed_rows, aggregate_rows = [], [], []
        for c in report["contrasts"]:
            ident = _flat_contrast_identity(c)
            s = c["summary"]
            aggregate_rows.append({**ident, "n_expected": s["n_expected"], "n_defined": s["n_defined"],
                                   "defined_seeds": json.dumps(s["defined_seeds"]), "undefined_seeds": json.dumps(s["undefined_seeds"]),
                                   "mean_collision_delta": s["equal_prompt_delta_equal_seed"],
                                   "collision_ci95": json.dumps(s["joint_population_effects"]["collision"]["ci95"]),
                                   "mean_correctness_delta_same_prompts": s["joint_population_effects"]["mean_correct"]["mean"],
                                   "pooled_pairs_delta_equal_seed": s["pooled_pairs_delta_equal_seed"],
                                   "pooled_minus_equal_prompt_delta": s["pooled_minus_equal_prompt_delta"],
                                   "shared_initial_output_pool": s["shared_initial_output_pool"]})
            for r in c["seeds"]:
                row = {**ident, "training_seed": r["training_seed"], "expected_prompts": r["expected_prompts"],
                       "joint_prompts": r["jointly_eligible_prompts"], "defined": r["defined"], "undefined_reason": r["undefined_reason"],
                       "joint_coverage": r["joint_coverage"], "pooled_collision_delta": r["pooled_pairs_same_joint_population"]["delta"]}
                for pop, values in r["populations"].items():
                    for side in ("left", "right", "delta"):
                        row.update({f"{pop}_{side}_{m}": values[side][m] for m in OUTCOMES})
                seed_rows.append(row)
                for p in r["prompts"]:
                    row = {**ident, "training_seed": r["training_seed"], "prompt_id": p["prompt_id"],
                           "row_sha256": p["row_sha256"], "jointly_eligible": p["jointly_eligible"],
                           "ineligible_reason": p["ineligible_reason"]}
                    for side in ("left", "right"):
                        for m in (*OUTCOMES, "draws", "correct_draws", "correct_pairs", "colliding_correct_pairs"):
                            row[f"{side}_{m}"] = p[side][m]
                    row.update({"delta_" + m: p["delta"][m] for m in OUTCOMES})
                    prompt_rows.append(row)
        json_write(stage / "report.json", report)
        (stage / "README.md").write_text(render_markdown(report))
        _csv(stage / "contrasts.csv", aggregate_rows)
        _csv(stage / "seed_contrasts.csv", seed_rows)
        _csv(stage / "prompt_contrasts.csv", prompt_rows)
        (stage / "code").mkdir()
        for source in report["analysis_sources"]:
            require(file_binding(source["path"])["sha256"] == source["sha256"], "Analysis source changed during execution")
            shutil.copy2(source["path"], stage / "code" / Path(source["path"]).name)
        manifest = {"schema": SCHEMA + "-manifest", "status": "complete", "created_at_utc": report["created_at_utc"],
                    "files": {str(p.relative_to(stage)): file_binding(p)["sha256"] for p in sorted(stage.rglob("*")) if p.is_file()},
                    "source_report_sha256": file_binding(stage / "report.json")["sha256"]}
        json_write(stage / "manifest.json", manifest)
        require(not output.exists(), "Output appeared during execution; refusing overwrite")
        stage.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("artifacts/modebench_discovery_curves_20260911"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/modebench_fresh_concentration_20260912/reanalysis"))
    parser.add_argument("--reconcile-existing", type=Path, help="published report.json to supplement with population/weighting sensitivities")
    args = parser.parse_args()
    if args.reconcile_existing:
        print(publish_reconciliation(args.reconcile_existing, args.output))
        return
    require(not args.output.exists(), "Output already exists; choose a new immutable analysis directory")
    report = build_report(args.base)
    print(write_artifacts(report, args.output))


if __name__ == "__main__":
    main()
