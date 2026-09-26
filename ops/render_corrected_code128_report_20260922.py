#!/usr/bin/env python3
"""Render the completed, independently audited fixed coding128 pilot.

Consumes results only; performs no inference, training, submission, regrading,
or protocol edits. Final128 is always primary. CPU sanity fixtures are confined
to a temporary directory, explicitly watermarked, and never accepted as results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
METHODS_DOCUMENTS = (ROOT / "docs/codecontests_hardened_adapter_handbook_20260921.md", ROOT / "docs/codecontests_pilot_citations_20260921.md")
SCHEMA = "independent-native-hf-code128-audit-20260922-v1"
TRAIN_IDS = ["1454_A", "1569_A", "361_A", "1408_A", "988_A", "1323_A", "1380_A", "1038_B"]
DEV_IDS = ["1513_A", "1352_B", "1016_D", "1360_G", "244_A", "1102_B", "1095_C", "1352_G", "482_A", "1339_B", "1371_D", "1051_B", "545_B"]
ARMS = ("base", "maxrl", "remax")
LABELS = {"base": "Base", "maxrl": "MaxRL", "remax": "Re:Max"}
COLORS = {"base": "#777777", "maxrl": "#2876B8", "remax": "#D55E00"}
COHORTS = {"train": (TRAIN_IDS, "Training", "8 questions"), "development": (DEV_IDS, "Development", "13 questions")}
METRICS = ("accuracy", "pass8", "ed8", "ed32", "pcmd")
METRIC_LABELS = {"accuracy": "Accuracy", "pass8": "Pass@8", "ed8": "Distinct correct @8", "ed32": "Distinct correct @32", "pcmd": "Conditional diversity"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value, *, nullable=False):
    return (nullable and value is None) or (type(value) in (int, float) and math.isfinite(value))


def close(left, right):
    return left is right if left is None or right is None else math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def value(group, arm, metric):
    if metric == "pcmd":
        return group["common_pair"]["macro_pcmd"].get(arm)
    summary = group["arm_wise"][arm]
    if metric == "accuracy":
        return summary["macro_accuracy"]
    if metric == "pass8":
        return summary["macro_pass_at_k"]["8"]
    return summary["macro_expected_distinct_valid_modes_at_k"]["8" if metric == "ed8" else "32"]


def validate_summary(summary, *, fixture=False):
    require(summary.get("synthetic_renderer_fixture", False) is fixture, "synthetic/validation fixtures cannot be rendered as results")
    require(summary.get("schema") == SCHEMA and summary.get("status") == "pass" and summary.get("kind") == "complete_fixed_native_code128_comparison", "completed authoritative auditor summary required")
    require(summary.get("primary_checkpoint") == 128 and summary.get("paired_training_seeds") == 1 and summary.get("samples") == 23040, "fixed primary, seed count or total denominator differs")
    require(len(summary.get("shards", [])) == 13 and set(summary["checkpoints"]) == {"32", "64", "128"}, "complete13-shard/3-checkpoint comparison required")
    for step in (32, 64, 128):
        checkpoint = summary["checkpoints"][str(step)]
        require(checkpoint["role"] == ("primary_final" if step == 128 else "descriptive_trajectory") and set(checkpoint["cohorts"]) == set(COHORTS), "checkpoint role or cohort differs")
        for cohort, (ids, _, _) in COHORTS.items():
            group = checkpoint["cohorts"][cohort]
            n = (512 if cohort == "train" else 128) if step == 128 else (128 if cohort == "train" else 32)
            require(group["task_ids"] == ids and group["samples_per_task"] == n and set(group["arm_wise"]) == set(ARMS) == set(group["task_metrics"]), "fixed cohort or per-task draws differ")
            for arm in ARMS:
                tasks, aggregate = group["task_metrics"][arm], group["arm_wise"][arm]
                require(set(tasks) == set(ids), "task omitted from denominator")
                eligible = [task for task in ids if tasks[task]["accepted"] >= 30]
                for task, row in tasks.items():
                    require(row["samples"] == n and type(row["accepted"]) is int and 0 <= row["accepted"] <= n, "invalid sample counts")
                    require(row["pcmd_accepted_threshold"] == 30 and row["pcmd_eligible"] == (task in eligible) and finite(row["pcmd"], nullable=True) and ((row["pcmd"] is None) == (task not in eligible)), "PCMD threshold/null semantics differ")
                    require(finite(row["accuracy"]) and close(row["accuracy"], row["accepted"] / n), "invalid task accuracy")
                    require(all(finite(row[key][str(k)]) for key in ("pass_at_k", "expected_distinct_valid_modes_at_k") for k in (1, 8, 32)), "nonfinite task metric")
                require(aggregate["tasks"] == aggregate["pcmd_total_tasks"] == len(ids) and aggregate["samples"] == n * len(ids) and aggregate["accepted"] == sum(tasks[t]["accepted"] for t in ids), "aggregate denominator differs")
                require(aggregate["pcmd_eligible_task_ids"] == eligible and aggregate["pcmd_eligible_tasks"] == len(eligible), "arm-wise eligibility differs")
                expected_pcmd = sum(tasks[t]["pcmd"] for t in eligible) / len(eligible) if eligible else None
                require(close(aggregate["macro_pcmd_over_eligible_tasks"], expected_pcmd), "arm-wise PCMD differs")
                require(close(aggregate["macro_accuracy"], sum(tasks[t]["accuracy"] for t in ids) / len(ids)), "aggregate accuracy differs")
                for output, source in (("macro_pass_at_k", "pass_at_k"), ("macro_expected_distinct_valid_modes_at_k", "expected_distinct_valid_modes_at_k")):
                    for k in (1, 8, 32):
                        require(close(aggregate[output][str(k)], sum(tasks[t][source][str(k)] for t in ids) / len(ids)), "aggregate metric differs")
            for key, arms in (("common_pair", ("maxrl", "remax")), ("common_all_three", ARMS)):
                selected = [t for t in ids if all(group["task_metrics"][a][t]["pcmd_eligible"] for a in arms)]
                common = group[key]
                require(common["task_ids"] == selected and common["eligible_tasks"] == len(selected) and common["total_tasks"] == len(ids) and set(common["macro_pcmd"]) == set(arms), "common-set eligibility differs")
                for arm in arms:
                    expected = sum(group["task_metrics"][arm][t]["pcmd"] for t in selected) / len(selected) if selected else None
                    require(close(common["macro_pcmd"][arm], expected), "common-set PCMD differs")
                expected = common["macro_pcmd"]["remax"] - common["macro_pcmd"]["maxrl"] if selected else None
                require(close(common["remax_minus_maxrl"], expected), "common-set contrast differs")
    return summary


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold", "axes.labelsize": 9, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save_figure(fig, directory, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")


def figures(summary, directory, *, fixture=False):
    plt = plot_style()
    from matplotlib.patches import Patch
    watermark = "SYNTHETIC RENDERER FIXTURE — NOT RESULTS" if fixture else "One paired training seed; no seed confidence intervals."
    final = summary["checkpoints"]["128"]["cohorts"]
    fig, axes = plt.subplots(2, 5, figsize=(13.2, 5.9), squeeze=False)
    for ri, (cohort, (_, label, count)) in enumerate(COHORTS.items()):
        group = final[cohort]
        for ci, metric in enumerate(METRICS):
            ax = axes[ri][ci]
            arms = ("maxrl", "remax") if metric == "pcmd" else ARMS
            vals = [value(group, arm, metric) for arm in arms]
            if all(v is None for v in vals):
                ax.text(.5, .45, "Undefined\n0 eligible questions", transform=ax.transAxes, ha="center", va="center", color="#555555")
                ax.set_ylim(0, 1)
            else:
                bars = ax.bar(range(len(arms)), vals, color=[COLORS[a] for a in arms], width=.68)
                ceiling = 1.16 if metric in ("accuracy", "pass8", "pcmd") else max(1., max(vals) * 1.28)
                ax.set_ylim(0, ceiling)
                for bar, val in zip(bars, vals):
                    text = f"{100 * val:.1f}%" if metric in ("accuracy", "pass8") else f"{val:.3f}"
                    ax.annotate(text, (bar.get_x() + bar.get_width()/2, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xticks(range(len(arms)), [LABELS[a] for a in arms], fontsize=8)
            if metric in ("accuracy", "pass8"):
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            if metric == "pcmd":
                ax.set_title(f"{METRIC_LABELS[metric]}\nCommon pair: {group['common_pair']['eligible_tasks']}/{len(group['task_ids'])}")
            else:
                ax.set_title(METRIC_LABELS[metric])
            ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e4e4e4", linewidth=.6)
            if ci == 0:
                ax.set_ylabel(f"{label}\n{count}")
    fig.suptitle("Coding pilot: fixed final checkpoint at 128 updates", fontsize=14, y=.99)
    fig.text(.5, .018, "512 draws per training question; 128 per development question. PCMD requires ≥30 accepted answers in both arms.\n" + watermark, ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.96), h_pad=2., w_pad=1.5)
    save_figure(fig, directory, "final128"); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4), squeeze=False)
    for ri, (cohort, (_, label, count)) in enumerate(COHORTS.items()):
        for ci, metric in enumerate(("accuracy", "ed32", "pcmd")):
            ax = axes[ri][ci]
            arms = ("maxrl", "remax") if metric == "pcmd" else ARMS
            for arm in arms:
                vals = [value(summary["checkpoints"][str(step)]["cohorts"][cohort], arm, metric) for step in (32,64,128)]
                ax.plot((32,64,128), [float("nan") if v is None else v for v in vals], marker="o", markersize=4, linewidth=1.6, linestyle="--" if arm == "base" else "-", color=COLORS[arm])
            ax.set_xticks([32,64,128]); ax.set_xlim(22,140); ax.set_xlabel("Training updates")
            if metric in ("accuracy", "pcmd"):
                ax.set_ylim(0, 1.04)
            else:
                ax.set_ylim(bottom=0)
            if metric == "accuracy":
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            if metric == "pcmd":
                group_ids = COHORTS[cohort][0]
                counts = [summary["checkpoints"][str(step)]["cohorts"][cohort]["common_pair"]["eligible_tasks"] for step in (32,64,128)]
                ax.set_title("Conditional diversity, common pair")
                ax.text(.03, .97, "Common eligible (32/64/128)\n" + " · ".join(f"{n}/{len(group_ids)}" for n in counts), transform=ax.transAxes, va="top", fontsize=8)
                if not any(counts):
                    ax.text(.5, .45, "Undefined throughout", transform=ax.transAxes, ha="center", color="#555555")
            else:
                ax.set_title(METRIC_LABELS[metric])
            ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e4e4e4", linewidth=.6)
            if ci == 0:
                ax.set_ylabel(f"{label}\n{count}")
    fig.suptitle("Descriptive trajectory; final128 remains the primary comparison", fontsize=13, y=.99)
    fig.legend(handles=[Patch(facecolor=COLORS[a], label=LABELS[a] + (" (matched draws)" if a == "base" else "")) for a in ARMS], loc="upper center", ncol=3, bbox_to_anchor=(.5,.945), frameon=False)
    fig.text(.5, .018, "At 32/64: 128 training and 32 development draws per question; at 128: 512 and 128.\nBase uses matching sample prefixes; PCMD common question sets may change between checkpoints.\n" + watermark, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0,.13,1,.90), h_pad=2., w_pad=1.6)
    save_figure(fig, directory, "trajectory"); plt.close(fig)


def fmt(value, *, percent=False, signed=False):
    if value is None:
        return "undefined"
    return format(value * (100 if percent else 1), ("+" if signed else "") + ".3f") + ("%" if percent else "")


def training_diagnostics(path, *, expected_identity_sha256, arm):
    path = Path(path)
    if path.is_dir():
        path = path / ("metrics.jsonl" if (path / "metrics.jsonl").exists() else "training/metrics.jsonl")
    identity_path = path.parent / "identity.json"
    require(digest(identity_path) == expected_identity_sha256, "optional training metrics belong to a different audited parent")
    identity = json.loads(identity_path.read_text())
    require(identity["arm"] == arm and identity["config"]["updates"] == 128, "optional training arm or target differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require([r["completed_updates"] for r in rows] == list(range(1,129)), "optional training diagnostics must contain all128 updates")
    require(all(all(finite(r[k]) and r[k] == 0 for k in ("logprobs_diff_min", "logprobs_diff_max", "pg_clipfrac")) for r in rows), "optional training metrics violate corrected scoring contract")
    require(all(finite(r['fresh_successes']) and float(r['fresh_successes']).is_integer() and 0 <= r['fresh_successes'] <= 16 for r in rows), "invalid fresh group counts")
    return {"path": str(path.resolve()), "sha256": digest(path), "parent_identity_sha256": expected_identity_sha256, "updates": 128, "accepted": sum(int(r["fresh_successes"]) for r in rows), "fresh_samples": 2048, "mixed_groups": sum(0 < r["fresh_successes"] < 16 for r in rows), "all_correct_groups": sum(r["fresh_successes"] == 16 for r in rows), "all_wrong_groups": sum(r["fresh_successes"] == 0 for r in rows), "final_bank_modes": rows[-1]["bank_modes"]}


def accounting_snapshot(path, *, expected_summary_sha=None):
    data = json.loads(Path(path).read_text())
    terminal = data.get("schema") == "corrected-code128-terminal-accounting-20260922-v1"
    require(terminal or data.get("schema") == "corrected-code128-driver-status-20260922-v1", "unsupported campaign accounting schema")
    rows = data["jobs"]
    require(len({r["job_id"] for r in rows}) == len(rows), "duplicate scheduler job in accounting")
    require(all(finite(r["allocated_gpu_hours"]) and r["allocated_gpu_hours"] >= 0 for r in rows), "invalid scheduler GPU-hour value")
    total = sum(r["allocated_gpu_hours"] for r in rows)
    common = {"path": str(Path(path).resolve()), "sha256": digest(path), "allocated_gpu_hours": total, "jobs": len(rows), "status": data["status"], "terminal": terminal, "states": dict(sorted({state: sum(r["state"] == state for r in rows) for state in {r["state"] for r in rows}}.items()))}
    if terminal:
        allowed = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
        require(data["status"] == "complete" and all(r["state"] in allowed for r in rows), "terminal accounting contains unfinished jobs")
        require(all(r["historical_attempt"] or (r["state"] == "COMPLETED" and r["exit_code"] == "0:0") for r in rows), "required campaign job did not finish successfully")
        require(close(total, data["campaign_gpu_hours"]) and close(sum(r["allocated_gpu_hours"] for r in rows if r["historical_attempt"]), data["historical_attempt_gpu_hours"]), "terminal accounting totals differ from jobs")
        require(finite(data["prior_gpu_hours"]) and data["prior_gpu_hours"] >= 0 and close(data["combined_gpu_hours"], data["prior_gpu_hours"] + total), "combined accounting total differs")
        require(expected_summary_sha is not None and data["summary"]["sha256"] == expected_summary_sha, "terminal accounting belongs to a different summary")
        previous = data["prior_accounting"]
        require(digest(previous["path"]) == previous["sha256"] and close(json.loads(Path(previous["path"]).read_text())["combined_gpu_hours"], data["prior_gpu_hours"]), "prior accounting binding differs")
        common.update(updated_at=data["created_at"], historical_attempt_gpu_hours=data["historical_attempt_gpu_hours"], prior_gpu_hours=data["prior_gpu_hours"], combined_gpu_hours=data["combined_gpu_hours"], scope="All submitted campaign allocations, including recorded historical unsuccessful attempts; prior and cumulative totals separately bound.")
    else:
        require(close(total, data["allocated_gpu_hours_so_far"]), "scheduler accounting sum differs")
        common.update(updated_at=data["updated_at"], scope="The supplied controller snapshot only; confirm historical unsuccessful allocation coverage separately.")
    return common


def markdown(summary, summary_path, summary_sha, diagnostics, accounting):
    final = summary["checkpoints"]["128"]["cohorts"]
    methods = ("These are human-authored Codeforces constructive programming tasks from CodeContests+, evaluated with released generated checking infrastructure and explicitly source-reviewed supplementary fixtures. Modes measure accepted output-witness behavior on fixed suites, not algorithm diversity. The cohort comprises eight previously screened training questions and 13 reused development questions, with one paired seed; the evidence is diagnostic, not a new held-out generalization estimate. All 21 fixed questions are retained without result-based filtering in this report. "
               f"See the [adapter handbook]({METHODS_DOCUMENTS[0]}) and [citation and interpretation notes]({METHODS_DOCUMENTS[1]}).")
    lines = ["# Fixed coding128 pilot", "", "Primary result: the prespecified final checkpoint at 128 updates, with one paired training seed. All 8 training and 13 development questions are retained, including zero-success questions.", "", "Final evaluation uses 512 draws per training question and 128 per development question. The complete audited campaign contains 23,040 draws across 13 shards. Base uses the same initialized policy as training.", "", methods, "", "| Cohort | Policy | Accuracy | Pass@8 | ED@8 | ED@32 | Arm-wise PCMD | Eligible questions |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cohort, (_, label, _) in COHORTS.items():
        group = final[cohort]
        for arm in ARMS:
            aggregate = group["arm_wise"][arm]
            lines.append(f"| {label} | {LABELS[arm]} | {fmt(value(group,arm,'accuracy'),percent=True)} | {fmt(value(group,arm,'pass8'),percent=True)} | {fmt(value(group,arm,'ed8'))} | {fmt(value(group,arm,'ed32'))} | {fmt(aggregate['macro_pcmd_over_eligible_tasks'])} | {aggregate['pcmd_eligible_tasks']}/{aggregate['pcmd_total_tasks']} |")
    lines += ["", "Primary Re:Max minus MaxRL comparison:", "", "| Cohort | Accuracy (pp) | Pass@8 (pp) | ED@8 | ED@32 | Common-pair PCMD | Common eligible questions |", "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort, (_, label, _) in COHORTS.items():
        group = final[cohort]; pair = group["common_pair"]
        differences = [value(group,"remax",metric) - value(group,"maxrl",metric) for metric in METRICS[:-1]]
        lines.append(f"| {label} | {fmt(differences[0]*100,signed=True)} | {fmt(differences[1]*100,signed=True)} | {fmt(differences[2],signed=True)} | {fmt(differences[3],signed=True)} | {fmt(pair['remax_minus_maxrl'],signed=True)} | {pair['eligible_tasks']}/{pair['total_tasks']} |")
    lines += ["", "PCMD is defined only with at least 30 accepted answers per question. Arm-wise means can describe different eligible questions; the primary PCMD difference uses the common MaxRL/Re:Max set. Restricting additionally to Base eligibility gives:", "", "| Cohort | Common eligible questions | Base | MaxRL | Re:Max | Re:Max − MaxRL |", "|---|---:|---:|---:|---:|---:|"]
    for cohort, (_, label, _) in COHORTS.items():
        common = final[cohort]["common_all_three"]
        lines.append(f"| {label} | {common['eligible_tasks']}/{common['total_tasks']} | {fmt(common['macro_pcmd']['base'])} | {fmt(common['macro_pcmd']['maxrl'])} | {fmt(common['macro_pcmd']['remax'])} | {fmt(common['remax_minus_maxrl'],signed=True)} |")
    lines += ["", "![Final128 comparison](final128.png)", "", "[Standalone final figure (PDF)](final128.pdf)", "", "Checkpoints 32/64 are descriptive. Their 128/32 draws per training/development question use matching Base prefixes. The final checkpoint budgets are 512/128. These unequal budgets affect estimator precision and PCMD eligibility; common eligible sets may also change across checkpoints.", "", "![Descriptive trajectory](trajectory.png)", "", "[Standalone trajectory figure (PDF)](trajectory.pdf)", "", "Accuracy, pass@8, and expected distinct correct answers are macro averages over every question in the stated cohort. ED@k and pass@k use the auditor’s finite-sample estimators. The figures show no uncertainty intervals: one paired seed does not establish training-seed variability. Development questions remain distinct from the eight training questions; this pilot supplies no reserved-test result."]
    if diagnostics:
        lines += ["", "Training diagnostics (128 updates; 16 fresh samples per update):", "", "| Policy | Fresh accepted /2048 | Mixed groups | All-correct groups | All-wrong groups | Final bank modes |", "|---|---:|---:|---:|---:|---:|"]
        for arm, row in diagnostics.items():
            lines.append(f"| {LABELS[arm]} | {row['accepted']} | {row['mixed_groups']} | {row['all_correct_groups']} | {row['all_wrong_groups']} | {row['final_bank_modes']} |")
        lines += ["", "All supplied training rows satisfy exact-zero old/live score differences and initial clipping diagnostics. These diagnostic inputs are separately hash-bound in render_manifest.json."]
    if accounting:
        if accounting["terminal"]:
            lines += ["", f"The completed campaign used {accounting['allocated_gpu_hours']:.3f} GPU-hours across {accounting['jobs']} allocations, including {accounting['historical_attempt_gpu_hours']:.3f} GPU-hours from historical unsuccessful attempts. Earlier pilots used {accounting['prior_gpu_hours']:.3f} GPU-hours; the cumulative total is {accounting['combined_gpu_hours']:.3f} GPU-hours. All costs use terminal scheduler allocation records."]
        else:
            lines += ["", f"The supplied scheduler snapshot records {accounting['allocated_gpu_hours']:.3f} GPU-hours across {accounting['jobs']} jobs as of {accounting['updated_at']} (controller status: {accounting['status']}). This is snapshot accounting; confirm inclusion of historical unsuccessful allocations before interpreting it as total campaign cost."]
    lines += ["", f"Source: [authoritative audited summary]({Path(summary_path).resolve()}).", "", f"Summary SHA-256: `{summary_sha}`.", "", f"Auditor SHA-256: `{summary['auditor_sha256']}`.", "", "No policy, checkpoint, question, or completion is selected by its outcome in this report.", ""]
    return "\n".join(lines)



def render(summary_path, output, *, maxrl_training=None, remax_training=None, accounting=None, auditor=None):
    summary_path, output = Path(summary_path), Path(output)
    summary = validate_summary(json.loads(summary_path.read_text()))
    auditor = Path(auditor) if auditor else Path(__file__).with_name("audit_real_domains_native_hf_20260922.py")
    require(digest(auditor) == summary["auditor_sha256"], "summary auditor bytes differ from supplied authoritative auditor")
    require(not output.exists(), "report output directory must be new")
    methods_bindings = [{"path":str(path),"sha256":digest(path)} for path in METHODS_DOCUMENTS]
    diagnostics = {arm: training_diagnostics(path, expected_identity_sha256=summary["shards"][0]["parent_training_identity_sha256"][arm], arm=arm) for arm,path in (("maxrl",maxrl_training),("remax",remax_training)) if path}
    summary_sha = digest(summary_path)
    accounting = accounting_snapshot(accounting, expected_summary_sha=summary_sha) if accounting else None
    output.mkdir(parents=True)
    figures(summary, output)
    (output / "report.md").write_text(markdown(summary,summary_path,summary_sha,diagnostics,accounting))
    require(digest(summary_path) == summary_sha, "summary changed during rendering")
    manifest = {"schema":"fixed-code128-report-render-20260922-v1", "status":"complete", "created_at":datetime.now(timezone.utc).isoformat(), "summary_path":str(summary_path.resolve()), "summary_sha256":summary_sha, "auditor_sha256":summary["auditor_sha256"], "renderer_sha256":digest(__file__), "primary_checkpoint":128, "paired_training_seeds":1, "uncertainty_intervals":False, "methods_document_bindings":methods_bindings, "training_diagnostics":diagnostics, "accounting_snapshot":accounting, "artifacts":{p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "render_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return manifest


def self_test():
    """Build explicit synthetic rendering data; discard every plotted artifact."""
    def task_metric(n, correct):
        counts = [correct//2, correct-correct//2]
        def discover(c,k):
            return 1. if n-c < k else 1.-math.prod((n-c-i)/(n-i) for i in range(k))
        return {"samples":n,"accepted":correct,"accuracy":correct/n,"pcmd_accepted_threshold":30,"pcmd_eligible":correct>=30,"pcmd":1.-sum(c*(c-1) for c in counts)/(correct*(correct-1)) if correct>=30 else None,"pass_at_k":{str(k):discover(correct,k) for k in (1,8,32)},"expected_distinct_valid_modes_at_k":{str(k):sum(discover(c,k) for c in counts) for k in (1,8,32)}}
    summary = {"schema":SCHEMA,"status":"pass","kind":"complete_fixed_native_code128_comparison","primary_checkpoint":128,"paired_training_seeds":1,"samples":23040,"shards":[{} for _ in range(13)],"checkpoints":{},"synthetic_renderer_fixture":True,"auditor_sha256":"FIXTURE_NOT_RESULTS"}
    for step in (32,64,128):
        groups={}
        for cohort,(ids,_,_) in COHORTS.items():
            n=(512 if cohort=='train' else 128) if step==128 else (128 if cohort=='train' else 32)
            tasks={a:{t:task_metric(n,0 if i==0 or cohort=='development' else n//(3+ai)) for i,t in enumerate(ids)} for ai,a in enumerate(ARMS)}
            aggregates={}
            for arm,rows in tasks.items():
                eligible=[t for t in ids if rows[t]['pcmd_eligible']]
                aggregates[arm]={'tasks':len(ids),'samples':n*len(ids),'accepted':sum(r['accepted'] for r in rows.values()),'macro_accuracy':sum(r['accuracy'] for r in rows.values())/len(ids),'pcmd_total_tasks':len(ids),'pcmd_eligible_task_ids':eligible,'pcmd_eligible_tasks':len(eligible),'macro_pcmd_over_eligible_tasks':sum(rows[t]['pcmd'] for t in eligible)/len(eligible) if eligible else None,**{out:{str(k):sum(r[key][str(k)] for r in rows.values())/len(ids) for k in (1,8,32)} for out,key in [('macro_pass_at_k','pass_at_k'),('macro_expected_distinct_valid_modes_at_k','expected_distinct_valid_modes_at_k')]}}
            group={'task_ids':ids,'samples_per_task':n,'task_metrics':tasks,'arm_wise':aggregates}
            for key,arms in [('common_pair',('maxrl','remax')),('common_all_three',ARMS)]:
                eligible=[t for t in ids if all(tasks[a][t]['pcmd_eligible'] for a in arms)]
                means={a:sum(tasks[a][t]['pcmd'] for t in eligible)/len(eligible) if eligible else None for a in arms}
                group[key]={'task_ids':eligible,'eligible_tasks':len(eligible),'total_tasks':len(ids),'macro_pcmd':means,'remax_minus_maxrl':means['remax']-means['maxrl'] if eligible else None}
            groups[cohort]=group
        summary['checkpoints'][str(step)]={'role':'primary_final' if step==128 else 'descriptive_trajectory','cohorts':groups}
    validate_summary(summary,fixture=True)
    for mutation in ('synthetic','missing_task','wrong_threshold','zero_success_pcmd'):
        changed=deepcopy(summary)
        if mutation=='synthetic':
            try:validate_summary(changed)
            except ValueError:continue
            raise AssertionError('synthetic results accepted')
        row=changed['checkpoints']['128']['cohorts']['train']['task_metrics']['maxrl']
        if mutation=='missing_task':row.pop(TRAIN_IDS[0])
        elif mutation=='wrong_threshold':row[TRAIN_IDS[0]]['pcmd_accepted_threshold']=2
        else:row[TRAIN_IDS[0]]['pcmd']=0.
        try:validate_summary(changed,fixture=True)
        except ValueError:continue
        raise AssertionError('invalid fixture accepted: '+mutation)
    with tempfile.TemporaryDirectory(prefix='code128_renderer_fixture_NOT_RESULTS_') as temporary:
        directory=Path(temporary);figures(summary,directory,fixture=True)
        require(all((directory/name).stat().st_size>1000 for name in ('final128.png','final128.pdf','trajectory.png','trajectory.pdf')), 'render fixture produced empty artifact')
        previous=directory/'prior_FIXTURE.json';previous.write_text(json.dumps({'combined_gpu_hours':2.}))
        accounting_path=directory/'accounting_FIXTURE.json'
        accounting={'schema':'corrected-code128-terminal-accounting-20260922-v1','status':'complete','created_at':'SYNTHETIC_FIXTURE','campaign_gpu_hours':1.25,'historical_attempt_gpu_hours':.25,'prior_gpu_hours':2.,'combined_gpu_hours':3.25,'summary':{'sha256':'FIXTURE'},'prior_accounting':{'path':str(previous),'sha256':digest(previous)},'jobs':[{'job_id':1,'allocated_gpu_hours':1.,'historical_attempt':False,'state':'COMPLETED','exit_code':'0:0'},{'job_id':2,'allocated_gpu_hours':.25,'historical_attempt':True,'state':'PREEMPTED','exit_code':'0:0'}]}
        accounting_path.write_text(json.dumps(accounting))
        require(accounting_snapshot(accounting_path,expected_summary_sha='FIXTURE')['combined_gpu_hours']==3.25,'terminal accounting fixture failed')
        for mutation in ('wrong_total','wrong_summary','unfinished','required_failed'):
            bad=deepcopy(accounting)
            if mutation=='wrong_total':bad['campaign_gpu_hours']=0.
            elif mutation=='wrong_summary':bad['summary']['sha256']='OTHER_FIXTURE'
            elif mutation=='unfinished':bad['jobs'][0]['state']='RUNNING'
            else:bad['jobs'][0]['state']='FAILED'
            accounting_path.write_text(json.dumps(bad))
            try:accounting_snapshot(accounting_path,expected_summary_sha='FIXTURE')
            except ValueError:continue
            raise AssertionError('invalid accounting accepted: '+mutation)
        # Production v2 emits fresh_successes via float(rewards.sum()); 3.0
        # is the actual wire representation observed in this coding128 run.
        training_dir=directory/'training_FIXTURE';training_dir.mkdir()
        training_identity=training_dir/'identity.json'
        training_identity.write_text(json.dumps({'arm':'maxrl','config':{'updates':128}}))
        metrics_path=training_dir/'metrics.jsonl'
        training_rows=[{'completed_updates':step,'fresh_successes':json.loads('3.0'),'logprobs_diff_min':0.,'logprobs_diff_max':0.,'pg_clipfrac':0.,'bank_modes':2} for step in range(1,129)]
        def save_training():
            metrics_path.write_text(''.join(json.dumps(row)+'\n' for row in training_rows))
        save_training()
        checked=training_diagnostics(training_dir,expected_identity_sha256=digest(training_identity),arm='maxrl')
        require(type(checked['accepted']) is int and checked['accepted']==384 and checked['mixed_groups']==128,'production integral-float training metrics rejected')
        for invalid in (True,3.5,float('nan'),float('inf'),-1.,17.):
            training_rows[0]['fresh_successes']=invalid;save_training()
            try:training_diagnostics(training_dir,expected_identity_sha256=digest(training_identity),arm='maxrl')
            except ValueError:continue
            raise AssertionError('invalid fresh-success count accepted: '+repr(invalid))
        training_rows[0]['fresh_successes']=3;save_training()
        require(training_diagnostics(training_dir,expected_identity_sha256=digest(training_identity),arm='maxrl')['accepted']==384,'integral integer training count rejected')
        text=markdown(summary,'SYNTHETIC_FIXTURE_NOT_RESULTS.json','FIXTURE',{},None)
        require('undefined' in text and '0/13' in text and '1408_A' not in text, 'null eligibility rendering failed')
        require('human-authored Codeforces' in text and 'generated checking infrastructure' in text and 'eight previously screened' in text and '13 reused development' in text and 'not algorithm diversity' in text, 'application interpretation missing')
    return {'status':'pass','scope':'CPU renderer sanity only; synthetic fixtures were watermarked and temporary outputs deleted','checks':['complete fixed comparison','reject synthetic result input','reject missing task','reject changed threshold','reject zero-success PCMD value','render both figures with zero-eligible cohort','render markdown null denominators','terminal and unsuccessful-attempt accounting','reject wrong accounting sum, summary, unfinished or failed required job','accept production integral-float training counts and integer counts','reject boolean, fractional, nonfinite and out-of-range training counts','retain task-source and diagnostic-cohort interpretation']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path);parser.add_argument('--output','--output-dir',dest='output',type=Path)
    parser.add_argument('--maxrl-training',type=Path);parser.add_argument('--remax-training',type=Path)
    parser.add_argument('--accounting',type=Path,help='Optional terminal_accounting.json or campaign_status.json')
    parser.add_argument('--auditor',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        require(args.summary is None and args.output is None,'self-test cannot write a result report');result=self_test()
    else:
        parser.error('--summary and --output are required') if args.summary is None or args.output is None else None
        result=render(args.summary,args.output,maxrl_training=args.maxrl_training,remax_training=args.remax_training,accounting=args.accounting,auditor=args.auditor)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
