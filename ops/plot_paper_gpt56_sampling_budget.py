#!/usr/bin/env python3
"""Render the main-paper GPT-5.6 Sol sampling-budget figure from frozen n=64 data.

The figure rarefies each complete observed pool without replacement. It neither
extrapolates beyond 64 draws nor treats certified support lower bounds as totals.
Raw grades, response identities, receipt hashes, cohorts, and retained means are
checked before plotting; pointwise intervals come from the frozen analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/results/modebench_discovery_curves_20260911.json"
OUTPUT = ROOT / "paper/figures/gpt56_sampling_budget"
DOMAINS = ("python_factors", "mathir", "pantry_plan")
LABELS = ("Python factors", "MathIR", "PantryPlan")
LEVELS = (2, 3)
ARMS = ("original", "neutral")
K_GRID = (1, 2, 4, 8, 16, 32, 64)
GRADING = "normalized_secondary"
FIGSIZE = (6.4, 2.0)
COLORS = {"original": "#00509E", "neutral": "#C76A3A"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def binding(path):
    return {"path": relative(path), "sha256": file_sha(path)}


def bound_file(record):
    path = resolve(record["path"])
    require(file_sha(path) == record["sha256"], f"Changed source binding: {path}")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity(record):
    return record["level"], record["domain"], record["row_index"]


def sample_identity(record):
    return (*identity(record), record["sample_index"])


def unique(records, key, label):
    result = {key(record): record for record in records}
    require(len(result) == len(records), f"Duplicate {label}")
    return result


def rarefaction(mode_counts, responses, k):
    """Expected distinct verified modes and pass@k in a uniform k-subset."""
    require(type(responses) is int and responses > 0 and type(k) is int
            and 0 <= k <= responses, "Invalid finite-pool sampling budget")
    require(all(type(count) is int and count > 0 for count in mode_counts)
            and sum(mode_counts) <= responses, "Invalid verified mode counts")
    denominator = math.comb(responses, k)

    def seen(count):
        return 1.0 - math.comb(responses - count, k) / denominator

    return {"distinct": math.fsum(seen(count) for count in mode_counts),
            "pass": seen(sum(mode_counts))}


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10),
            f"Metric differs from complete-pool reconstruction: {label}")


def validate_prompt(prompt):
    require(prompt["responses"] == 64, "Every selected prompt must retain all 64 responses")
    counts = prompt["mode_counts"]
    require(sum(counts) == prompt["correct_draws"]
            and prompt["correct_draws"] + prompt["failed_draws"] == 64,
            "Mode counts do not sum to the complete correctness inventory")
    require(set(prompt["rarefaction"]) == {str(k) for k in K_GRID},
            "Incomplete retained rarefaction grid")
    support = prompt["support_reference"]
    require(support["support_kind"] == "certified_lower_bound"
            and type(support["support_count"]) is int and support["support_count"] > 0,
            "Support must be a positive certified lower bound, not an exact total")
    points = {}
    for k in K_GRID:
        points[k] = rarefaction(counts, 64, k)
        for metric, value in points[k].items():
            close(prompt["rarefaction"][str(k)][metric], value,
                  f"{identity(prompt)}/{metric}/k{k}")
    return points


def summarize_cell(prompts, cell, arm):
    require(len(prompts) == 16 and len({identity(p) for p in prompts}) == 16,
            "Every cell must retain all 16 distinct registered prompts")
    reconstructed = [validate_prompt(prompt) for prompt in prompts]
    counts = cell["counts"][arm]
    require(counts["prompts"] == 16 and counts["responses"] == 1024
            and counts["correct_draws"] == sum(p["correct_draws"] for p in prompts)
            and counts["failed_draws"] == sum(p["failed_draws"] for p in prompts)
            and counts["support_kinds"] == {"certified_lower_bound": 16},
            "Cell counts differ from the complete registered cohort")
    points = []
    for k in K_GRID:
        metrics = {}
        for metric in ("distinct", "pass"):
            retained = cell[arm][f"rarefaction/{metric}/k{k}"]
            expected = math.fsum(p[k][metric] for p in reconstructed) / 16
            close(retained["estimate"], expected, f"{arm}/{metric}/k{k}")
            ci = retained.get("ci95")
            require(isinstance(ci, list) and len(ci) == 2
                    and all(type(v) in (int, float) and math.isfinite(v) for v in ci)
                    and -1e-10 <= ci[0] <= retained["estimate"] + 1e-10
                    and retained["estimate"] - 1e-10 <= ci[1]
                    and ci[1] <= (k if metric == "distinct" else 1) + 1e-10
                    and retained.get("defined_bootstrap_replicates") == 20000,
                    f"Invalid retained pointwise interval: {arm}/{metric}/k{k}")
            metrics[metric] = deepcopy(retained)
        points.append({"k": k, **metrics})
    support_counts = [p["support_reference"]["support_count"] for p in prompts]
    return {"arm": arm, "counts": deepcopy(counts), "points": points,
            "prompt_ids": [p["support_reference"]["pair_id"] for p in prompts],
            "support_lower_bound": {"mean": math.fsum(support_counts) / 16,
                                    "range": [min(support_counts), max(support_counts)],
                                    "counts": support_counts,
                                    "kind": "certified_lower_bound"},
            "tail_gain_32_to_64": points[-1]["distinct"]["estimate"] - points[-2]["distinct"]["estimate"],
            "endpoint_pass64": deepcopy(points[-1]["pass"])}


def authenticate_condition(condition, arm, selected):
    sources = condition["sources"]
    paths = {name: bound_file(record) for name, record in sources.items()}
    directory = resolve(condition["directory"])
    audit = read_json(paths["completion_audit.json"])
    evidence = read_json(paths["evidence_file_sha256.json"])
    require(audit.get("status") == "pass" and audit.get("expected_responses") == 6144
            and audit.get("saved_samples") == 6144
            and audit.get("evidence_inventory_sha256") == object_sha(evidence),
            f"Incomplete or stale native completion audit: {arm}")
    require(all((directory / name).resolve().is_relative_to(directory.resolve()) for name in evidence),
            "Evidence inventory escapes its run directory")

    def check_evidence(item):
        name, digest = item
        data = (directory / name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest,
                f"Changed source binding: {directory / name}")
        return name, object_sha(json.loads(data)) if name.startswith("raw_responses/") else None

    # Native receipts are small independent immutable files on shared storage.
    with ThreadPoolExecutor(max_workers=16) as pool:
        receipt_hashes = dict(pool.map(check_evidence, evidence.items()))
    samples = unique(read_jsonl(paths["samples.jsonl"]), sample_identity, "raw sample slot")
    grades = unique(read_jsonl(paths["discovery_hosted_grades.jsonl"]), sample_identity, "grade slot")
    expected = {(*key, index) for key in selected for index in range(64)}
    require(set(samples) == set(grades) == expected,
            f"Raw grades and samples must cover every registered prompt and all 64 slots: {arm}")
    mode_counts = defaultdict(Counter)
    settings = Counter()
    for key, sample in samples.items():
        grade = grades[key]
        require(grade["raw_sample_sha256"] == object_sha(sample), "Grade belongs to another response")
        require(sample["row_sha256"] == selected[key[:3]]["row_sha256"]
                and sample["prompt_arm"] == arm and sample["model"] == "gpt-5.6-sol"
                and sample.get("fresh_response_cohort") is True,
                "Raw response differs from registered problem, model, or arm")
        require(receipt_hashes.get(sample["raw_receipt"]) == sample["raw_receipt_sha256"],
                "Raw response receipt binding differs from authenticated inventory")
        requested = sample["requested_settings"]
        require(requested.get("reasoning", {}).get("effort") == "medium"
                and requested.get("max_output_tokens") == 8192
                and "temperature" not in requested and "top_p" not in requested,
                "Generation controls differ from the medium-reasoning discovery protocol")
        normal = grade["normalization"]
        require(type(normal["verified"]) is bool and normal["original_text"] == sample["text"],
                "Invalid normalized grade or changed original response")
        require(not grade["strict"]["verified"] or
                (normal["verified"] and normal["canonical_key"] == grade["strict"]["canonical_key"]),
                "Formatting normalization changed a strict success")
        if normal["verified"]:
            require(isinstance(normal["canonical_key"], str) and normal["canonical_key"],
                    "A verified response requires a canonical mode key")
            mode_counts[key[:3]][normal["canonical_key"]] += 1
        settings[json.dumps({"requested": sample["requested_settings"],
                             "returned": sample["returned_settings"]}, sort_keys=True)] += 1
    return mode_counts, {"sources": deepcopy(sources), "authenticated_evidence_files": len(evidence),
                         "authenticated_native_receipts": len({s["raw_receipt"] for s in samples.values()}),
                         "generation_settings": [{**json.loads(key), "responses": count}
                                                 for key, count in sorted(settings.items())],
                         "served_model_snapshots": audit["served_model_snapshots_by_logical_sample"]}


def build_record(source=SOURCE):
    source = Path(source)
    report = read_json(source)
    require(report.get("schema") == "modebench-discovery-curves-analysis-v1"
            and report.get("status") == report.get("experiment_status") == "complete"
            and report["inventory"]["status"] == "complete",
            "Use the complete active publication analysis, not a preview or partial campaign")
    publication = resolve(report["publication_source"]["report_path"])
    require(source.read_bytes() == publication.read_bytes(),
            "Paper result is not the exact active publication report")
    artifact_manifest = publication.parent / "artifact_manifest.json"
    manifest = read_json(artifact_manifest)
    require(manifest["status"] == "complete"
            and bound_file(manifest["outputs"]["analysis.json"]).resolve() == publication.resolve(),
            "Publication manifest does not bind this complete report")
    records = [*report["design"].values(), report["analyzer_source"],
               *report["analysis_dependencies"].values(), report["registries"]["hosted"],
               report["prospective_analysis_plan"], *report["prospective_amendments"].values()]
    for record in records:
        bound_file(record)
    protocol = report["protocol"]
    require(protocol["k_grid"] == list(K_GRID) and protocol["sample_count"] == 64
            and protocol["bootstrap"]["replicates"] == 20000
            and protocol["bootstrap"]["pointwise"] is True,
            "Sampling grid or pointwise bootstrap differs from the registered design")
    selection = read_json(resolve(report["design"]["selection.json"]["path"]))
    selected = unique(selection["selected"], identity, "registered prompt")
    require(len(selected) == 96 and selection["prompts_per_cell"] == 16,
            "The complete figure cohort requires all 96 selected prompts")
    support = read_json(resolve(report["design"]["support_reference.json"]["path"]))["references"]
    models = [m for m in report["models"] if m["model_id"] == "gpt56sol"]
    require(len(models) == 1 and models[0]["family"] == "frontier", "Ambiguous GPT-5.6 Sol model")
    model = models[0]
    require(set(model["conditions"]) == set(ARMS), "Both registered prompt wordings are required")
    analysis = model["analyses"][GRADING]
    expected_cells = {f"level{level}/{domain}" for level in LEVELS for domain in DOMAINS}
    require(set(analysis["cells"]) == expected_cells and set(analysis["prompts"]) == set(ARMS),
            "Incomplete domain-level or prompt-wording grid")
    panels = {key: {"cell": key, "curves": {}} for key in expected_cells}
    conditions = {}
    for arm in ARMS:
        raw_counts, conditions[arm] = authenticate_condition(model["conditions"][arm], arm, selected)
        prompts = unique(analysis["prompts"][arm], identity, f"{arm} retained prompt")
        require(set(prompts) == set(selected), "Retained analysis changes the frozen prompt cohort")
        for key, prompt in prompts.items():
            require(prompt["row_sha256"] == selected[key]["row_sha256"]
                    and prompt["support_reference"] == support[selected[key]["pair_id"]],
                    "Retained prompt or support certificate differs from frozen design")
            require(prompt["mode_counts"] == sorted(raw_counts[key].values(), reverse=True),
                    "Retained mode counts differ from authenticated raw normalized grades")
        for level in LEVELS:
            for domain in DOMAINS:
                key = f"level{level}/{domain}"
                cohort = sorted((p for p in prompts.values() if identity(p)[:2] == (level, domain)),
                                key=lambda p: p["support_reference"]["rank_in_cell"])
                panels[key]["curves"][arm] = summarize_cell(cohort, analysis["cells"][key], arm)
    ordered_panels = []
    for level in LEVELS:
        for domain in DOMAINS:
            panel = panels[f"level{level}/{domain}"]
            original, neutral = (panel["curves"][arm] for arm in ARMS)
            require(original["prompt_ids"] == neutral["prompt_ids"]
                    and original["support_lower_bound"] == neutral["support_lower_bound"],
                    "Prompt wordings must retain identical paired problems and support references")
            panel.update(level=level, domain=domain,
                         support_lower_bound=deepcopy(original["support_lower_bound"]))
            ordered_panels.append(panel)
    return {"schema": "paper-gpt56-sampling-budget-v1", "status": "complete",
            "source": binding(source), "publication_report": binding(publication),
            "publication_manifest": binding(artifact_manifest), "renderer": binding(__file__),
            "model": "gpt-5.6-sol", "grading": GRADING,
            "design_sources": deepcopy(report["design"]),
            "analysis_sources": records[len(report["design"]):], "conditions": conditions,
            "sampling": {"domains": list(DOMAINS), "levels": list(LEVELS), "arms": list(ARMS),
                         "k_grid": list(K_GRID), "prompts_per_cell": 16, "responses_per_prompt": 64,
                         "total_prompts_per_arm": 96, "total_responses": 12288,
                         "selection_rule": selection["rule"], "all_selected_prompts_retained": True},
            "bootstrap": deepcopy(protocol["bootstrap"]),
            "display": {"x": "Samples per prompt, k", "y": "Mean distinct verified modes",
                        "figure_inches": list(FIGSIZE), "x_scale": "log2",
                        "y_scale": "linear; shared within domain; includes certified support reference",
                        "estimator": "Exact rarefaction without replacement from each complete 64-response pool, then equal mean over all 16 prompts.",
                        "support_reference": "Mean certified lower bound on existing modes; not an exhaustive count or exact coverage denominator.",
                        "intervals": "Retained pointwise 95% paired whole-prompt bootstrap intervals; no simultaneous coverage claim."},
            "panels": ordered_panels,
            "limitations": ["No sampling or extrapolation beyond 64 responses.",
                            "Flattening within an observed finite pool does not establish asymptotic saturation.",
                            "All unsuccessful draws and zero-success prompts remain in the denominator.",
                            "Certified mode counts are lower bounds; unenumerated modes may also exist.",
                            "The strict analysis remains in the discovery-curve appendix."],
            "validation": {"publication_copy_exact": True, "source_bindings_authenticated": True,
                           "all_native_evidence_hashes_authenticated": True,
                           "raw_grade_and_response_slots_complete": True,
                           "canonical_counts_reconstructed": True, "prompt_and_cell_means_reconstructed": True,
                           "api_calls": 0}}


def build_figure(record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, StrMethodFormatter

    rc = {"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
          "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
          "ps.fonttype": 42, "text.color": "#19324A", "axes.labelcolor": "#19324A"}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True)
        fig.subplots_adjust(left=.09, right=.985, bottom=.24, top=.74, wspace=.26)
        for col, domain in enumerate(DOMAINS):
            ax = axes[col]
            domain_panels = [p for p in record["panels"] if p["domain"] == domain]
            for level in LEVELS:
                panel = next(p for p in domain_panels if p["level"] == level)
                for arm in ARMS:
                    points = panel["curves"][arm]["points"]
                    xs = [p["k"] for p in points]
                    ys = [p["distinct"]["estimate"] for p in points]
                    ax.fill_between(xs, [p["distinct"]["ci95"][0] for p in points],
                                    [p["distinct"]["ci95"][1] for p in points],
                                    color=COLORS[arm], alpha=.09, linewidth=0)
                    ax.plot(xs, ys, color=COLORS[arm], marker="o" if level == 2 else "s",
                            linestyle="-" if level == 2 else "--",
                            markerfacecolor=COLORS[arm] if level == 2 else "white",
                            markersize=2.5, linewidth=1.2, zorder=3)
                lower = panel["support_lower_bound"]["mean"]
                ax.axhline(lower, color="#71808C", linestyle=(0, (3, 2)), linewidth=.9, zorder=1)
            support_means = sorted({p["support_lower_bound"]["mean"] for p in domain_panels})
            number = f"{support_means[0]:g}" if len(support_means) == 1 else f"{support_means[0]:.2f}–{support_means[-1]:g}"
            label = f"At least {number} modes (mean)" if domain == "pantry_plan" else f"At least {number} known modes"
            ax.annotate(label, (1.05, support_means[-1]), xytext=(0, 3), textcoords="offset points",
                        fontsize=6.4, color="#566574", ha="left", va="bottom")
            maximum = max(support_means[-1], max(q["distinct"]["ci95"][1] for p in domain_panels
                          for curve in p["curves"].values() for q in curve["points"]))
            ax.set_ylim(0, maximum * 1.21)
            ax.set_xlim(.95, 72)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(K_GRID))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.set_yticks((0, 1, 2) if domain == "python_factors" else (0, 2, 4, 6) if domain == "mathir" else (0, 5, 10, 15, 20))
            ax.tick_params(length=2.5, width=.6, colors="#607487")
            ax.grid(axis="y", color="#D8E2EA", linewidth=.5, alpha=.65)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#AABAC7")
                ax.spines[side].set_linewidth(.6)
            ax.set_title(LABELS[col], fontsize=8.7, fontweight="bold", pad=7)
        handles = [Line2D([], [], color=COLORS[arm], linewidth=1.3, label=arm.capitalize()) for arm in ARMS]
        handles += [Line2D([], [], color="#526678", linestyle="-" if level == 2 else "--",
                          marker="o" if level == 2 else "s", markersize=2.5,
                          markerfacecolor="#526678" if level == 2 else "white", linewidth=1.2,
                          label=f"L{level}") for level in LEVELS]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.996, 1.035),
                   ncol=4, frameon=False, fontsize=7.2, handlelength=1.8,
                   handletextpad=.45, columnspacing=1.)
        fig.text(.09, .976, "GPT-5.6 Sol (medium)", ha="left", va="top", fontsize=8, fontweight="bold")
        fig.text(.025, .49, "Mean distinct\nverified modes", rotation=90, ha="center", va="center", fontsize=8)
        fig.text(.55, .05, "Samples per prompt, $k$", ha="center", va="center", fontsize=8)
    return fig


def render(record, output=OUTPUT):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(record)
    result = deepcopy(record)
    result["outputs"] = {}
    for extension in ("pdf", "png"):
        path = output.with_suffix("." + extension)
        kwargs = {"dpi": 260} if extension == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}
        figure.savefig(path, facecolor="white", **kwargs)
        result["outputs"][extension] = binding(path)
    plt.close(figure)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def check(output=OUTPUT, source=SOURCE):
    output = Path(output)
    record = build_record(source)
    retained = read_json(output.with_suffix(".json"))
    outputs = retained.pop("outputs")
    require(retained == record, "Figure metadata is stale or differs from complete-pool reconstruction")
    require(set(outputs) == {"pdf", "png"}, "Incomplete rendered figure outputs")
    for extension, item in outputs.items():
        require(bound_file(item).resolve() == output.with_suffix("." + extension).resolve(),
                "Rendered output path differs from figure destination")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Authenticate retained figure metadata and output hashes")
    args = parser.parse_args()
    if args.check:
        record = check(args.output, args.source)
    else:
        record = build_record(args.source)
        render(record, args.output)
    print(json.dumps({"status": "pass", "mode": "check" if args.check else "render",
                      "output": relative(args.output), "panels": len(record["panels"]),
                      "responses": record["sampling"]["total_responses"], "api_calls": 0}))


if __name__ == "__main__":
    main()
