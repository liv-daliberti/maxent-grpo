#!/usr/bin/env python3
"""Render all E118/E119/E120 paired endpoints from a dated, audited snapshot.

This is a presentation-only builder: no live run access, endpoint selection,
imputation, recomputed uncertainty, or changes to the historical E120 analysis.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"qwen05b": "Qwen-0.5B", "falcon1b": "Falcon-1B", "qwen3b": "Qwen-3B"}
DOMAINS = {"graph_coloring": "Graph", "countdown": "Countdown", "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "PantryPlan"}
CAMPAIGNS = {"e118": "E118: MaxRL factorial", "e119": "E119: Level-2 factorial", "e120": "E120-R1: fresh-frequency replay ablation"}
CONTRASTS = {
    "replay_maxrl_minus_maxrl": ("Re:MaxRL", "MaxRL", "MaxRL"),
    "replay_drgrpo_minus_drgrpo": ("Re:Dr.GRPO", "Dr.GRPO", "Dr.GRPO"),
    "uniform_minus_frequency": ("Uniform", "Frequency", "Uniform"),
}
METRICS = ("pass8", "distinct8", "breadth8")
INTERVALS = ("student_t_95", "paired_bootstrap_percentile_95")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def close(a: float, b: float, context: str) -> None:
    if not math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError(f"inconsistent {context}: {a} != {b}")


def validate_contrast(value: dict[str, Any], context: str) -> None:
    n = value["n"]
    if not 0 <= n <= 5 or len(set(value["paired_seeds"])) != n:
        raise ValueError(f"invalid pair count: {context}")
    if value["complete_five_seed_block"] != (n == 5):
        raise ValueError(f"invalid completion flag: {context}")
    for metric in METRICS:
        s = value["summaries"][metric]
        if s["n"] != n or set(s["per_seed"]) != set(map(str, value["paired_seeds"])):
            raise ValueError(f"inconsistent seed set: {context}/{metric}")
        intervals = [key for key in INTERVALS if key in s]
        if n < 5 and intervals:
            raise ValueError(f"partial block has an interval: {context}/{metric}")
        if n == 5 and len(intervals) != 1:
            raise ValueError(f"complete block lacks one stored interval: {context}/{metric}")
        if n:
            close(s["mean"], statistics.fmean(s["per_seed"].values()), context)
            close(s["mean"], s["left_mean"] - s["right_mean"], context)
            for key in intervals:
                lo, hi = s[key]
                if not all(map(math.isfinite, (lo, hi))) or lo > hi:
                    raise ValueError(f"invalid interval: {context}/{metric}")
    if n:
        p, d, b = (value["summaries"][m] for m in METRICS)
        for key in ("mean", "left_mean", "right_mean"):
            close(b[key], d[key] - p[key], context + "/breadth")
        for seed in map(str, value["paired_seeds"]):
            close(b["per_seed"][seed], d["per_seed"][seed] - p["per_seed"][seed], context + "/seed breadth")


def prepare(snapshot: dict[str, Any], source: Path) -> dict[str, Any]:
    rows, factorial_rows, coverage = [], [], []
    for campaign, label in CAMPAIGNS.items():
        c = snapshot["campaigns"][campaign]
        paired = 0
        for block in sorted(c["blocks"], key=lambda b: (list(MODELS).index(b["model_key"]), list(DOMAINS).index(b["domain"]))):
            for name, contrast in sorted(block["contrasts"].items()):
                validate_contrast(contrast, "/".join((campaign, block["model_key"], block["domain"], name)))
                is_factorial = name.startswith("four_arm_")
                primary_name = name.removeprefix("four_arm_")
                left, right, short = CONTRASTS[primary_name]
                row = {
                    "campaign": campaign, "model_key": block["model_key"], "model": MODELS[block["model_key"]],
                    "domain_key": block["domain"], "domain": DOMAINS[block["domain"]],
                    "contrast": name, "left_arm": left, "right_arm": right, "short_contrast": short,
                    "n": contrast["n"], "planned_pairs": 5, "paired_seeds": contrast["paired_seeds"],
                    "complete_five_seed_block": contrast["complete_five_seed_block"],
                    "common_factorial_seeds": block["paired_seeds"],
                    "summaries": {m: contrast["summaries"][m] for m in METRICS},
                }
                if campaign == "e120":
                    row.update({k: block[k] for k in ("mechanism_validated_seeds", "mechanism_validated_complete_block", "mechanism_status")})
                    if not set(row["mechanism_validated_seeds"]) <= set(row["paired_seeds"]):
                        raise ValueError("mechanism eligibility includes an unpaired seed")
                (factorial_rows if is_factorial else rows).append(row)
                if not is_factorial:
                    paired += row["n"]
        coverage.append({
            "campaign": campaign, "label": label,
            **{key: c[key] for key in ("registered_cells", "completion_receipts", "admitted_terminal_endpoints", "endpoint_status_counts", "registered_blocks", "complete_five_seed_blocks")},
            "paired_contrasts": paired,
            "planned_paired_contrasts": c["registered_blocks"] * (10 if campaign == "e119" else 5),
            "complete_pairwise_contrasts": sum(r["n"] == 5 for r in rows if r["campaign"] == campaign),
            **({"mechanism_validated_complete_blocks": c["mechanism_validated_complete_blocks"]} if campaign == "e120" else {}),
        })
    return {
        "schema": "paper-current-campaign-paired-results-v1", "analysis_date": snapshot["analysis_date"],
        "target_step": snapshot["target_step"], "source_snapshot": {"path": relative(source), "sha256": sha256(source)},
        "collection_started_at_utc": snapshot["collection_started_at_utc"],
        "collection_finished_at_utc": snapshot["collection_finished_at_utc"],
        "scope": snapshot["scope"], "metric_definitions": {
            "pass8": "P8 = pass@8: probability of at least one correct answer among eight samples",
            "distinct8": "D8 = distinct@8: expected number of distinct correct semantic modes among eight samples",
            "breadth8": "B8 = D8 - P8: expected number of distinct correct modes beyond the first",
        },
        "contrast_direction": "left_arm minus right_arm; absolute means use exactly the same paired seeds",
        "uncertainty": "Stored paired 95% intervals only for complete n=5 blocks: unadjusted Student-t for E118/E119; exhaustive 5^5 percentile bootstrap for E120. No intervals for incomplete blocks.",
        "coverage": coverage, "rows": rows, "e119_common_four_arm_rows": factorial_rows,
    }


def interval(summary: dict[str, Any]) -> tuple[str | None, list[float] | None]:
    for key in INTERVALS:
        if key in summary:
            return key, summary[key]
    return None, None


def effect(row: dict[str, Any], metric: str, *, tex: bool = False, with_interval: bool = True) -> str:
    if not row["n"]:
        return "---" if tex else "—"
    s = row["summaries"][metric]
    result = f"{s['mean']:+.3f}"
    _, bounds = interval(s)
    if bounds is not None and with_interval:
        result += (r"\;" if tex else " ") + f"[{bounds[0]:+.3f}, {bounds[1]:+.3f}]"
    return f"${result}$" if tex else result


def lookup(data: dict[str, Any], campaign: str, model: str, domain: str, contrast: str) -> dict[str, Any] | None:
    return next((r for r in data["rows"] if (r["campaign"], r["model_key"], r["domain_key"], r["contrast"]) == (campaign, model, domain, contrast) and r["n"]), None)


def findings(data: dict[str, Any]) -> list[str]:
    result = []
    full = [r for r in data["rows"] if r["campaign"] == "e118" and r["n"] == 5]
    if full:
        positives = {m: sum(r["summaries"][m]["mean"] > 0 for r in full) for m in METRICS}
        result.append(f"E118 has {len(full)} complete five-seed model/domain comparisons. Re:MaxRL has positive point estimates in {positives['pass8']}/{len(full)} for pass@8, {positives['distinct8']}/{len(full)} for distinct@8, and {positives['breadth8']}/{len(full)} for B8. These are directional counts, not a pooled effect or a significance test.")
    for model, domain in (("qwen05b", "graph_coloring"), ("qwen05b", "countdown"), ("qwen3b", "python_factors"), ("qwen3b", "graph_coloring"), ("qwen3b", "pantry_plan"), ("falcon1b", "graph_coloring")):
        row = lookup(data, "e118", model, domain, "replay_maxrl_minus_maxrl")
        if row:
            result.append(f"E118 {row['model']} {row['domain']} ({row['n']}/5 pairs): ΔP8 {effect(row, 'pass8')}; ΔB8 {effect(row, 'breadth8')}.")
    for domain in ("graph_coloring", "countdown", "python_factors", "mathir"):
        selected = [lookup(data, "e119", "qwen05b", domain, contrast) for contrast in ("replay_drgrpo_minus_drgrpo", "replay_maxrl_minus_maxrl")]
        for row in (r for r in selected if r):
            result.append(f"E119 Level-2 {row['domain']}, {row['left_arm']} minus {row['right_arm']} ({row['n']}/5 pairs): ΔP8 {effect(row, 'pass8')}; ΔB8 {effect(row, 'breadth8')}.")
    for model, domain in (("qwen05b", "graph_coloring"), ("qwen05b", "countdown"), ("qwen05b", "pantry_plan"), ("falcon1b", "graph_coloring"), ("falcon1b", "pantry_plan"), ("qwen3b", "graph_coloring")):
        row = lookup(data, "e120", model, domain, "uniform_minus_frequency")
        if row:
            eligible = len(row["mechanism_validated_seeds"])
            result.append(f"E120 uniform minus frequency, {row['model']} {row['domain']} ({row['n']}/5 pairs; mechanism eligibility {eligible}/{row['n']} available pairs): ΔP8 {effect(row, 'pass8')}; ΔB8 {effect(row, 'breadth8')}.")
    return result


def render_markdown(data: dict[str, Any]) -> str:
    lines = [f"# Current campaign results — {data['analysis_date']}", "",
             "All available audited, exactly step-3072 paired endpoints from E118, E119 and E120-R1. Completed blocks are usable paper results; unfinished blocks retain their observed n/5 and do not stand in for five seeds.", "",
             f"Snapshot collection: {data['collection_started_at_utc']} to {data['collection_finished_at_utc']}.", "",
             "## Findings", ""]
    lines.extend("- " + finding for finding in data["findings"])
    lines.extend(["", "## Coverage", "", "| Campaign | Training receipts | Admitted endpoints | Complete factorial blocks | Matched seed contrasts |", "|---|---:|---:|---:|---:|"])
    for c in data["coverage"]:
        lines.append(f"| {c['campaign'].upper()} | {c['completion_receipts']}/{c['registered_cells']} | {c['admitted_terminal_endpoints']}/{c['registered_cells']} | {c['complete_five_seed_blocks']}/{c['registered_blocks']} | {c['paired_contrasts']}/{c['planned_paired_contrasts']} |")
    lines.extend(["", "E118 blocks require both MaxRL arms; E119 blocks require all four arms. E119 matched contrasts sum the two pairwise comparisons, each using its own available seed intersection. E120 receipts/endpoints count the 45 frequency treatment runs; uniform comparator endpoints are reused from the corresponding baseline campaign. Its complete blocks require both arms.", "",
                  "## Reading the tables", "",
                  "P8 is pass@8, D8 is distinct@8, and B8 = D8 − P8 counts correct modes beyond the first. Pass@8 is a probability (multiply its differences by 100 for percentage points); D8 and B8 are counts of modes per prompt. Absolute means and differences use the same paired seeds. Effects are replay minus its base optimizer for E118/E119, and uniform minus frequency replay for E120.", "",
                  "Brackets are stored descriptive paired 95% intervals: unadjusted Student-t for complete E118/E119 blocks and exhaustive 5^5 paired percentile bootstrap for complete E120 blocks. Partial blocks have no intervals. Intervals are not adjusted for the many comparisons; broad intervals and intervals spanning zero preclude strong directional claims. No missing run is imputed and no cross-domain/model estimate is pooled."])
    for campaign, label in CAMPAIGNS.items():
        rows = [r for r in data["rows"] if r["campaign"] == campaign]
        lines.extend(["", f"## {label}", "", "| Model | Domain | Difference | n/5 | ΔP8 [95%] | ΔD8 [95%] | ΔB8 [95%] |", "|---|---|---|---:|---:|---:|---:|"])
        for row in rows:
            lines.append("| " + " | ".join([row["model"], row["domain"], row["left_arm"] + " − " + row["right_arm"], f"{row['n']}/5"] + [effect(row, m) for m in METRICS]) + " |")
        lines.extend(["", "Absolute paired means, shown **base → replay** (E120: **frequency → uniform**):", "", "| Model | Domain | Left arm | n/5 | P8 | D8 | B8 |", "|---|---|---|---:|---:|---:|---:|"])
        for row in rows:
            means = [f"{row['summaries'][m]['right_mean']:.3f} → {row['summaries'][m]['left_mean']:.3f}" if row["n"] else "—" for m in METRICS]
            lines.append("| " + " | ".join([row["model"], row["domain"], row["left_arm"], f"{row['n']}/5"] + means) + " |")
        if campaign == "e119":
            lines.extend(["", "Common four-arm seed coverage (for strict factorial comparisons): " + "; ".join(f"{r['domain']}: {len(r['common_factorial_seeds'])}/5 ({', '.join(map(str, r['common_factorial_seeds'])) or 'none'})" for r in rows if r["contrast"] == "replay_maxrl_minus_maxrl") + ". The JSON also preserves both contrasts restricted to those common seeds."])
        if campaign == "e120":
            lines.extend(["", "Mechanism eligibility is separate from endpoint availability. A seed is eligible only when the stored telemetry audit passes and both arms have completion receipts. Full-block mechanism interpretation requires all five eligible paired seeds.", "", "| Model | Domain | Endpoint pairs | Eligible pairs | Full mechanism block |", "|---|---|---:|---:|---|"])
            for row in rows:
                lines.append(f"| {row['model']} | {row['domain']} | {row['n']}/5 | {len(row['mechanism_validated_seeds'])}/5 | {'Yes' if row['mechanism_validated_complete_block'] else 'No'} |")
            lines.extend(["", "A partial effect remains descriptive even when every currently available pair passes telemetry. Opposite-signed domain effects must remain visible; this ablation does not establish a universal advantage for uniform replay. The historical frozen E120 primary analysis is unchanged."])
    lines.extend(["", "## Reproduction", "", f"Source: `{data['source_snapshot']['path']}` (SHA-256 `{data['source_snapshot']['sha256']}`).", "", "```bash", f"python ops/exp_scaling/build_paper_current_campaign_results.py --snapshot {data['source_snapshot']['path']} --figures", "```", "", "The CSV and JSON retain full numerical precision, exact paired seed sets, and every stored metric interval. Three campaign TeX table bodies show all planned model/domain rows (including 0/5), effect estimates, and B8 intervals; the coverage table body reports endpoint and pairing progress.", ""])
    return "\n".join(lines)


def render_csv(data: dict[str, Any]) -> str:
    fields = ["campaign", "model", "domain", "contrast", "left_arm", "right_arm", "n", "planned_pairs", "paired_seeds", "common_factorial_seeds", "mechanism_validated_seeds", "mechanism_validated_complete_block"]
    fields += [m + "_" + suffix for m in METRICS for suffix in ("left_mean", "right_mean", "effect", "lower95", "upper95", "interval_method")]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in data["rows"]:
        flat = {key: row.get(key, "") for key in fields[:12]}
        for key in ("paired_seeds", "common_factorial_seeds", "mechanism_validated_seeds"):
            flat[key] = ";".join(map(str, row[key])) if key in row else ""
        for m in METRICS:
            s = row["summaries"][m]
            method, bounds = interval(s)
            flat.update({m + "_left_mean": s.get("left_mean", ""), m + "_right_mean": s.get("right_mean", ""), m + "_effect": s.get("mean", ""), m + "_lower95": bounds[0] if bounds else "", m + "_upper95": bounds[1] if bounds else "", m + "_interval_method": method or ""})
        writer.writerow(flat)
    return stream.getvalue()


def render_tex(data: dict[str, Any], campaign: str) -> str:
    lines = ["% Model & Domain & Contrast & n/5 & Delta P8 & Delta D8 & Delta B8 [95%]", "% E118/E119: replay minus base. E120: uniform minus frequency.", "% Intervals are stored, descriptive, and only present at n=5.", "% E120 dagger: at least one available endpoint pair lacks mechanism eligibility."]
    for row in (r for r in data["rows"] if r["campaign"] == campaign):
        n = f"{row['n']}/5"
        if campaign == "e120" and row["n"] > len(row["mechanism_validated_seeds"]):
            n += r"$^{\dagger}$"
        contrast = "Re:MaxRL" if campaign == "e118" else row["short_contrast"]
        lines.append(" & ".join([row["model"], row["domain"], contrast, n, effect(row, "pass8", tex=True, with_interval=False), effect(row, "distinct8", tex=True, with_interval=False), effect(row, "breadth8", tex=True)]) + r" \\")
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"



def render_forest(data: dict[str, Any], target: Path) -> list[Path]:
    """Export the same paired effects; hollow markers identify partial blocks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 15.5), sharey=True)
    fig.subplots_adjust(left=.335, right=.98, top=.91, bottom=.14, wspace=.16)
    positions, labels, heading_positions = [], [], []
    plotted = []
    colors = {"e118": "#137E81", "e119": "#3766B0", "e120": "#8B4C9D"}
    cursor = 0
    for campaign, label in CAMPAIGNS.items():
        heading_positions.append(cursor)
        positions.append(cursor)
        labels.append(label)
        cursor += 1.1
        for row in (r for r in data["rows"] if r["campaign"] == campaign):
            positions.append(cursor)
            suffix = " / " + row["short_contrast"] if campaign == "e119" else ""
            labels.append(f"{row['model']}  {row['domain']}{suffix}   {row['n']}/5")
            plotted.append((cursor, row))
            cursor += 1
        cursor += .8
    for ax, metric, xlabel in zip(axes, ("breadth8", "pass8"), ("Δ B8: additional correct modes", "Δ pass@8: probability")):
        ax.axvline(0, color="#555555", linewidth=.8, linestyle="--", zorder=1)
        for y, row in plotted:
            if not row["n"]:
                ax.text(0, y, "—", ha="center", va="center", color="#999999", fontsize=9)
                continue
            s = row["summaries"][metric]
            _, bounds = interval(s)
            color = colors[row["campaign"]]
            if bounds:
                ax.hlines(y, bounds[0], bounds[1], colors=color, linewidth=1.25, zorder=2)
                ax.vlines(bounds, y-.1, y+.1, colors=color, linewidth=1, zorder=2)
            ax.plot(s["mean"], y, marker="o", markersize=5.2, markeredgewidth=1.3, markeredgecolor=color,
                    markerfacecolor=color if row["n"] == 5 else "white", zorder=3)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylim(cursor-.5, -1)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x", color="#E4E4E4", linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#BBBBBB")
        for y in heading_positions:
            ax.axhline(y+.5, color="#DDDDDD", linewidth=.7)
    axes[0].set_yticks(positions, labels, fontsize=9)
    for tick, y in zip(axes[0].get_yticklabels(), positions):
        if y in heading_positions:
            tick.set_fontweight("bold")
            tick.set_fontsize(10)
    fig.suptitle("Current campaign results at step 3072", x=.335, y=.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(.335, .958, f"{data['analysis_date']} · All planned comparisons; available paired endpoints only", fontsize=11)
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="#555555", markersize=5, label="5/5 pairs: point estimate and stored 95% interval"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", linestyle="none", markersize=5, label="1–4/5 pairs: descriptive point estimate"),
    ], loc="upper left", bbox_to_anchor=(.327, .948), frameon=False, fontsize=9)
    fig.text(.03, .038,
             "E118/E119: replay minus base optimizer. E120: uniform minus frequency replay. B8 = distinct@8 − pass@8.\n"
             "Complete blocks: unadjusted paired Student-t intervals (E118/E119); exhaustive paired percentile bootstrap (E120).\n"
             "Partial blocks have no intervals; 0/5 has no estimate. E120 mechanism eligibility is reported separately in the results digest.",
             fontsize=9, linespacing=1.5)
    fig.text(.03, .012, f"Source: {data['source_snapshot']['path']} · SHA-256 {data['source_snapshot']['sha256'][:16]}", fontsize=8, color="#555555")
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = [target.with_suffix(".png"), target.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=180, metadata={"Software": "current-campaign-results"})
    fig.savefig(paths[1], metadata={"Creator": "current-campaign-results", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="dated latest_results_YYYYMMDD.json")
    parser.add_argument("--output-dir", type=Path, help="defaults to the source snapshot directory")
    parser.add_argument("--figures", action="store_true", help="also export the standalone two-panel forest plot")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    source = args.snapshot.resolve()
    snapshot = json.loads(source.read_text())
    data = prepare(snapshot, source)
    data["findings"] = findings(data)
    stamp = date.fromisoformat(data["analysis_date"]).strftime("%Y%m%d")
    stem = f"current_campaign_results_{stamp}"
    output = args.output_dir or source.parent
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        stem + ".json": json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        stem + ".csv": render_csv(data), stem + ".md": render_markdown(data),
        **{f"{stem}_{campaign}_table_body.tex": render_tex(data, campaign) for campaign in CAMPAIGNS},
    }
    coverage_lines = ["% Campaign & Training receipts & Admitted endpoints & Complete factorial blocks & Matched seed contrasts"]
    for c in data["coverage"]:
        coverage_lines.append(" & ".join([c["campaign"].upper(), f"{c['completion_receipts']}/{c['registered_cells']}", f"{c['admitted_terminal_endpoints']}/{c['registered_cells']}", f"{c['complete_five_seed_blocks']}/{c['registered_blocks']}", f"{c['paired_contrasts']}/{c['planned_paired_contrasts']}"]) + r" \\")
    artifacts[stem + "_coverage_table_body.tex"] = "\n".join(coverage_lines) + "\n" + r"\bottomrule" + "\n"
    for name, content in artifacts.items():
        (output / name).write_text(content)
    figure_paths = render_forest(data, args.figure_dir / stem) if args.figures else []
    print(json.dumps({"figures": [relative(p) for p in figure_paths], "source": relative(source), "rows": len(data["rows"]), "available_rows": sum(r["n"] > 0 for r in data["rows"]), "coverage": data["coverage"], "outputs": [relative(output / name) for name in artifacts]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
