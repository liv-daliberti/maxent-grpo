#!/usr/bin/env python3
"""Aggregate the E72 B3a cohort into its registered three-arm comparison.

B3a removes only the replay gradient from the treatment, so it is read twice
against the same cohort:

    B3a vs xGRPO           isolates verified replay
    B3a vs matched Dr.GRPO measures what count-based discovery credit alone buys

The reference arms are not recomputed here. They are read from the frozen
scaling-curve values already bound in the frontier manifest, which are the exact
numbers the paper reports, so the ablation cannot drift against a
differently-derived baseline.

Per the protocol, a domain is reportable only when all five seeds are terminal.
Incomplete domains are withheld unless ``--interim`` is passed, and interim
output is labelled as such everywhere it appears: a partial seed set is a
monitoring aid, not a result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

SEEDS = (43, 44, 45, 46, 47)
DOMAIN_BY_PREFIX = {
    "gc": "graph_coloring",
    "cd": "countdown",
    "py": "python_factors",
    "mi": "mathir",
    "pp": "pantry_plan",
}
DOMAIN_TITLES = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR action menu",
    "pantry_plan": "PantryPlan",
}
RUN_GLOB = "var/data/xdr_qwen25_0p5b_instruct_verified_first_replay_gradient_ablation_*"
RUN_NAME = re.compile(r"_(gc|cd|py|mi|pp)e7[01]_.*_b3a_s(?P<seed>\d+)$")

METRICS = ("greedy", "mean8", "pass8", "distinct8")
METRIC_KEYS = {
    "greedy": "eval/multi_answer/accuracy",
    "mean8": "eval/multi_answer/sampled_mean_at_8",
    "pass8": "eval/multi_answer/sampled_any_correct_at_8",
    "distinct8": "eval/multi_answer/sampled_distinct_correct_at_8",
}

# Integrity conditions specific to this arm, from the protocol: the replay
# gradient must be identically zero on every logged update, and discovery credit
# must actually have acted.
ZERO_GRADIENT_KEYS = (
    "train/canonical_replay_applied_score_gradient_l2",
    "train/canonical_replay_applied_score_gradient_sum",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_terminal_row(attempt: Path) -> dict[str, Any] | None:
    """Return the deepest evaluation this attempt logged, with integrity flags."""
    metrics_path = attempt / "train_metrics.jsonl"
    if not metrics_path.is_file():
        return None
    terminal: dict[str, Any] | None = None
    compute_only_seen = False
    nonzero_replay_gradient = 0
    novelty_active = False
    for line in metrics_path.open("r", encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if METRIC_KEYS["distinct8"] in record:
            terminal = record
        if record.get("train/canonical_replay_compute_only") == 1.0:
            compute_only_seen = True
        for key in ZERO_GRADIENT_KEYS:
            value = record.get(key)
            if value is not None and abs(float(value)) > 0.0:
                nonzero_replay_gradient += 1
        if float(record.get("train/online_canonical_novelty_advantage_rms", 0.0) or 0.0) > 0:
            novelty_active = True
    if terminal is None:
        return None
    return {
        "step": int(terminal.get("misc/global_step", -1)),
        **{name: terminal.get(key) for name, key in METRIC_KEYS.items()},
        "integrity": {
            "replay_compute_only_observed": compute_only_seen,
            "nonzero_replay_gradient_updates": nonzero_replay_gradient,
            "novelty_credit_active": novelty_active,
        },
    }


def collect_b3a(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    found: dict[str, dict[int, dict[str, Any]]] = {}
    for run_dir in sorted(root.glob(RUN_GLOB.replace("var/data/", "var/data/"))):
        match = RUN_NAME.search(run_dir.name)
        if match is None:
            continue
        domain = DOMAIN_BY_PREFIX[match.group(1)]
        seed = int(match.group("seed"))
        terminal_marker = run_dir / "TRAINING_COMPLETE.json"
        attempts = sorted(run_dir.glob("debug_job*"))
        row = None
        for attempt in reversed(attempts):
            row = read_terminal_row(attempt)
            if row is not None:
                break
        if row is None:
            continue
        row["terminal"] = terminal_marker.is_file()
        row["run_dir"] = str(run_dir)
        found.setdefault(domain, {})[seed] = row
    return found


def reference_arms(manifest: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (run["domain"], run["arm"], int(run["seed"])): run["published_terminal"]
        for run in manifest["runs"]
        if run.get("published_terminal")
    }


def summarize(
    b3a: dict[str, dict[int, dict[str, Any]]],
    reference: dict[tuple[str, str, int], dict[str, Any]],
    *,
    interim: bool,
) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    for domain in DOMAIN_TITLES:
        cells = b3a.get(domain, {})
        terminal_seeds = sorted(seed for seed, row in cells.items() if row["terminal"])
        reportable = len(terminal_seeds) == len(SEEDS)
        if not reportable and not interim:
            domains.append(
                {
                    "domain": domain,
                    "reportable": False,
                    "terminal_seeds": terminal_seeds,
                    "withheld": "fewer than five terminal seeds",
                }
            )
            continue
        usable = terminal_seeds if reportable else sorted(cells)
        if not usable:
            domains.append(
                {"domain": domain, "reportable": False, "terminal_seeds": [], "withheld": "no runs"}
            )
            continue

        per_seed = []
        for seed in usable:
            row = cells[seed]
            entry = {
                "seed": seed,
                "terminal": row["terminal"],
                "step": row["step"],
                "b3a": {metric: row[metric] for metric in METRICS},
                "drgrpo": reference.get((domain, "drgrpo", seed)),
                "xgrpo": reference.get((domain, "xgrpo", seed)),
                "integrity": row["integrity"],
            }
            per_seed.append(entry)

        def arm_mean(arm: str, metric: str) -> float | None:
            key = "distinct8" if metric == "distinct8" else metric
            values = []
            for entry in per_seed:
                source = entry[arm]
                if source is None:
                    continue
                value = source.get(key)
                if value is not None:
                    values.append(float(value))
            return statistics.fmean(values) if values else None

        means = {
            arm: {metric: arm_mean(arm, metric) for metric in METRICS}
            for arm in ("drgrpo", "b3a", "xgrpo")
        }
        paired = {}
        for opponent in ("xgrpo", "drgrpo"):
            deltas = []
            for entry in per_seed:
                other = entry[opponent]
                if other is None or entry["b3a"]["distinct8"] is None:
                    continue
                deltas.append(float(entry["b3a"]["distinct8"]) - float(other["distinct8"]))
            paired[f"b3a_minus_{opponent}_distinct8"] = {
                "per_seed": deltas,
                "mean": statistics.fmean(deltas) if deltas else None,
                "wins": sum(1 for value in deltas if value > 0),
                "n": len(deltas),
            }

        violations = [
            entry["seed"]
            for entry in per_seed
            if entry["integrity"]["nonzero_replay_gradient_updates"]
            or not entry["integrity"]["replay_compute_only_observed"]
            or not entry["integrity"]["novelty_credit_active"]
        ]
        domains.append(
            {
                "domain": domain,
                "reportable": reportable,
                "interim": not reportable,
                "terminal_seeds": terminal_seeds,
                "seeds_used": usable,
                "means": means,
                "paired": paired,
                "per_seed": per_seed,
                "integrity_violations": violations,
            }
        )
    return {
        "schema": "e72_b3a_summary_v1",
        "interim_mode": interim,
        "reportable_domains": [d["domain"] for d in domains if d.get("reportable")],
        "domains": domains,
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# E72 B3a — replay-gradient ablation" + (" (INTERIM)" if summary["interim_mode"] else ""),
        "",
    ]
    if summary["interim_mode"]:
        lines += [
            "**Interim: not a result, and biased against B3a.** The protocol reports a",
            "domain only when all five seeds are terminal. Rows below may include seeds",
            "still training, whose most recent evaluation is at an *earlier* pass than",
            "the pass-12 references they are differenced against. Breadth rises through",
            "training in every arm measured so far, so a partial row understates B3a by",
            "an unknown amount. Read these only as evidence that the pipeline is",
            "producing numbers, never as a direction.",
            "",
        ]
    lines += [
        "`#modes` is distinct@8 at the terminal pass-12 checkpoint. Reference arms",
        "are the frozen published values, not recomputed here.",
        "",
        "| domain | seeds | Dr.GRPO #modes | B3a #modes | xGRPO #modes | B3a-xGRPO | B3a-Dr.GRPO |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in summary["domains"]:
        if not entry.get("means"):
            lines.append(
                f"| {DOMAIN_TITLES[entry['domain']]} | "
                f"{len(entry.get('terminal_seeds', []))}/5 | _withheld: "
                f"{entry.get('withheld', 'incomplete')}_ | | | | |"
            )
            continue

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        means = entry["means"]
        paired = entry["paired"]
        # Label by terminal seeds, never by how many rows happen to be usable:
        # an interim row mixes mid-training checkpoints with terminal ones.
        label = f"{len(entry['terminal_seeds'])}/5"
        if entry.get("interim"):
            label += f" (+{len(entry['seeds_used']) - len(entry['terminal_seeds'])} partial)"
        lines.append(
            f"| {DOMAIN_TITLES[entry['domain']]} | {label} | "
            f"{fmt(means['drgrpo']['distinct8'])} | {fmt(means['b3a']['distinct8'])} | "
            f"{fmt(means['xgrpo']['distinct8'])} | "
            f"{fmt(paired['b3a_minus_xgrpo_distinct8']['mean'])} | "
            f"{fmt(paired['b3a_minus_drgrpo_distinct8']['mean'])} |"
        )
    violations = [
        (entry["domain"], entry["integrity_violations"])
        for entry in summary["domains"]
        if entry.get("integrity_violations")
    ]
    lines += ["", "## Integrity", ""]
    if violations:
        lines.append("**VIOLATIONS** — these runs are not this arm and must be discarded:")
        for domain, seeds in violations:
            lines.append(f"- {domain}: seeds {seeds}")
    else:
        lines.append(
            "All summarized runs observed `replay_compute_only=1`, zero applied replay "
            "gradient on every logged update, and active novelty credit."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "var" / "artifacts" / "e72_b3a_summary.json",
    )
    parser.add_argument(
        "--interim",
        action="store_true",
        help="include domains with fewer than five terminal seeds, labelled as interim",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    summary = summarize(
        collect_b3a(root), reference_arms(manifest), interim=args.interim
    )

    handle, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(summary, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, args.output)

    report = root / "paper" / "results" / (
        "e72_b3a_interim.md" if args.interim else "e72_b3a_terminal.md"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    write_report(summary, report)

    reportable = summary["reportable_domains"]
    print(
        f"[e72-b3a] reportable domains: {len(reportable)}/5 "
        f"{reportable if reportable else ''}"
    )
    print(f"[e72-b3a] wrote {args.output} and {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
