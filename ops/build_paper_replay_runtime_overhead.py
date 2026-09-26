#!/usr/bin/env python3
"""What replay actually costs at runtime, against what the control was given.

App. "Compute accounting" prices a replay likelihood pass analytically and then
says, of its own table, that it "does not measure replay/control runtime
differences". That gap is what makes the equal-compute objection land: the
extended-horizon control is described as receiving more computation than replay
consumes, but replay's consumption was never quantified, so the comparison
rested on an assertion.

It is measurable from cohorts already on disk. E128 (control), E130
(capacity-one replay), E131 (control at twelve passes) and E132 (Re:Dr) all ran
on the same two ``cs`` A5000 nodes, on the same prompts and seeds, under the
same runtime. Pairing by domain and seed gives replay's overhead directly.

Two measures are emitted because they fail differently.

``learn_batch_time``
    Median per-update learner seconds times the cell's own update count, read
    from each cell's ``train_metrics.jsonl`` and ``TRAINING_COMPLETE.json``.
    This is the primary measure. Replay adds a likelihood pass inside the
    learner so its overhead shows per update, while the extended-horizon
    control buys its extra compute in update *count*; multiplying the two is
    what puts both on one axis. A median over thousands of updates is
    insensitive to a slow neighbour on the node, and the whole thing is
    reproducible from disk indefinitely.
``wall_clock``
    Whole-job elapsed seconds from the scheduler. This captures evaluation,
    checkpointing and I/O that the learner timer excludes, but it is sensitive
    to node sharing and the accounting database ages out. Supplied via
    ``--sacct`` (a ``JobID|Elapsed|State|NodeList`` dump) and cached into the
    payload so the number survives the record expiring.

The two agreeing is the point: a per-update measure and a whole-job measure
that fail for different reasons both put replay's overhead in the low single
digits, against the tens of percent handed to the extended-horizon control.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COHORTS = {
    "e128_control": {
        "ledger": "var/artifacts/e128_matched_control_05b_jobs.json",
        "label": "control, eight passes",
        "role": "baseline",
    },
    "e132_redr": {
        "ledger": "var/artifacts/e132_matched_redr_05b_jobs.json",
        "label": "Re:Dr replay, eight passes",
        "role": "what replay costs",
    },
    "e130_capacity_one": {
        "ledger": "var/artifacts/e130_mode_agnostic_replay_05b_jobs.json",
        "label": "capacity-one replay, eight passes",
        "role": "what replay costs",
    },
    "e131_extended": {
        "ledger": "var/artifacts/e131_extended_horizon_control_05b_jobs.json",
        "label": "control, twelve passes",
        "role": "what the control was given",
    },
}
BASELINE = "e128_control"
LEARNER_KEY = "train/learn_batch_time"
SCHEMA = "paper-replay-runtime-overhead-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_elapsed(text: str) -> int | None:
    """Slurm elapsed, ``[D-]HH:MM:SS``, to seconds."""

    text = text.strip()
    if not text:
        return None
    days, _, hms = text.partition("-")
    if not hms:
        days, hms = "0", days
    parts = hms.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
        return int(days) * 86400 + h * 3600 + m * 60 + s
    except ValueError:
        return None


def learner_seconds(run_dir: Path) -> float | None:
    """Total learner seconds: median per-update time times the update count."""

    complete = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    steps = int(complete["terminal_step"])
    values = []
    for line in (Path(complete["terminal_attempt"]) / "train_metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get(LEARNER_KEY)
        if value is not None:
            values.append(value)
    # Total learner seconds, not per-update: the horizon arm's extra compute is
    # in the update count, so a per-update ratio would report it as free.
    return st.median(values) * steps if values else None


def fmt(value: float, places: int = 3) -> str:
    text = f"{value:.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def pct(value: float) -> str:
    """A signed percentage, for an overhead stated relative to the control."""

    return f"{100 * value:+.1f}\\%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    parser.add_argument("--stamp", default=dt.date.today().isoformat().replace("-", ""))
    parser.add_argument(
        "--sacct",
        type=Path,
        help="optional 'JobID|Elapsed|State|NodeList' dump (sacct -X -P -n) for "
             "the whole-job measure; without it only the learner measure is emitted",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    wall_by_job: dict[str, int] = {}
    if args.sacct:
        for line in args.sacct.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3 or parts[2] != "COMPLETED":
                continue
            seconds = parse_elapsed(parts[1])
            if seconds is not None:
                wall_by_job[parts[0]] = seconds

    cohorts: dict[str, dict] = {}
    for name, spec in COHORTS.items():
        ledger_path = ROOT / spec["ledger"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        cells = {}
        for run in ledger["runs"]:
            key = f"{run['domain']}|{run['seed']}"
            cells[key] = {
                "learner_seconds": learner_seconds(Path(run["run_dir"])),
                "wall_seconds": wall_by_job.get(str(run["job_id"])),
                "nodes": ledger.get("placement", {}).get("nodelist"),
            }
        cohorts[name] = {
            "label": spec["label"], "role": spec["role"],
            "sha256": sha256(ledger_path), "path": spec["ledger"],
            "cells": cells,
        }

    base = cohorts[BASELINE]["cells"]
    for name, cohort in cohorts.items():
        for measure, field in (("learner", "learner_seconds"), ("wall", "wall_seconds")):
            paired = [
                (c[field], base[k][field])
                for k, c in cohort["cells"].items()
                if k in base and c[field] is not None and base[k][field] is not None
            ]
            cohort[f"{measure}_mean_seconds"] = (
                st.fmean(v for v, _ in paired) if paired else None
            )
            cohort[f"{measure}_ratio"] = (
                st.fmean(v / b for v, b in paired) if paired else None
            )
            cohort[f"{measure}_paired_cells"] = len(paired)

    redr = cohorts["e132_redr"]
    extended = cohorts["e131_extended"]
    headroom = {}
    for measure in ("learner", "wall"):
        overhead = redr.get(f"{measure}_ratio")
        given = extended.get(f"{measure}_ratio")
        if overhead and given and overhead > 1.0:
            headroom[measure] = {
                "replay_overhead": overhead - 1.0,
                "control_was_given": given - 1.0,
                "multiple_of_replay_margin": (given - 1.0) / (overhead - 1.0),
            }

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": BASELINE,
        "measures": {
            "learner": f"median per-update {LEARNER_KEY}, from each cell's "
                       "train_metrics.jsonl; primary, reproducible from disk",
            "wall": "whole-job COMPLETED elapsed from the scheduler; includes "
                    "evaluation, checkpointing and I/O, sensitive to node sharing",
        },
        "placement": "cs partition, allcs account, A5000, node203/node204 for "
                     "every cohort compared here",
        "wall_clock_available": bool(wall_by_job),
        "cohorts": cohorts,
        "headroom": headroom,
        "closes": "App. compute accounting states that its table does not "
                  "measure replay/control runtime differences; this payload "
                  "supplies that measurement",
    }
    (out / f"replay_runtime_overhead_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    macros = ["% Generated by ops/build_paper_replay_runtime_overhead.py; do not hand edit."]
    macros.append(rf"\newcommand{{\RTcells}}{{{redr['learner_paired_cells']}}}")
    macros.append(rf"\newcommand{{\RTlearneroverhead}}{{{pct(redr['learner_ratio'] - 1)}}}")
    macros.append(rf"\newcommand{{\RTcapacityoneoverhead}}"
                  rf"{{{pct(cohorts['e130_capacity_one']['learner_ratio'] - 1)}}}")
    macros.append(rf"\newcommand{{\RTcontrolgiven}}{{{pct(extended['learner_ratio'] - 1)}}}")
    if "wall" in headroom:
        macros.append(rf"\newcommand{{\RTwalloverhead}}{{{pct(headroom['wall']['replay_overhead'])}}}")
        macros.append(rf"\newcommand{{\RTwallgiven}}{{{pct(headroom['wall']['control_was_given'])}}}")
        macros.append(rf"\newcommand{{\RTwallmultiple}}"
                      rf"{{{headroom['wall']['multiple_of_replay_margin']:.0f}}}")
    if "learner" in headroom:
        macros.append(rf"\newcommand{{\RTlearnermultiple}}"
                      rf"{{{headroom['learner']['multiple_of_replay_margin']:.0f}}}")
    (out / f"replay_runtime_overhead_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    order = ("e128_control", "e132_redr", "e130_capacity_one", "e131_extended")
    body = ["% Generated by ops/build_paper_replay_runtime_overhead.py; do not hand edit."]
    for name in order:
        c = cohorts[name]
        lr = "---" if name == BASELINE else pct(c["learner_ratio"] - 1)
        wr = ("---" if name == BASELINE
              else (pct(c["wall_ratio"] - 1) if c.get("wall_ratio") else "n/a"))
        wall = f"{c['wall_mean_seconds']/3600:.2f}" if c.get("wall_mean_seconds") else "n/a"
        body.append(
            f"  {c['label']} & {fmt(c['learner_mean_seconds']/3600, 2)} & {lr}"
            f" & {wall} & {wr} \\\\"
        )
    body.append(r"  \bottomrule")
    (out / f"replay_runtime_overhead_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote replay_runtime_overhead_{args.stamp}.{{json,_macros.tex,_table_body.tex}}")
    for name in order:
        c = cohorts[name]
        lr = "baseline" if name == BASELINE else f"{c['learner_ratio']:.3f}x learner-total"
        wr = ("" if name == BASELINE or not c.get("wall_ratio")
              else f", {c['wall_ratio']:.3f}x wall")
        print(f"  {c['label']:36s} {lr}{wr}  (n={c['learner_paired_cells']})")
    for measure, h in headroom.items():
        print(f"  [{measure}] replay costs {100*h['replay_overhead']:+.1f}%, control was "
              f"given {100*h['control_was_given']:+.1f}% "
              f"= {h['multiple_of_replay_margin']:.0f}x replay's margin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
