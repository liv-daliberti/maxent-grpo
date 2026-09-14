#!/usr/bin/env python3
"""Audit the canonical paper experiment matrix against released run ledgers.

cohorts.py answers an operational question: which released campaigns must
appear in the live scheduler dashboard? This module answers a different,
scientific question: which cells of the final paper design exist, are terminal,
or are still missing?

The distinction is intentional. Repair cohorts, admission gates, superseded
pilots, and historical campaigns are real compute but are not additional cells
in the final method x model x domain x seed matrix. Conversely, an absent
method cell must remain visible here even when no launcher or ledger exists for
it yet.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as status_reader  # noqa: E402


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    paper_role: str
    description: str


@dataclass(frozen=True)
class Scale:
    key: str
    label: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Domain:
    key: str
    label: str
    stratum: str


@dataclass(frozen=True)
class Source:
    """Map arms in one immutable released ledger to canonical methods."""

    tag: str
    ledger: str
    scale: str
    arm_methods: tuple[tuple[str, str], ...]
    reader: str = "static"
    note: str = ""
    blocked_because: str = ""

    def method_for(self, arm: str) -> str | None:
        return dict(self.arm_methods).get(arm)


@dataclass(frozen=True, order=True)
class CellKey:
    method: str
    scale: str
    domain: str
    seed: int


@dataclass(frozen=True)
class Cell:
    key: CellKey
    source: str
    ledger: str
    arm: str
    job_id: int
    state: str
    step: int
    target: int
    blocked_because: str = ""
    replaces: str = ""

    @property
    def status(self) -> str:
        # Match campaign_stats.py: a scientific endpoint is terminal only when
        # its registered optimizer horizon is realized. A scheduler job that
        # exits cleanly early is not silently promoted to a terminal result.
        if self.step >= self.target:
            return "terminal"
        if self.blocked_because:
            return "blocked"
        if self.state == "RUNNING":
            return "running"
        if self.state == "PENDING":
            return "pending"
        if self.state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            return "failed"
        if self.step > 0:
            return "partial"
        return "inactive"


# The order is the order used in status tables and the paper planning document.
METHODS: tuple[Method, ...] = (
    Method(
        "grpo",
        "GRPO",
        "direct comparator",
        "Ordinary group-relative policy optimization.",
    ),
    Method(
        "drgrpo",
        "Dr.GRPO",
        "core control",
        "Length-debiased GRPO and the matched zero-replay control.",
    ),
    Method(
        "ucpo",
        "UCPO",
        "direct comparator",
        "Online redistribution toward a uniform correct conditional policy.",
    ),
    Method(
        "rlep_dr",
        "RLEP-Dr",
        "direct comparator",
        "Prompt-matched verified-success replay on eligible prompts, with an "
        "unchanged Dr.GRPO update otherwise.",
    ),
    Method(
        "replay_grpo",
        "Re:Dr.GRPO",
        "headline method",
        "Canonical mode-balanced verified replay on top of Dr.GRPO.",
    ),
    Method(
        "adaptive_replay_grpo",
        "Adaptive Re:Dr.GRPO",
        "replay ablation",
        "Re:Dr.GRPO with bank-normalized replay dose.",
    ),
    Method(
        "semantic_maxent",
        "Semantic MaxEnt",
        "semantic ablation",
        "Fixed semantic-MaxEnt derivative without verified replay.",
    ),
    Method(
        "adaptive_semantic_maxent",
        "Adaptive Semantic MaxEnt",
        "semantic ablation",
        "Adaptive semantic-MaxEnt derivative without verified replay.",
    ),
    Method(
        "replay_semantic_maxent",
        "Re:Dr.GRPO + Semantic MaxEnt",
        "semantic factorial",
        "Fixed semantic MaxEnt added to canonical verified replay.",
    ),
    Method(
        "adaptive_semantic_replay",
        "Adaptive Semantic MaxEnt + Re:Dr.GRPO",
        "semantic factorial",
        "Adaptive semantic MaxEnt added to canonical verified replay.",
    ),
)

SCALES: tuple[Scale, ...] = (
    Scale("qwen05b", "Qwen2.5-0.5B", (43, 44, 45, 46, 47)),
    Scale("falcon1b", "Falcon3-1B", (55, 56, 57, 58, 59)),
    Scale("qwen3b", "Qwen2.5-3B", (70, 71, 72, 73, 74)),
)

DOMAINS: tuple[Domain, ...] = (
    Domain("graph_coloring", "Graph coloring", "static"),
    Domain("countdown", "Countdown", "static"),
    Domain("python_factors", "Python factors", "static"),
    Domain("mathir", "MathIR", "static"),
    Domain("pantry_plan", "PantryPlan", "static"),
)


# E98 is retained as a named historical source so its failed feasibility result
# cannot disappear when the separately preregistered E98-R1 repair becomes the
# active scientific source.  It is intentionally absent from SOURCES: loading it
# there would overwrite the live E98-R1 cells with permanently blocked records.
E98_FAILED_FEASIBILITY = Source(
    "e98",
    "e98_rlep_dr_05b_jobs.json",
    "qwen05b",
    (("rlep_dr", "rlep_dr"),),
    blocked_because=(
        "the preregistered Graph/s43 replay-pool audit failed with 253 "
        "ineligible prompts, so shared smoke job 30508710 cannot run"
    ),
)


SOURCES: tuple[Source, ...] = (
    # Matched Dr.GRPO and canonical replay.
    Source(
        "e78",
        "e78_verified_replay_only_05b_jobs.json",
        "qwen05b",
        (("control", "drgrpo"), ("replay", "replay_grpo")),
    ),
    Source(
        "e79",
        "e79_falcon1b_aligned_verified_replay_jobs.json",
        "falcon1b",
        (("control", "drgrpo"), ("replay", "replay_grpo")),
    ),
    Source(
        "e80r1",
        "e80r1_qwen3b_aligned_verified_replay_jobs.json",
        "qwen3b",
        (("control", "drgrpo"), ("replay", "replay_grpo")),
    ),
    # Fixed semantic factorial arms.
    Source(
        "e81",
        "e81_semantic_maxent_verified_replay_05b_jobs.json",
        "qwen05b",
        (("semantic", "replay_semantic_maxent"),),
    ),
    Source(
        "e82",
        "e82_falcon_semantic_maxent_verified_replay_jobs.json",
        "falcon1b",
        (("semantic", "replay_semantic_maxent"),),
    ),
    Source(
        "e83",
        "e83_semantic_maxent_without_replay_05b_jobs.json",
        "qwen05b",
        (("semantic_only", "semantic_maxent"),),
    ),
    Source(
        "e86",
        "e86_falcon_semantic_maxent_without_replay_jobs.json",
        "falcon1b",
        (("semantic_only", "semantic_maxent"),),
    ),
    Source(
        "e87",
        "e87_qwen3b_semantic_maxent_seed70_jobs.json",
        "qwen3b",
        (("semantic", "replay_semantic_maxent"),),
        note="one registered seed per static domain",
    ),
    # The failed E88 controller gate is intentionally not a scientific source.
    Source(
        "e89",
        "e89_adaptive_semantic_maxent_reachable_05b_jobs.json",
        "qwen05b",
        (("adaptive_semantic_reachable", "adaptive_semantic_replay"),),
    ),
    Source(
        "e90",
        "e90_bank_normalized_replay_05b_jobs.json",
        "qwen05b",
        (("bank_normalized_replay", "adaptive_replay_grpo"),),
    ),
    Source(
        "e91",
        "e91_falcon_adaptive_semantic_maxent_jobs.json",
        "falcon1b",
        (("adaptive_semantic_reachable", "adaptive_semantic_replay"),),
    ),
    Source(
        "e92",
        "e92_qwen3b_adaptive_semantic_maxent_jobs.json",
        "qwen3b",
        (("adaptive_semantic_reachable", "adaptive_semantic_replay"),),
        note="one registered seed per static domain",
    ),
    # Ordinary GRPO bridge.
    Source(
        "e95_05b",
        "e95_plain_grpo_Qwen25-05B_jobs.json",
        "qwen05b",
        (("grpo_plain_control", "grpo"),),
    ),
    Source(
        "e95_1b",
        "e95_plain_grpo_Falcon3-1B_jobs.json",
        "falcon1b",
        (("grpo_plain_control", "grpo"),),
    ),
    Source(
        "e95_3b",
        "e95_plain_grpo_Qwen25-3B_jobs.json",
        "qwen3b",
        (("grpo_plain_control", "grpo"),),
    ),
    Source(
        "e114",
        "e114_plain_grpo_qwen3b_extension_jobs.json",
        "qwen3b",
        (("grpo_plain_control", "grpo"),),
        note="prospective E95 seed extension covering Qwen2.5-3B seeds 71--74",
    ),
    # Closest direct baselines currently cover three representative domains.
    Source(
        "e97",
        "e97_ucpo_05b_jobs.json",
        "qwen05b",
        (("ucpo", "ucpo"),),
    ),
    Source(
        "e98r1",
        "e98r1_sparse_rlep_dr_05b_jobs.json",
        "qwen05b",
        (("rlep_dr_sparse", "rlep_dr"),),
        note=(
            "separately preregistered sparse repair after E98 failed its "
            "all-prompts feasibility gate"
        ),
    ),
    Source(
        "e99",
        "e99_ucpo_falcon1b_jobs.json",
        "falcon1b",
        (("ucpo", "ucpo"),),
    ),
    Source(
        "e100",
        "e100_sparse_rlep_dr_falcon1b_jobs.json",
        "falcon1b",
        (("rlep_dr_sparse", "rlep_dr"),),
        note=(
            "prospective sparse prompt-matched replay with per-cell Falcon "
            "prior-policy pool and fail-closed pool/smoke gates"
        ),
    ),
    Source(
        "e115_05b",
        "e115_ucpo_qwen05b_domain_extension_jobs.json",
        "qwen05b",
        (("ucpo", "ucpo"),),
        note="prospective E97 extension to Countdown and MathIR",
    ),
    Source(
        "e115_3b",
        "e115_ucpo_qwen3b_jobs.json",
        "qwen3b",
        (("ucpo", "ucpo"),),
        note="prospective five-domain Qwen2.5-3B UCPO scale port",
    ),
    Source(
        "e116_05b",
        "e116_sparse_rlep_qwen05b_domain_extension_jobs.json",
        "qwen05b",
        (("rlep_dr_sparse", "rlep_dr"),),
        note="prospective sparse E98-R1 extension to Countdown and MathIR",
    ),
    Source(
        "e116_3b",
        "e116_sparse_rlep_qwen3b_jobs.json",
        "qwen3b",
        (("rlep_dr_sparse", "rlep_dr"),),
        note="prospective five-domain Qwen2.5-3B sparse RLEP-Dr scale port",
    ),
)


# E85 reruns defective PantryPlan cells from three sources. It is loaded after
# SOURCES and replaces, rather than adds to, those canonical cells.
REPAIR_LEDGER = "e85_pantry_semantic_repair_jobs.json"
REPAIR_METHODS = {
    "e81": "replay_semantic_maxent",
    "e82": "replay_semantic_maxent",
    "e83": "semantic_maxent",
}

METHOD_BY_KEY = {item.key: item for item in METHODS}
SCALE_BY_KEY = {item.key: item for item in SCALES}
DOMAIN_BY_KEY = {item.key: item for item in DOMAINS}
STATUS_ORDER = (
    "terminal",
    "running",
    "pending",
    "blocked",
    "partial",
    "failed",
    "inactive",
    "missing",
)


def desired_keys() -> tuple[CellKey, ...]:
    return tuple(
        CellKey(method.key, scale.key, domain.key, seed)
        for method in METHODS
        for scale in SCALES
        for domain in DOMAINS
        for seed in scale.seeds
    )


def validate_spec() -> None:
    """Fail closed if the paper's 10 x 3 x 5 x 5 grid drifts."""

    for label, values in (
        ("method", [item.key for item in METHODS]),
        ("scale", [item.key for item in SCALES]),
        ("domain", [item.key for item in DOMAINS]),
        ("source tag", [item.tag for item in SOURCES]),
        ("source ledger", [item.ledger for item in SOURCES]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} in paper matrix")
    if len(METHODS) != 10 or len(SCALES) != 3 or len(DOMAINS) != 5:
        raise ValueError("paper matrix must remain 10 methods x 3 scales x 5 domains")
    if any(len(scale.seeds) != 5 for scale in SCALES):
        raise ValueError("every model scale must declare exactly five seeds")
    keys = desired_keys()
    if len(keys) != 750 or len(set(keys)) != 750:
        raise ValueError("paper matrix must contain exactly 750 unique cells")
    for source in SOURCES:
        if source.scale not in SCALE_BY_KEY:
            raise ValueError(f"{source.tag}: unknown scale {source.scale!r}")
        if source.reader not in {"static", "point"}:
            raise ValueError(f"{source.tag}: unknown reader {source.reader!r}")
        for arm, method in source.arm_methods:
            if not arm or method not in METHOD_BY_KEY:
                raise ValueError(f"{source.tag}: invalid arm mapping {(arm, method)!r}")


def _snapshot(source: Source) -> tuple[dict[str, object], dict[str, object]]:
    path = ARTIFACTS / source.ledger
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if source.reader == "point":
        snapshot = status_reader.load_point_snapshot(path)
    else:
        snapshot = status_reader.load_snapshot(path)
    return ledger, snapshot


def _source_cells(source: Source) -> Iterable[Cell]:
    path = ARTIFACTS / source.ledger
    if not path.is_file():
        return ()
    ledger, snapshot = _snapshot(source)
    rows = {int(row["job_id"]): row for row in snapshot["rows"]}
    out: list[Cell] = []
    for run in ledger["runs"]:
        method = source.method_for(str(run["arm"]))
        if method is None:
            continue
        job_id = int(run["job_id"])
        row = rows[job_id]
        # The immutable ledger run is authoritative for domain identity.
        domain = str(run["domain"])
        key = CellKey(method, source.scale, domain, int(run["seed"]))
        out.append(
            Cell(
                key=key,
                source=source.tag,
                ledger=source.ledger,
                arm=str(run["arm"]),
                job_id=job_id,
                state=str(row["state"]),
                step=int(row["step"]),
                target=int(snapshot["target"]),
                blocked_because=source.blocked_because,
            )
        )
    # A per-cell scientific hard gate can fail before a training job becomes
    # runnable. Keep those cells out of the operational campaign denominator
    # while retaining them as explicit blocked cells in the paper matrix.
    for run in ledger.get("blocked_runs", []):
        method = source.method_for(str(run["arm"]))
        if method is None:
            continue
        reason = str(run.get("blocked_because", "")).strip()
        if not reason:
            raise ValueError(f"{source.tag}: blocked run lacks a reason")
        out.append(
            Cell(
                key=CellKey(
                    method,
                    source.scale,
                    str(run["domain"]),
                    int(run["seed"]),
                ),
                source=source.tag,
                ledger=source.ledger,
                arm=str(run["arm"]),
                job_id=int(run["job_id"]),
                state="CANCELLED",
                step=0,
                target=int(snapshot["target"]),
                blocked_because=reason,
            )
        )
    return out


def _repair_cells() -> Iterable[Cell]:
    path = ARTIFACTS / REPAIR_LEDGER
    if not path.is_file():
        return ()
    ledger = json.loads(path.read_text(encoding="utf-8"))
    snapshot = status_reader.load_snapshot(path)
    rows = {int(row["job_id"]): row for row in snapshot["rows"]}
    out: list[Cell] = []
    for run in ledger["runs"]:
        replaced = str(run["supersedes"]["run_stamp"])
        cohort = replaced.split("_", maxsplit=1)[0]
        method = REPAIR_METHODS.get(cohort)
        if method is None:
            raise ValueError(f"E85 repair has unknown parent {replaced!r}")
        seed = int(run["seed"])
        if seed in SCALE_BY_KEY["qwen05b"].seeds:
            scale = "qwen05b"
        elif seed in SCALE_BY_KEY["falcon1b"].seeds:
            scale = "falcon1b"
        else:
            raise ValueError(f"E85 repair has seed outside known scales: {seed}")
        job_id = int(run["job_id"])
        row = rows[job_id]
        out.append(
            Cell(
                key=CellKey(method, scale, str(run["domain"]), seed),
                source="e85",
                ledger=REPAIR_LEDGER,
                arm=str(run["arm"]),
                job_id=job_id,
                state=str(row["state"]),
                step=int(row["step"]),
                target=int(snapshot["target"]),
                replaces=replaced,
            )
        )
    return out


def load_cells() -> dict[CellKey, Cell]:
    """Load canonical cells, applying registered scientific replacements."""

    validate_spec()
    desired = set(desired_keys())
    cells: dict[CellKey, Cell] = {}
    for source in SOURCES:
        for cell in _source_cells(source):
            if cell.key not in desired:
                raise ValueError(f"{cell.source}: cell outside canonical grid: {cell.key}")
            if cell.key in cells:
                raise ValueError(
                    f"duplicate canonical cell from {cells[cell.key].source} "
                    f"and {cell.source}: {cell.key}"
                )
            cells[cell.key] = cell
    for repair in _repair_cells():
        if repair.key not in cells:
            raise ValueError(f"repair has no canonical parent cell: {repair.key}")
        cells[repair.key] = repair
    return cells


def counts(cells: dict[CellKey, Cell], keys: Iterable[CellKey]) -> Counter[str]:
    out: Counter[str] = Counter()
    for key in keys:
        cell = cells.get(key)
        out[cell.status if cell is not None else "missing"] += 1
    return out


def _groups(view: str) -> list[tuple[str, list[CellKey]]]:
    keys = desired_keys()
    if view == "method":
        return [
            (method.label, [key for key in keys if key.method == method.key])
            for method in METHODS
        ]
    if view == "scale":
        return [
            (scale.label, [key for key in keys if key.scale == scale.key])
            for scale in SCALES
        ]
    if view == "domain":
        return [
            (domain.label, [key for key in keys if key.domain == domain.key])
            for domain in DOMAINS
        ]
    raise ValueError(view)


def render(cells: dict[CellKey, Cell], view: str, *, markdown: bool) -> str:
    groups = _groups(view)
    rows = [(label, len(keys), counts(cells, keys)) for label, keys in groups]
    headings = (
        "target",
        "reg",
        "term",
        "run",
        "pend",
        "block",
        "part",
        "fail",
        "idle",
        "miss",
    )
    if markdown:
        out = [
            f"### By {view}",
            "",
            "| group | " + " | ".join(headings) + " |",
            "|---|" + "---:|" * len(headings),
        ]
        for label, target, row in rows:
            registered = target - row["missing"]
            values = (
                target,
                registered,
                row["terminal"],
                row["running"],
                row["pending"],
                row["blocked"],
                row["partial"],
                row["failed"],
                row["inactive"],
                row["missing"],
            )
            out.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
        return "\n".join(out)

    width = max(len("group"), *(len(label) for label, _, _ in rows))
    out = [
        f"by {view}",
        f"{'group':<{width}} " + " ".join(f"{name:>6}" for name in headings),
        "-" * (width + 7 * len(headings)),
    ]
    for label, target, row in rows:
        registered = target - row["missing"]
        values = (
            target,
            registered,
            row["terminal"],
            row["running"],
            row["pending"],
            row["blocked"],
            row["partial"],
            row["failed"],
            row["inactive"],
            row["missing"],
        )
        out.append(f"{label:<{width}} " + " ".join(f"{value:>6}" for value in values))
    return "\n".join(out)


def json_payload(cells: dict[CellKey, Cell]) -> dict[str, object]:
    desired = desired_keys()
    total = counts(cells, desired)
    records = []
    for key in desired:
        cell = cells.get(key)
        record: dict[str, object] = asdict(key)
        if cell is None:
            record["status"] = "missing"
        else:
            record.update(asdict(cell))
            # asdict(cell) nests the key; keep one flat identity in JSON.
            record.pop("key", None)
            record["status"] = cell.status
        records.append(record)
    return {
        "schema": "modebench-paper-matrix-v2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "shape": {
            "methods": len(METHODS),
            "scales": len(SCALES),
            "domains": len(DOMAINS),
            "seeds_per_scale": 5,
            "cells": len(desired),
        },
        "methods": [asdict(item) for item in METHODS],
        "scales": [asdict(item) for item in SCALES],
        "domains": [asdict(item) for item in DOMAINS],
        "counts": {name: total[name] for name in STATUS_ORDER},
        "registered": len(cells),
        "cells": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        choices=("method", "scale", "domain", "all"),
        default="method",
        help="aggregation to print (default: method)",
    )
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit all 750 cells")
    args = parser.parse_args()

    cells = load_cells()
    if args.json:
        print(json.dumps(json_payload(cells), indent=2))
        return 0

    views = ("method", "scale", "domain") if args.view == "all" else (args.view,)
    overall = counts(cells, desired_keys())
    print(datetime.now().astimezone().strftime("paper matrix  %Y-%m-%d %H:%M:%S %Z"))
    print(
        f"750 target cells; {len(cells)} registered; "
        f"{overall['terminal']} terminal; {overall['missing']} not registered"
    )
    for index, view in enumerate(views):
        if index or not args.markdown:
            print()
        print(render(cells, view, markdown=args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
