#!/usr/bin/env python3
"""Render the interim three-model Figure 4 preview from E78, E79, and E80."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as style  # noqa: E402
import plot_e78_figure4_preview as live  # noqa: E402
import cohorts as registry  # noqa: E402


E78_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
E79_LEDGER = ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"
E80R1_LEDGER = ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
# E81 and E82 add a third arm to the Qwen-0.5B and Falcon-1B rows. Each shares
# its family's domains, seeds, schedule, and evaluation cadence, so their curves
# are read with the same reader and land on the same pass coordinate.
E81_LEDGER = ROOT / "var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json"
E82_LEDGER = (
    ROOT / "var/artifacts/e82_falcon_semantic_maxent_verified_replay_jobs.json"
)
E83_LEDGER = (
    ROOT / "var/artifacts/e83_semantic_maxent_without_replay_05b_jobs.json"
)
# E85 re-runs the PantryPlan cells whose semantic term never fired. Any domain
# it supersedes is drawn from the repair, never from the parent cohort.
REPAIR_LEDGER = ROOT / "var/artifacts/e85_pantry_semantic_repair_jobs.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/figure4_with_falcon_interim"
ROW_LABELS = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
STATIC_DOMAINS = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)


def _family_snapshot(ledger: Path) -> dict[str, Any]:
    """Read one model scale and retain only domains registered in its ledger."""

    snapshot = live._snapshot(ledger.resolve())
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    registered = {str(run["domain"]) for run in payload["runs"]}
    domains = [domain for domain in STATIC_DOMAINS if domain in registered]
    snapshot["domains"] = domains
    snapshot["curves"] = {
        domain: snapshot["curves"].get(domain, {}) for domain in domains
    }
    snapshot["total_runs"] = len(payload["runs"])
    return snapshot


def _attach_semantic(
    snapshot: dict[str, Any],
    ledger: Path,
    *,
    arm: str = "semantic",
    comparator: str = "replay",
    cohort: str | None = None,
) -> dict[str, Any]:
    """Attach one semantic-MaxEnt arm to a family snapshot, if it exists.

    Arms are kept in their own key rather than merged into ``curves`` so the
    existing two-arm pairing, the recorded provenance schema, and every
    downstream consumer of that schema stay exactly as they were.

    ``comparator`` names the arm this one's paired difference is taken against,
    and it is not the same for every arm: E81/E82 add the semantic term on top
    of ``replay`` and are read against it, while E83 adds it on top of
    ``control``. Pairing E83 against ``replay`` would report the sum of two
    interventions as the effect of one.

    ``cohort`` is the tag the repair ledger uses to name this arm's parent. Any
    domain the repair supersedes is dropped from the parent entirely and
    redrawn from the repair cells, because the superseded cells did not apply
    the treatment at all: PantryPlan's semantic term never fired, so those
    curves are the comparator's curves wearing the treatment's colour.
    """

    snapshot.setdefault("semantic_arms", [])
    if not ledger.is_file():
        return snapshot
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    interval = int(payload["checkpoint_interval_steps"])
    target = int(payload["target_steps"])
    curves: dict[str, dict[int, dict[int, float]]] = {}
    for run in payload["runs"]:
        curve = live._run_curve(
            Path(run["run_dir"]), interval=interval, target=target
        )
        if curve:
            curves.setdefault(str(run["domain"]), {})[int(run["seed"])] = curve

    superseded: list[str] = []
    if cohort and REPAIR_LEDGER.is_file():
        repair = json.loads(REPAIR_LEDGER.read_text(encoding="utf-8"))
        mine = [r for r in repair["runs"] if str(r.get("parent")) == cohort]
        superseded = sorted({str(r["domain"]) for r in mine})
        for domain in superseded:
            curves.pop(domain, None)
        r_interval = int(repair["checkpoint_interval_steps"])
        r_target = int(repair["target_steps"])
        for run in mine:
            curve = live._run_curve(
                Path(run["run_dir"]), interval=r_interval, target=r_target
            )
            if curve:
                curves.setdefault(str(run["domain"]), {})[int(run["seed"])] = curve
    snapshot["semantic_arms"].append(
        {
            "arm": arm,
            "comparator": comparator,
            "curves": curves,
            "total_runs": len(payload["runs"]),
            "observed_runs": sum(len(seeds) for seeds in curves.values()),
            "coefficient": float(payload.get("semantic_coefficient", 0.10)),
            "superseded_domains": superseded,
        }
    )
    return snapshot


def _semantic_arms(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return list(snapshot.get("semantic_arms", []))


def _observed_runs(snapshot: dict[str, Any]) -> int:
    if "frozen_observed_runs" in snapshot:
        return int(snapshot["frozen_observed_runs"])
    return sum(
        len(arm_curves)
        for domain_curves in snapshot["curves"].values()
        for arm_curves in domain_curves.values()
    )


def _family_from_provenance(payload: dict[str, Any], label: str) -> dict[str, Any]:
    """Adapt one frozen family record without rereading live run telemetry."""

    family = payload["families"][label]
    return {
        "passes": 8,
        "steps_per_pass": 384,
        "target": 3072,
        "interval": 192,
        "total_runs": int(family["total_runs"]),
        "domains": list(family["domains"]),
        "curves": {},
        # A frozen record carries its semantic arms as recorded metadata; the
        # curves themselves live in `frozen_domains` and are not re-read.
        "semantic_arms": [
            {
                "arm": str(spec["arm"]),
                "comparator": str(spec.get("comparator", "replay")),
                "curves": {},
                "total_runs": int(spec["total_runs"]),
                "observed_runs": int(spec["observed_runs"]),
                "coefficient": float(spec.get("coefficient", 0.10)),
            }
            for spec in family.get("semantic_arms", [])
        ],
        "frozen_domains": family["domains"],
        "frozen_observed_runs": int(family["observed_runs"]),
    }


def _compact_seeds(seeds: list[int]) -> str:
    """``43, 44, 45, 46, 47`` needs two lines in a 1.2in panel; ``43-47`` needs one."""

    if len(seeds) > 2 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ", ".join(map(str, seeds))


def _clearest_corner(axis, *, lines: int) -> tuple[float, float, str, str]:
    """Anchor for the provenance note: the corner holding the least data.

    A fixed top-left anchor was fine when the note was two lines. With five arms
    it is five lines tall and lands on the curves in whichever panels rise early
    -- PantryPlan most of all, where every arm is high from the first pass.
    Rather than shrink the note until it says nothing, put it where the panel is
    empty.
    """

    height = min(0.055 * lines + 0.04, 0.62)
    width = 0.62
    candidates = (
        (0.03, 0.96, "left", "top", (0.0, width, 0.96 - height, 0.96)),
        (0.97, 0.96, "right", "top", (1.0 - width, 1.0, 0.96 - height, 0.96)),
        (0.03, 0.04, "left", "bottom", (0.0, width, 0.04, 0.04 + height)),
        (0.97, 0.04, "right", "bottom", (1.0 - width, 1.0, 0.04, 0.04 + height)),
    )
    to_axes = axis.transData + axis.transAxes.inverted()
    points: list[Any] = []
    for line in axis.get_lines():
        data = line.get_xydata()
        if len(data):
            points.extend(to_axes.transform(data))

    best, best_load = candidates[0], None
    for candidate in candidates:
        x0, x1, y0, y1 = candidate[4]
        load = sum(1 for px, py in points if x0 <= px <= x1 and y0 <= py <= y1)
        if best_load is None or load < best_load:
            best, best_load = candidate, load
        if best_load == 0:
            break
    return best[0], best[1], best[2], best[3]


def _panel(
    axis: Any,
    *,
    snapshot: dict[str, Any],
    domain: str,
    row: int,
    total_rows: int,
    arm_styles: dict[str, tuple[Any, Any, str]] | None = None,
    mean_marker: str | None = None,
) -> dict[str, Any]:
    arm_styles = live.ARM_STYLE if arm_styles is None else arm_styles
    if row == 0:
        style.style_axis(
            axis,
            grid="both",
            title=live.DOMAIN_TITLES.get(domain, domain),
        )
    else:
        style.style_axis(axis, grid="both")
    axis.set_xlim(0, snapshot["passes"])
    axis.set_xticks([0, 2, 4, 6, 8])
    axis.axhline(
        1.0, color=style.MUTED, linewidth=0.65, linestyle=(0, (2, 2))
    )
    frozen = snapshot.get("frozen_domains", {}).get(domain)
    if frozen is not None:
        summaries = frozen["paired_summary_by_pass"]
        pass_values = sorted(float(value) for value in summaries)
        for arm in ("control", "replay"):
            color, dash, _label = arm_styles[arm]
            means = [summaries[str(value)][f"{arm}_mean"] for value in pass_values]
            ranges = [summaries[str(value)][f"{arm}_range"] for value in pass_values]
            axis.fill_between(
                pass_values,
                [pair[0] for pair in ranges],
                [pair[1] for pair in ranges],
                color=color,
                alpha=style.BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                pass_values,
                means,
                color=color,
                linestyle=dash,
                linewidth=style.MEAN_LW,
                marker=mean_marker,
                markevery=2 if mean_marker else None,
                markersize=2.4 if mean_marker else None,
                zorder=3,
            )
        by_arm = dict(frozen.get("semantic_summary_by_arm") or {})
        legacy = frozen.get("semantic_summary_by_pass")
        if legacy and "semantic" not in by_arm:
            by_arm["semantic"] = legacy
        for arm_name, semantic in by_arm.items():
            if not semantic:
                continue
            color, dash, _label = arm_styles[arm_name]
            semantic_passes = sorted(float(value) for value in semantic)
            axis.fill_between(
                semantic_passes,
                [semantic[str(v)]["semantic_range"][0] for v in semantic_passes],
                [semantic[str(v)]["semantic_range"][1] for v in semantic_passes],
                color=color,
                alpha=style.BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                semantic_passes,
                [semantic[str(v)]["semantic_mean"] for v in semantic_passes],
                color=color,
                linestyle=dash,
                linewidth=style.MEAN_LW,
                marker=mean_marker,
                markevery=2 if mean_marker else None,
                markersize=2.4 if mean_marker else None,
                zorder=3,
            )
        if pass_values:
            deepest = max(pass_values)
            if deepest < snapshot["passes"]:
                axis.axvspan(
                    deepest,
                    snapshot["passes"],
                    color="#F2F4F6",
                    alpha=0.82,
                    linewidth=0,
                    zorder=-2,
                )
            deepest_key = str(deepest)
            paired = summaries[deepest_key]["paired_seeds"]
            axis.text(
                0.04,
                0.95,
                "paired seeds\n" + ", ".join(map(str, paired)),
                transform=axis.transAxes,
                va="top",
                fontsize=style.SMALL_FONT,
                color=style.MUTED,
            )
        else:
            axis.axvspan(
                0,
                snapshot["passes"],
                color="#F2F4F6",
                alpha=0.82,
                linewidth=0,
                zorder=-2,
            )
            axis.text(
                0.5,
                0.52,
                "awaiting first\nregistered evaluation",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=style.SMALL_FONT,
                color=style.MUTED,
            )
        if row == total_rows - 1:
            axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)
        return frozen

    domain_curves = snapshot["curves"].get(domain, {})
    available_steps: set[int] = set()
    record: dict[str, Any] = {
        "arms": {},
        "paired_seeds_by_pass": {},
        "paired_summary_by_pass": {},
        # Semantic arms are recorded beside the two-arm estimator, never inside
        # it, and each keeps the comparator its difference was taken against.
        "semantic_seeds_by_arm": {},
        "semantic_summary_by_arm": {},
    }

    for arm in ("control", "replay"):
        color, dash, _label = arm_styles[arm]
        arm_curves = domain_curves.get(arm, {})
        record["arms"][arm] = sorted(arm_curves)
        for _seed, curve in sorted(arm_curves.items()):
            points = sorted(curve.items())
            if not points:
                continue
            available_steps.update(step for step, _value in points)
            axis.plot(
                [step / snapshot["steps_per_pass"] for step, _value in points],
                [value for _step, value in points],
                color=color,
                linestyle=dash,
                linewidth=style.SEED_LW,
                marker="o" if len(points) == 1 else None,
                markersize=2.8,
                alpha=0.34,
                zorder=2,
            )

    registered = range(
        0, snapshot["target"] + 1, snapshot["interval"]
    )
    for arm in ("control", "replay"):
        color, dash, _label = arm_styles[arm]
        xs: list[float] = []
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for step in registered:
            control, replay, paired = live._paired_values(domain_curves, step)
            values = control if arm == "control" else replay
            if not paired:
                continue
            pass_value = step / snapshot["steps_per_pass"]
            pass_key = str(pass_value)
            record["paired_seeds_by_pass"][pass_key] = paired
            record["paired_summary_by_pass"][pass_key] = {
                "paired_seeds": paired,
                "control_mean": sum(control) / len(control),
                "replay_mean": sum(replay) / len(replay),
                "replay_minus_control": (
                    sum(replay) / len(replay) - sum(control) / len(control)
                ),
                "control_range": [min(control), max(control)],
                "replay_range": [min(replay), max(replay)],
            }
            xs.append(pass_value)
            means.append(sum(values) / len(values))
            lows.append(min(values))
            highs.append(max(values))
        if xs:
            axis.fill_between(
                xs,
                lows,
                highs,
                color=color,
                alpha=style.BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                xs,
                means,
                color=color,
                linestyle=dash,
                linewidth=style.MEAN_LW,
                marker=mean_marker,
                markevery=2 if mean_marker else None,
                markersize=2.4 if mean_marker else None,
                zorder=3,
            )

    # --- semantic-MaxEnt arms ---------------------------------------------
    # Each arm's thick curve is taken over the seeds where it and its own
    # comparator are both present at that checkpoint: the same paired-only
    # discipline the two-arm curves use, applied to each arm's own comparison.
    # The comparator differs by arm -- E81/E82 read against `replay`, E83
    # against `control` -- so this loop must not assume one of them.
    for spec in _semantic_arms(snapshot):
        arm_name = str(spec["arm"])
        semantic_curves = spec["curves"].get(domain, {})
        if not semantic_curves:
            continue
        color, dash, _label = arm_styles[arm_name]
        record["semantic_seeds_by_arm"][arm_name] = sorted(semantic_curves)
        record["semantic_summary_by_arm"].setdefault(arm_name, {})
        for _seed, curve in sorted(semantic_curves.items()):
            points = sorted(curve.items())
            if not points:
                continue
            available_steps.update(step for step, _value in points)
            axis.plot(
                [step / snapshot["steps_per_pass"] for step, _value in points],
                [value for _step, value in points],
                color=color,
                linestyle=dash,
                linewidth=style.SEED_LW,
                marker="o" if len(points) == 1 else None,
                markersize=2.8,
                alpha=0.34,
                zorder=2,
            )
        comparator = str(spec["comparator"])
        against_curves = domain_curves.get(comparator, {})
        xs, means, lows, highs = [], [], [], []
        # Step 0 is the shared initialization: every arm is the same checkpoint
        # there, so its paired difference is exactly zero by construction. A
        # freshly launched cohort has nothing else yet, and admitting pass 0
        # would enter it in the legend and the panel note advertising a real
        # comparison that is arithmetic. The seed curves still start at 0.
        for step in registered:
            if step == 0:
                continue
            paired = sorted(
                seed
                for seed, curve in semantic_curves.items()
                if step in curve and step in against_curves.get(seed, {})
            )
            if not paired:
                continue
            values = [semantic_curves[seed][step] for seed in paired]
            against = [against_curves[seed][step] for seed in paired]
            pass_value = step / snapshot["steps_per_pass"]
            record["semantic_summary_by_arm"][arm_name][str(pass_value)] = {
                "paired_seeds": paired,
                "comparator": comparator,
                "semantic_mean": sum(values) / len(values),
                "comparator_mean": sum(against) / len(against),
                "semantic_minus_comparator": (
                    sum(values) / len(values) - sum(against) / len(against)
                ),
                "semantic_range": [min(values), max(values)],
            }
            xs.append(pass_value)
            means.append(sum(values) / len(values))
            lows.append(min(values))
            highs.append(max(values))
        if xs:
            axis.fill_between(
                xs,
                lows,
                highs,
                color=color,
                alpha=style.BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                xs,
                means,
                color=color,
                linestyle=dash,
                linewidth=style.MEAN_LW,
                marker=mean_marker,
                markevery=2 if mean_marker else None,
                markersize=2.4 if mean_marker else None,
                zorder=3,
            )

    if available_steps:
        deepest = max(available_steps) / snapshot["steps_per_pass"]
        if deepest < snapshot["passes"]:
            axis.axvspan(
                deepest,
                snapshot["passes"],
                color="#F2F4F6",
                alpha=0.82,
                linewidth=0,
                zorder=-2,
            )
        paired_any = sorted(
            set(domain_curves.get("control", {}))
            & set(domain_curves.get("replay", {}))
        )
        if paired_any:
            note = "paired seeds " + _compact_seeds(paired_any)
        else:
            control_seeds = sorted(domain_curves.get("control", {}))
            replay_seeds = sorted(domain_curves.get("replay", {}))
            if control_seeds and not replay_seeds:
                note = "control only " + _compact_seeds(control_seeds)
            elif replay_seeds and not control_seeds:
                note = "x-Mode Dr.GRPO only " + _compact_seeds(replay_seeds)
            else:
                note = "arms not paired yet"
        # With five arms the per-arm seed lists ran two lines each and collided
        # with the data. Report paired counts; the seed identities stay in the
        # provenance JSON, which is where anyone checking them would look.
        tags = {"semantic": "+MaxEnt", "semantic_only": "MaxEnt only",
                "adaptive_semantic": "adaptive"}
        for arm_name, summary in record["semantic_summary_by_arm"].items():
            tag = tags.get(arm_name, arm_name)
            if summary:
                deepest_key = max(summary, key=float)
                note += f"\n{tag} n={len(summary[deepest_key]['paired_seeds'])}"
            elif record["semantic_seeds_by_arm"].get(arm_name):
                note += f"\n{tag} unpaired"
        x, y, ha, va = _clearest_corner(axis, lines=note.count("\n") + 1)
        axis.text(
            x,
            y,
            note,
            transform=axis.transAxes,
            ha=ha,
            va=va,
            fontsize=style.SMALL_FONT,
            color=style.MUTED,
            zorder=6,
            # PantryPlan on Qwen has no clear corner -- every arm is high from
            # the first pass. A bare label there is unreadable over the curves.
            bbox=dict(
                facecolor=style.WHITE, edgecolor="none", alpha=0.72, pad=1.4
            ),
        )
    else:
        axis.axvspan(
            0,
            snapshot["passes"],
            color="#F2F4F6",
            alpha=0.82,
            linewidth=0,
            zorder=-2,
        )
        axis.text(
            0.5,
            0.52,
            "awaiting first\nregistered evaluation",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=style.SMALL_FONT,
            color=style.MUTED,
        )
    if row == total_rows - 1:
        axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)
    return record


def render(
    qwen: dict[str, Any],
    falcon: dict[str, Any],
    qwen3b: dict[str, Any],
    output: Path,
    *,
    frozen_provenance: dict[str, Any] | None = None,
) -> None:
    style.apply_rcparams()
    domains = list(qwen["domains"])
    snapshots = (qwen, falcon, qwen3b)
    figure, axes = plt.subplots(
        len(snapshots),
        len(domains),
        figsize=(style.WIDTH * 1.20, 6.45),
        sharex=True,
        sharey=True,
    )
    maximum = 1.0
    for snapshot in snapshots:
        for frozen in snapshot.get("frozen_domains", {}).values():
            for summary in frozen["paired_summary_by_pass"].values():
                maximum = max(maximum, *summary["control_range"], *summary["replay_range"])
            frozen_arms = dict(frozen.get("semantic_summary_by_arm") or {})
            if frozen.get("semantic_summary_by_pass"):
                frozen_arms.setdefault(
                    "semantic", frozen["semantic_summary_by_pass"]
                )
            for by_pass in frozen_arms.values():
                for summary in by_pass.values():
                    maximum = max(maximum, *summary["semantic_range"])
        for domain_curves in snapshot["curves"].values():
            for arm_curves in domain_curves.values():
                for curve in arm_curves.values():
                    maximum = max(maximum, *curve.values())
        for spec in _semantic_arms(snapshot):
            for seed_curves in spec["curves"].values():
                for curve in seed_curves.values():
                    maximum = max(maximum, *curve.values())
    y_high = max(2.0, math.ceil(maximum * 2) / 2 + 0.25)

    provenance: dict[str, Any] = frozen_provenance if frozen_provenance is not None else {
        "schema": "figure4_multimodel_interim_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "mean distinct correct@8; thick curves use paired seeds only",
        "families": {},
    }
    for row, (label, snapshot) in enumerate(zip(ROW_LABELS, snapshots)):
        family_record: dict[str, Any] = {
            "observed_runs": _observed_runs(snapshot),
            "total_runs": snapshot["total_runs"],
            "domains": {},
        }
        if _semantic_arms(snapshot):
            family_record["semantic_arms"] = [
                {
                    "arm": spec["arm"],
                    "comparator": spec["comparator"],
                    "observed_runs": int(spec["observed_runs"]),
                    "total_runs": int(spec["total_runs"]),
                    "coefficient": float(spec["coefficient"]),
                    # Records why a domain is absent from this arm rather than
                    # leaving a reader to guess it simply has no data yet.
                    "superseded_domains": list(spec.get("superseded_domains") or []),
                }
                for spec in _semantic_arms(snapshot)
            ]
        for column, domain in enumerate(domains):
            axis = axes[row][column]
            axis.set_ylim(0, y_high)
            family_record["domains"][domain] = _panel(
                axis,
                snapshot=snapshot,
                domain=domain,
                row=row,
                total_rows=len(snapshots),
            )
        axes[row][0].set_ylabel(
            f"{label}\nmean distinct correct@8",
            fontsize=style.LABEL_FONT,
        )
        provenance["families"][label] = family_record

    # Legend only names arms actually drawn, in a fixed order, so a rollout
    # that has one semantic arm and not the other cannot imply both.
    drawn_arms: list[str] = []
    for arm_name in (
        "bank_normalized_replay",
        "semantic",
        "semantic_only",
        "adaptive_semantic",
    ):
        if any(
            domain_record.get("semantic_summary_by_arm", {}).get(arm_name)
            for family in provenance["families"].values()
            for domain_record in family["domains"].values()
        ):
            drawn_arms.append(arm_name)

    # One coverage note per family and arm, so a Qwen-only or Falcon-only
    # rollout is never mistaken for both.
    semantic_parts = []
    coefficient = 0.10
    for label, snapshot in zip(ROW_LABELS, snapshots):
        for spec in _semantic_arms(snapshot):
            coefficient = float(spec["coefficient"])
            tag = {"semantic_only": "no replay",
                   "adaptive_semantic": "adaptive"}.get(spec["arm"], "with replay")
            semantic_parts.append(
                f"{label} {tag} {int(spec['observed_runs'])}/{int(spec['total_runs'])}"
            )
    if semantic_parts:
        half = (len(semantic_parts) + 1) // 2
        semantic_note = (
            f"semantic MaxEnt (eta = {coefficient:g}) cells with registered "
            "evaluations: " + "; ".join(semantic_parts[:half])
            + ("\n" + "; ".join(semantic_parts[half:]) if semantic_parts[half:] else "")
        )
    else:
        semantic_note = ""
    handles = []
    labels = []
    for arm in ("control", "replay", *drawn_arms):
        color, dash, label = live.ARM_STYLE[arm]
        handles.append(Line2D([0], [0], color=color, linestyle=dash, lw=1.4))
        labels.append(label)
    if frozen_provenance is None:
        handles.append(
            Line2D([0], [0], color=style.MUTED, lw=style.SEED_LW, alpha=0.55)
        )
        labels.append("available seed (thin)")
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=style.FONT,
        bbox_to_anchor=(0.5, -0.004),
        handlelength=2.7,
    )
    qwen_seen = _observed_runs(qwen)
    falcon_seen = _observed_runs(falcon)
    qwen3b_seen = _observed_runs(qwen3b)
    figure.suptitle(
        "INTERIM RESULTS — x-Mode Dr.GRPO across model scales",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.963,
        (
            f"Qwen-0.5B {qwen_seen}/{qwen['total_runs']}; "
            f"Falcon-1B {falcon_seen}/{falcon['total_runs']}; "
            f"Qwen-3B {qwen3b_seen}/{qwen3b['total_runs']} cells have "
            "registered evaluations; descriptive snapshot, final estimates pending"
            + (f"\n{semantic_note}" if semantic_note else "")
        ),
        ha="center",
        # A multi-line block anchored on "baseline" grows *upward* from its last
        # line, so the coverage note climbed into the title as arms were added.
        # "top" pins the block's top edge and grows it down into the gap.
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.079,
        right=0.995,
        # The coverage note gains a line per two semantic arms; give it room
        # rather than letting it crowd the first row of panels.
        top=0.862 if semantic_note else 0.89,
        bottom=0.10,
        hspace=0.20,
        wspace=0.20,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)
    if frozen_provenance is None:
        output.with_suffix(".json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frozen-json",
        type=Path,
        help="render the recorded paired summaries without rereading live runs",
    )
    args = parser.parse_args()
    if args.frozen_json is not None:
        provenance = json.loads(args.frozen_json.read_text(encoding="utf-8"))
        qwen, falcon, qwen3b = (
            _family_from_provenance(provenance, label) for label in ROW_LABELS
        )
        render(
            qwen,
            falcon,
            qwen3b,
            args.output.resolve(),
            frozen_provenance=provenance,
        )
    else:
        qwen = _family_snapshot(E78_LEDGER)
        falcon = _family_snapshot(E79_LEDGER)
        qwen3b = _family_snapshot(E80R1_LEDGER)
        # Every plotted semantic arm comes from the registry, so a launched arm
        # cannot be missing here without failing the registry test.
        for snapshot, family in (
            (qwen, "Qwen2.5-0.5B"),
            (falcon, "Falcon3-1B"),
            (qwen3b, "Qwen2.5-3B"),
        ):
            for arm in registry.semantic_arms(family):
                _attach_semantic(
                    snapshot,
                    arm.path(),
                    arm=str(arm.arm),
                    comparator=str(arm.comparator),
                    cohort=arm.tag,
                )
        render(qwen, falcon, qwen3b, args.output.resolve())
    print(
        f"wrote {args.output.with_suffix('.png')} and "
        f"{args.output.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
