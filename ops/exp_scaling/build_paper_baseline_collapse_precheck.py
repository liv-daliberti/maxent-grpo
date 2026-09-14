#!/usr/bin/env python3
"""Freeze the pass-0 and pass-8 problem precheck for both binary baselines.

The main-text precheck used to be three hand-copied Qwen2.5-0.5B rows. Reading
it as a figure means reading the same endpoints at all three scales for both
verifier-only objectives, so this builder computes every registered cell once
and freezes it with per-file provenance; the plotter never touches a run
directory.

Two facts about the design are load-bearing and are recorded rather than
assumed:

- Pass 0 is *measured per arm*, not shared. It is numerically shared wherever
  the two arms inherited the same evaluation requests, which is every
  Qwen2.5-0.5B and Falcon3-1B cell, and three of five Qwen2.5-3B domains. On
  Qwen2.5-3B MathIR and PantryPlan, E80-R1 seeds 71--74 carry marginally
  different pass-0 requests from the plain-GRPO arm, so a single shared origin
  would be a small fiction. ``pass0_arm_agreement`` records the comparison.
- ``extra_modes`` is ``distinct8 - pass8``: verified solution breadth beyond the
  first correct mode. Its pass-0 value is the denominator of every retention
  statement, so cells whose initial breadth sits under ``BREADTH_FLOOR`` are
  flagged: a ratio against .003 modes per prompt is arithmetic, not evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/results/baseline_collapse_precheck.json"
TABLE_OUTPUT = ROOT / "paper/results/baseline_collapse_precheck_table_body.tex"
EXPECTED_DRAWS = 4
EVALUATION_KIND = "fixed_seed_sampled_k_neutral"
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
# Each arm is one or more immutable released ledgers plus the arm key inside
# them. Qwen2.5-3B plain GRPO is E95 seed 70 plus the E114 seed extension; they
# are one arm by protocol and are combined by cell, never averaged separately.
ARM_SOURCES: dict[str, dict[str, Any]] = {
    "drgrpo": {
        "label": "Dr.GRPO",
        "objective": "reward-only Dr.GRPO (compute-matched control)",
        "scales": {
            "qwen05b": (("e78_verified_replay_only_05b_jobs.json", "control"),),
            "falcon1b": (
                ("e79_falcon1b_aligned_verified_replay_jobs.json", "control"),
            ),
            "qwen3b": (
                ("e80r1_qwen3b_aligned_verified_replay_jobs.json", "control"),
            ),
        },
    },
    "grpo": {
        "label": "GRPO",
        "objective": "plain group-relative GRPO (second control)",
        "scales": {
            "qwen05b": (("e95_plain_grpo_Qwen25-05B_jobs.json", "grpo_plain_control"),),
            "falcon1b": (("e95_plain_grpo_Falcon3-1B_jobs.json", "grpo_plain_control"),),
            "qwen3b": (
                ("e95_plain_grpo_Qwen25-3B_jobs.json", "grpo_plain_control"),
                ("e114_plain_grpo_qwen3b_extension_jobs.json", "grpo_plain_control"),
            ),
        },
    },
}
SCALE_MODEL = {
    "qwen05b": "Qwen2.5-0.5B",
    "falcon1b": "Falcon3-1B",
    "qwen3b": "Qwen2.5-3B",
}
SCALES = ("qwen05b", "falcon1b", "qwen3b")
METRIC_FIELDS = {
    "pass8": "any_correct_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
# Below this many extra verified modes per prompt at pass 0 there is no breadth
# to lose, and a retained fraction is not reportable. One extra mode found in
# one draw of one prompt moves the four-draw mean by 1/(128*4) = .002 modes per
# prompt, so this floor is about 25 resolution units. The registered data
# separates cleanly on either side of it: Graph and PantryPlan start between
# .105 and 1.525 extra modes per prompt at every scale, while Countdown,
# MathIR, and Python factors never exceed .011 --- roughly five observed extra
# modes in an entire evaluation. A percentage change against that denominator
# is arithmetic on noise, and at .01 the Qwen2.5-0.5B Countdown cell (.0109)
# would have squeaked past and printed a +54% "gain" in breadth built out of
# about six samples.
BREADTH_FLOOR = 0.05
T_CRIT_DF4 = 2.7764451051977987


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    return float(value)


def sampled_endpoint(
    run_dir: Path, *, step: int
) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    """Mean the four registered fixed-seed draws at one exact step."""

    records: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        payload = path.read_bytes()
        sources.append({
            "path": str(path.relative_to(ROOT)),
            "byte_length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        for raw in payload.splitlines():
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (
                row.get("evaluation_kind") != EVALUATION_KIND
                or row.get("step") != step
            ):
                continue
            draw = row.get("draw_index")
            metrics = row.get("metrics")
            if isinstance(draw, int) and isinstance(metrics, dict):
                records[draw] = metrics
    if sorted(records) != list(range(EXPECTED_DRAWS)):
        return None, sources
    endpoint = {
        name: statistics.fmean(
            finite(
                records[draw].get(field),
                name=f"{run_dir}:{step}:draw{draw}:{field}",
            )
            for draw in range(EXPECTED_DRAWS)
        )
        for name, field in METRIC_FIELDS.items()
    }
    endpoint["extra_modes"] = endpoint["distinct8"] - endpoint["pass8"]
    return endpoint, sources


def summarize(values: list[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mean": statistics.fmean(values),
        "n": len(values),
        "range": [min(values), max(values)],
    }
    if len(values) == 5:
        spread = statistics.stdev(values)
        half = T_CRIT_DF4 * spread / math.sqrt(len(values))
        summary["student_t_95"] = [summary["mean"] - half, summary["mean"] + half]
    return summary


def paired_change(
    per_seed: dict[str, dict[str, dict[str, float]]], metric: str
) -> dict[str, Any]:
    """Summarize the within-seed pass-8-minus-pass-0 change for one metric."""

    deltas = [
        record["pass8"][metric] - record["pass0"][metric]
        for record in per_seed.values()
    ]
    return summarize(deltas)


def build() -> dict[str, Any]:
    ledger_records: dict[str, Any] = {}
    telemetry: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}

    for arm_key, spec in ARM_SOURCES.items():
        scales: dict[str, Any] = {}
        for scale in SCALES:
            index: dict[tuple[str, int], dict[str, Any]] = {}
            target: int | None = None
            passes: int | None = None
            for ledger_name, arm_field in spec["scales"][scale]:
                path = ROOT / "var/artifacts" / ledger_name
                ledger = load(path)
                if ledger.get("released") is not True:
                    raise RuntimeError(f"{ledger_name}: ledger was not released")
                ledger_target = int(ledger["target_steps"])
                ledger_passes = int(ledger["passes"])
                if ledger_target != int(ledger["train_rows"]) * ledger_passes:
                    raise RuntimeError(f"{ledger_name}: target/pass geometry drifted")
                if target is None:
                    target, passes = ledger_target, ledger_passes
                elif (target, passes) != (ledger_target, ledger_passes):
                    raise RuntimeError(
                        f"{ledger_name}: pass geometry differs from its co-arm ledger"
                    )
                ledger_records[ledger_name] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "arm": arm_key,
                    "scale": scale,
                    "arm_field": arm_field,
                }
                for run in ledger.get("runs", []):
                    if (
                        str(run.get("arm")) != arm_field
                        or str(run.get("domain")) not in DOMAINS
                    ):
                        continue
                    cell = (str(run["domain"]), int(run["seed"]))
                    if cell in index:
                        raise RuntimeError(
                            f"{arm_key}/{scale}: {cell} is registered twice"
                        )
                    index[cell] = run

            domains: dict[str, Any] = {}
            for domain in DOMAINS:
                per_seed: dict[str, dict[str, dict[str, float]]] = {}
                for (run_domain, seed), run in sorted(index.items()):
                    if run_domain != domain:
                        continue
                    run_dir = Path(str(run["run_dir"]))
                    origin, origin_sources = sampled_endpoint(run_dir, step=0)
                    terminal, terminal_sources = sampled_endpoint(
                        run_dir, step=target
                    )
                    telemetry.extend({
                        **source, "arm": arm_key, "scale": scale,
                        "domain": domain, "seed": seed,
                    } for source in origin_sources)
                    if origin is None or terminal is None:
                        continue
                    per_seed[str(seed)] = {"pass0": origin, "pass8": terminal}
                if not per_seed:
                    continue
                domains[domain] = {
                    "per_seed": per_seed,
                    "endpoints": {
                        endpoint: {
                            metric: summarize(
                                [record[endpoint][metric] for record in per_seed.values()]
                            )
                            for metric in ("pass8", "distinct8", "extra_modes")
                        }
                        for endpoint in ("pass0", "pass8")
                    },
                    "paired_change": {
                        metric: paired_change(per_seed, metric)
                        for metric in ("pass8", "distinct8", "extra_modes")
                    },
                }
            scales[scale] = {
                "model": SCALE_MODEL[scale],
                "target_step": target,
                "training_pass": passes,
                "domains": domains,
                "macro": macro_summary(domains),
            }
        arms[arm_key] = {
            "label": spec["label"],
            "objective": spec["objective"],
            "scales": scales,
        }

    payload: dict[str, Any] = {
        "schema": "paper-baseline-collapse-precheck-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": (
            "does a verifier-only binary objective raise correctness while "
            "removing verified solution breadth beyond the first correct mode?"
        ),
        "selection_rule": (
            "every registered domain/seed with all four fixed-seed sampled-K "
            "draws at both step 0 and the exact pass-8 target; arms are read "
            "independently and pass 0 is measured per arm"
        ),
        "expected_draws": EXPECTED_DRAWS,
        "evaluation_kind": EVALUATION_KIND,
        "breadth_floor": BREADTH_FLOOR,
        "breadth_floor_note": (
            "extra-modes retention is reported only where pass-0 extra modes "
            "exceed this floor; below it the denominator is at the four-draw "
            "measurement resolution"
        ),
        "domains": list(DOMAINS),
        "scales": list(SCALES),
        "ledgers": ledger_records,
        "arms": arms,
    }
    payload["pass0_arm_agreement"] = pass0_agreement(arms)
    payload["telemetry_sources"] = telemetry
    return payload


def macro_summary(domains: dict[str, Any]) -> dict[str, Any]:
    """Average the five domain means, then read retention off that average.

    Macro-averaging over domains rather than pooling seeds is what the
    main-text precheck reports, and it is the right weighting here: the five
    domains have very different initial breadth, and seed-pooling would let
    PantryPlan alone decide the number.
    """

    if len(domains) != len(DOMAINS):
        return {"complete": False, "domains_present": sorted(domains)}
    macro: dict[str, Any] = {"complete": True, "domains_present": list(DOMAINS)}
    for endpoint in ("pass0", "pass8"):
        macro[endpoint] = {
            metric: statistics.fmean(
                domains[domain]["endpoints"][endpoint][metric]["mean"]
                for domain in DOMAINS
            )
            for metric in ("pass8", "distinct8", "extra_modes")
        }
    macro["change"] = {
        metric: macro["pass8"][metric] - macro["pass0"][metric]
        for metric in ("pass8", "distinct8", "extra_modes")
    }
    macro["extra_modes_retained_fraction"] = (
        macro["pass8"]["extra_modes"] / macro["pass0"]["extra_modes"]
        if macro["pass0"]["extra_modes"] > BREADTH_FLOOR
        else None
    )
    return macro


def retained_fraction(cell: dict[str, Any]) -> float | None:
    """Terminal extra modes as a fraction of initial extra modes, or None."""

    origin = cell["endpoints"]["pass0"]["extra_modes"]["mean"]
    if origin <= BREADTH_FLOOR:
        return None
    return cell["endpoints"]["pass8"]["extra_modes"]["mean"] / origin


def pass0_agreement(arms: dict[str, Any]) -> dict[str, Any]:
    """Record, per scale and domain, whether the two arms share pass 0 exactly."""

    agreement: dict[str, Any] = {}
    for scale in SCALES:
        by_domain: dict[str, Any] = {}
        for domain in DOMAINS:
            left = arms["drgrpo"]["scales"][scale]["domains"].get(domain)
            right = arms["grpo"]["scales"][scale]["domains"].get(domain)
            if left is None or right is None:
                by_domain[domain] = {"comparable": False}
                continue
            shared_seeds = sorted(set(left["per_seed"]) & set(right["per_seed"]))
            mismatches = [
                {
                    "seed": int(seed),
                    "metric": metric,
                    "drgrpo": left["per_seed"][seed]["pass0"][metric],
                    "grpo": right["per_seed"][seed]["pass0"][metric],
                }
                for seed in shared_seeds
                for metric in ("pass8", "distinct8")
                if left["per_seed"][seed]["pass0"][metric]
                != right["per_seed"][seed]["pass0"][metric]
            ]
            by_domain[domain] = {
                "comparable": True,
                "shared_seeds": [int(seed) for seed in shared_seeds],
                "identical": not mismatches,
                "mismatches": mismatches,
            }
        agreement[scale] = by_domain
    return agreement


def table_body(payload: dict[str, Any]) -> str:
    """One macro row per scale, both arms, for the appendix provenance table."""

    lines: list[str] = []
    for scale in SCALES:
        cells: list[str] = []
        for arm in ("drgrpo", "grpo"):
            macro = payload["arms"][arm]["scales"][scale]["macro"]
            if not macro.get("complete"):
                cells.append(r"\multicolumn{1}{c}{--}")
                continue
            retained = macro["extra_modes_retained_fraction"]
            removed = "--" if retained is None else f"{100 * (1 - retained):.1f}\\%"
            cells.append(
                f"\\({macro['pass0']['pass8']:.3f} \\to {macro['pass8']['pass8']:.3f}\\) & "
                f"\\({macro['pass0']['extra_modes']:.3f} \\to "
                f"{macro['pass8']['extra_modes']:.3f}\\) & {removed}"
            )
        lines.append(f"{SCALE_MODEL[scale]} & " + " & ".join(cells) + r" \\")
    # The rule belongs to the generated body, not to main.tex: \input leaves a
    # space token at end of file, and a space after the final \\ opens a new
    # row, so a \bottomrule written after the \input lands inside a cell.
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    TABLE_OUTPUT.write_text(table_body(payload), encoding="utf-8")
    for arm in ("drgrpo", "grpo"):
        for scale in SCALES:
            block = payload["arms"][arm]["scales"][scale]
            macro = block["macro"]
            cells = sum(
                len(cell["per_seed"]) for cell in block["domains"].values()
            )
            if not macro.get("complete"):
                print(f"{arm:7} {scale:9} incomplete: {macro['domains_present']}")
                continue
            retained = macro["extra_modes_retained_fraction"]
            print(
                f"{arm:7} {scale:9} cells {cells:3d}  "
                f"pass@8 {macro['pass0']['pass8']:.3f}->{macro['pass8']['pass8']:.3f}  "
                f"distinct@8 {macro['pass0']['distinct8']:.3f}->{macro['pass8']['distinct8']:.3f}  "
                f"extra {macro['pass0']['extra_modes']:.3f}->{macro['pass8']['extra_modes']:.3f}  "
                f"removed {'--' if retained is None else format(100 * (1 - retained), '.1f') + '%'}"
            )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"wrote {TABLE_OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
