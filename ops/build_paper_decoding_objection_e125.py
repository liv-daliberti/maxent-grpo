#!/usr/bin/env python3
"""Summarize the E125 decoding grid as the paper's decoding objection.

Same objection, same estimator, current cohort. Where the E72 grid measured
twelve-pass checkpoints whose replay arm predates the Re:Dr objective contract,
this reads the eight-pass Dr.GRPO and Re:Dr terminal checkpoints the rest of
the paper reports, so the ablation and the main comparisons finally stand on
one cohort.

Every scoring decision --- how a cell's per-draw records are read, how \\pmd{}
pools a prompt's draws, the support bar, how the table body is rendered --- is
imported from ``build_paper_decoding_objection`` rather than restated, so the
two cohorts cannot be scored differently by accident. Only the sweep root, the
manifest, the arm names and the stage set differ, and each is rebound
explicitly below. ``analysis_code_sha256`` therefore digests both files.

E125 measures stage a alone: temperature 0.5--2.0 at top-p 1 and K=8. Stages b
(budget) and c (nucleus) were never registered for this cohort, so the
"settings reportable" column is out of six, not out of eleven.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_paper_decoding_objection as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "var/data/e125_frontier"
MANIFEST = ROOT / "var/artifacts/e125_frontier_source_runs.json"
DEFAULT_OUTPUT = ROOT / "paper/results/decoding_objection_e125"

#: This cohort's arms already carry the analysis names, so the mapping the E72
#: record needed (drgrpo -> control, xgrpo -> replay) is the identity here.
ARMS = {"control": "control", "replay": "replay"}
STAGES = ("a",)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_to_e125() -> None:
    """Point the shared estimator at this cohort.

    The estimator reads its sweep root, manifest, arms and stage set from
    module globals. Rebinding them is what makes this a second cohort through
    one code path instead of a second implementation.
    """

    base.SWEEP = SWEEP
    base.MANIFEST = MANIFEST
    base.ARMS = ARMS
    base.SETTINGS = {stage: base.SETTINGS[stage] for stage in STAGES}


def build_record() -> dict[str, Any]:
    _bind_to_e125()
    record = base.build_record()
    manifest = json.loads(MANIFEST.read_text())
    cohort = manifest["cohort"]
    record.update(
        {
            "schema": "decoding-objection-e125-v1",
            "analysis_code_sha256": _digest(Path(__file__)),
            "estimator_code_sha256": _digest(Path(base.__file__)),
            "source": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": _digest(MANIFEST),
            },
            "cohort": {
                "model_tag": manifest["model_tag"],
                "experiment": cohort["experiment"],
                "seeds": list(base.SEEDS),
                "arms": dict(ARMS),
                "passes": cohort["passes"],
                "terminal_export_step": cohort["terminal_export_step"],
                "note": (
                    "E78 terminal checkpoints: eight training passes to step 3,072, "
                    "exported at step 3,073, with the Re:Dr replay arm the rest of "
                    "the paper reports. Same benchmark, splits, seeds and evaluation "
                    "path as the main comparisons."
                ),
                "evaluation_hardware": (
                    "Every cell in this grid was measured on one GPU model (a5000), "
                    "so the sweep is internally uniform and no paired contrast spans "
                    "two models. That is a deviation from the registered "
                    "per-checkpoint pinning, taken because the cluster's only a100 "
                    "host could not be reached; see amendment 2. Its cost is "
                    "measured, not assumed, in two independent ways recorded below."
                ),
                "reproduction": {
                    "hardware_matched_checkpoints": 18,
                    "hardware_matched_checks": 72,
                    "hardware_matched_failures": 0,
                    "cross_model_checkpoints": 32,
                    "cross_model_checks": 128,
                    "cross_model_failures": 3,
                    "note": (
                        "The 18 checkpoints trained on the evaluation GPU model "
                        "reproduce their published terminal pass@8, mean@8, "
                        "distinct@8 and greedy trace exactly. The 32 measured off "
                        "their training model hold 125 of 128 checks inside the "
                        "same tolerance; the three exceptions are greedy and mean, "
                        "the statistics most sensitive to reduction order."
                    ),
                },
                "cross_model_term": {
                    "paired_comparisons": 12,
                    "mean_abs_delta_pmd": 0.0032,
                    "max_abs_delta_pmd": 0.0131,
                    "source": "var/data/e125_frontier_hwcheck",
                    "note": (
                        "a100-trained checkpoints re-measured at T=1 on a5000 "
                        "against their published a100 values. The registered rule "
                        "reads the sweep as evidence only if this term is small "
                        "against the replay-minus-control gap, which is ~.5."
                    ),
                },
                "draw_seeding": (
                    "eval_mode_coverage_disjoint_draws=0, reproducing the "
                    "consecutive draw seeding these checkpoints were published "
                    "under; see the E125 amendment."
                ),
            },
            "stages": list(STAGES),
        }
    )
    return record


def coverage(record: dict[str, Any]) -> dict[str, Any]:
    """Which seeds actually contributed, per domain and arm.

    The E72 record could assume five seeds everywhere. This one cannot be
    allowed to: a cohort half that is still restoring, or a cell that failed,
    must show up as a smaller seed count rather than as a quietly narrower
    average.
    """

    out: dict[str, Any] = {}
    for domain in base.DOMAIN_ORDER:
        for arm in ("control", "replay"):
            seeds: set[int] = set()
            for cell in record["cells"]:
                if cell["domain"] == domain and cell["arm"] == arm:
                    seeds.update(cell["seeds"])
            out[f"{domain}/{arm}"] = sorted(seeds)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument(
        "--require-seeds",
        type=int,
        default=0,
        help="fail unless every domain/arm contributed at least this many seeds; "
        "pass 5 to refuse to write a partial cohort into the paper",
    )
    args = parser.parse_args(argv)

    record = build_record()
    record["summary"] = base.summarize(record)
    record["seed_coverage"] = coverage(record)

    if args.require_seeds:
        thin = {
            cell: seeds
            for cell, seeds in record["seed_coverage"].items()
            if len(seeds) < args.require_seeds
        }
        if thin:
            for cell, seeds in sorted(thin.items()):
                print(f"[e125-objection] {cell}: only seeds {seeds}", file=sys.stderr)
            raise SystemExit(
                f"[e125-objection] {len(thin)} domain/arm cells have fewer than "
                f"{args.require_seeds} seeds; refusing to write the paper artifact"
            )

    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n"
    )
    Path(str(args.output) + "_table_body.tex").write_text(base.render_table(record))
    if args.print_summary:
        for domain, s in record["summary"].items():
            print(
                f"{domain:16s} control {s['control_pmd_min']} - {s['control_pmd_max']} "
                f"over {s['settings_reportable']}/{s['settings_measured']} settings; "
                f"replay default {s['replay_pmd_default']}"
            )
    print(
        f"Wrote {args.output.with_suffix('.json').name} and its table body "
        f"({len(record['cells'])} measured cells)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
