#!/usr/bin/env python3
"""Render the current E90 replay-dose evidence with exact paired subsets.

The live comparison preview can mix different seed sets across methods.  That is
useful operationally but is not the right paper estimand.  This renderer selects
every terminal Adaptive Re:Dr seed at render time and restricts Dr.GRPO,
Re:Dr, and Adaptive Re:Dr to that same within-domain subset.
Complete five-seed domains and exact terminal prefixes therefore coexist without
changing n along a trajectory or freezing an obsolete dated slice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_matrix as matrix  # noqa: E402
import plot_paper_comparison_families as families  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "paper/figures/comparisons"
SCALE = families.SCALE_BY_KEY["qwen05b"]
COMPARISON = families.COMPARISON_BY_KEY["replay_dose"]
REQUIRED_METHODS = ("drgrpo", "replay_grpo", "adaptive_replay_grpo")


def _terminal_seeds_by_domain(
    cells: dict[matrix.CellKey, matrix.Cell],
) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for domain in families.STATIC_DOMAIN_COLUMNS:
        seeds = []
        for seed in matrix.SCALE_BY_KEY[SCALE.key].seeds:
            method_cells = []
            for method in REQUIRED_METHODS:
                key = matrix.CellKey(method, SCALE.key, domain, seed)
                cell = cells.get(key)
                method_cells.append(cell)
            if all(
                cell is not None and cell.status == "terminal"
                for cell in method_cells
            ):
                seeds.append(seed)
        if not seeds:
            raise RuntimeError(f"{domain}: no terminal paired replay-dose seed")
        selected[domain] = seeds
    return selected


def _restrict_snapshot(
    snapshot: dict[str, Any],
    domains: list[str],
    selected_seeds: dict[str, list[int]],
) -> None:
    for domain in domains:
        selected = set(selected_seeds[domain])
        for arm in ("control", "replay"):
            curves = snapshot["curves"][domain][arm]
            snapshot["curves"][domain][arm] = {
                seed: curve for seed, curve in curves.items() if seed in selected
            }
            if set(snapshot["curves"][domain][arm]) != selected:
                raise RuntimeError(
                    f"{domain}/{arm}: missing a frozen paired trajectory"
                )
        for spec in snapshot.get("semantic_arms", []):
            curves = spec["curves"].get(domain, {})
            spec["curves"][domain] = {
                seed: curve for seed, curve in curves.items() if seed in selected
            }
            if set(spec["curves"][domain]) != selected:
                raise RuntimeError(
                    f"{domain}/{spec['arm']}: missing a frozen paired trajectory"
                )


def _amend_provenance(
    path: Path,
    domains: list[str],
    selected_seeds: dict[str, list[int]],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema": "paper-comparison-progress-figure-v2",
            "snapshot_at": datetime.now(timezone.utc).isoformat(),
            "registered_seed_block": [43, 44, 45, 46, 47],
            "paired_terminal_seeds_by_domain": {
                domain: selected_seeds[domain] for domain in domains
            },
            "five_seed_estimand_complete_by_domain": {
                domain: len(selected_seeds[domain]) == 5 for domain in domains
            },
            "selection_rule": (
                "all E90 adaptive-replay cells terminal at render time; every "
                "displayed arm is restricted to that domain's exact terminal "
                "paired subset"
            ),
        }
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def render(output_dir: Path) -> list[Path]:
    cells = matrix.load_cells()
    selected_seeds = _terminal_seeds_by_domain(cells)
    domains = list(families.STATIC_DOMAIN_COLUMNS)
    outputs: list[Path] = []
    for part, start in enumerate(range(0, len(domains), 3), start=1):
        chunk = domains[start : start + 3]
        snapshot = families.build_snapshot(SCALE, COMPARISON, chunk)
        _restrict_snapshot(snapshot, chunk, selected_seeds)
        output = output_dir / f"replay_dose_qwen05b_progress_part{part}"
        families.render_chunk(
            snapshot,
            SCALE,
            COMPARISON,
            chunk,
            output,
            evidence="progress",
        )
        _amend_provenance(output.with_suffix(".json"), chunk, selected_seeds)
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for output in render(args.output_dir.resolve()):
        print(f"wrote {output.with_suffix('.pdf')}")
        print(f"wrote {output.with_suffix('.png')}")
        print(f"wrote {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
