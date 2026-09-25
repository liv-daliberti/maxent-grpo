#!/usr/bin/env python3
"""Freeze the enumerated-support index GAPO reads at training time.

GAPO's reward needs ``L``, the size of a prompt's valid response set. ModeBench
certifies that number as the ``answer_mode_count`` column, which is why this
comparison can run faithfully here rather than on an estimated support.

The index is keyed by a digest of the prompt's reference payload, not by an
instance id, for two reasons. The reference is what the learner already holds
for every row, so no dataset column has to be plumbed through the rollout path.
And the support's counterpart *inside* the reference payload is not uniform
across domains --- PythonFactors carries a ``num_modes`` of 384 against a
certified support of 32 --- so parsing the payload would quietly produce the
wrong ``L`` on one domain and the right one everywhere else.

Both splits are indexed. Only the training split can reach GAPO's reward path,
but a lookup miss is a hard failure by design, and indexing the evaluation
split costs nothing while removing one way to fail for an uninteresting reason.

Run this once, before submission. The cohort ledger pins the result by digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "gapo_support_index_v1"

#: The Level-1 Qwen2.5-0.5B panel, with the dataset roots the E78 controls used.
DOMAIN_ROOTS = {
    "countdown": "var/data/exact_countdown_easy3_probe",
    "graph_coloring": "var/data/graph_coloring_modebench_v2",
    "mathir": "var/data/mathir_action_menu_v1",
    "pantry_plan": "var/data/pantry_plan_modebench_v2",
    "python_factors": "var/data/python_factor_modebench_v1",
}
SUPPORT_COLUMN = "answer_mode_count"
REFERENCE_COLUMN = "answer"


def reference_key(reference: object) -> str:
    return hashlib.sha256(str(reference).encode("utf-8")).hexdigest()


def _rows(path: Path) -> list[Any]:
    """Return every sub-split under one dataset root.

    Evaluation roots carry both a ``multi_answer`` and a ``unique_answer``
    split; training roots carry one. Both are indexed, so the caller never has
    to know which shape a domain uses.
    """

    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    parts = (
        [dataset[name] for name in dataset.keys()]
        if hasattr(dataset, "keys")
        else [dataset]
    )
    for part in parts:
        for column in (REFERENCE_COLUMN, SUPPORT_COLUMN):
            if column not in part.column_names:
                raise SystemExit(f"{path}: missing required column {column!r}")
    return parts


def build(root: Path) -> dict[str, Any]:
    sizes: dict[str, int] = {}
    per_domain: dict[str, Any] = {}
    for domain, relative in sorted(DOMAIN_ROOTS.items()):
        domain_root = root / relative
        domain_sizes: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        observed: list[int] = []
        for split in ("train", "eval"):
            split_path = domain_root / split
            if not split_path.exists():
                raise SystemExit(f"{domain}: absent split {split_path}")
            references: list[str] = []
            supports: list[int] = []
            for part in _rows(split_path):
                references.extend(str(value) for value in part[REFERENCE_COLUMN])
                supports.extend(int(value) for value in part[SUPPORT_COLUMN])
            if any(value <= 0 for value in supports):
                raise SystemExit(f"{domain}/{split}: non-positive support size")
            for reference, support in zip(references, supports):
                key = reference_key(reference)
                previous = domain_sizes.get(key)
                if previous is not None and previous != support:
                    raise SystemExit(
                        f"{domain}/{split}: reference maps to two support "
                        f"sizes ({previous} and {support})"
                    )
                domain_sizes[key] = support
            split_counts[split] = len(references)
            observed.extend(supports)
        overlap = sorted(set(domain_sizes) & set(sizes))
        if overlap:
            raise SystemExit(
                f"{domain}: {len(overlap)} references collide with another domain"
            )
        sizes.update(domain_sizes)
        per_domain[domain] = {
            "dataset_root": relative,
            "prompts": split_counts,
            "distinct_references": len(domain_sizes),
            "support_min": min(observed),
            "support_max": max(observed),
            "support_mean": round(sum(observed) / len(observed), 4),
        }
    return {
        "schema": SCHEMA,
        "support_sizes": sizes,
        "domains": per_domain,
        "support_column": SUPPORT_COLUMN,
        "reference_column": REFERENCE_COLUMN,
        "total_prompts": len(sizes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "var/artifacts/gapo_support_index.json",
        help="destination for the frozen index",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite a frozen index: {args.out}")

    payload = build(ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    for domain, summary in sorted(payload["domains"].items()):
        print(
            f"[gapo] {domain:16s} prompts={summary['prompts']} "
            f"L in [{summary['support_min']}, {summary['support_max']}] "
            f"mean={summary['support_mean']}"
        )
    print(f"[gapo] {payload['total_prompts']} prompts -> {args.out}")
    print(f"[gapo] sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
