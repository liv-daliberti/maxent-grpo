#!/usr/bin/env python3
"""Replay the source, support, and three-way split audit for PantryPlan v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from datasets import load_from_disk


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1]))
OPS = Path(os.environ.get("OAT_ZERO_OPS_ROOT", ROOT / "ops"))
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_pantry_plan_mode_data import (  # noqa: E402
    _audit_manual_review,
    _audit_rows,
    _canonical_sha256,
    _sha256_file,
)
from curate_pantry_plan_source import ARCHIVE_SHA256, build_curation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "var/data/pantry_plan_modebench_v2",
    )
    parser.add_argument(
        "--curation",
        type=Path,
        default=ROOT / "var/data/pantry_plan_v1/ingredients.json",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            ROOT
            / "var/source_data/usda_fdc_foundation_2026_04_30"
            / "FoodData_Central_foundation_food_json_2026-04-30.json"
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=(
            ROOT
            / "var/source_data/usda_fdc_foundation_2026_04_30"
            / "FoodData_Central_foundation_food_json_2026-04-30.zip"
        ),
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        default=ROOT / "var/artifacts/pantry_plan_v1_manual_source_review.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "var/artifacts/pantry_plan_modebench_v2_admission_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh PantryPlan v2 audit required: {args.output}")
    identity = json.loads((args.data_root / "identity.json").read_text())
    if identity.get("schema") != "pantry_plan_modebench_v2":
        raise ValueError("PantryPlan v2 identity schema mismatch")
    curation = json.loads(args.curation.read_text())
    review = json.loads(args.manual_review.read_text())
    archive_sha256 = _sha256_file(args.archive)
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError("USDA source archive hash mismatch")
    replayed_curation = build_curation(
        json.loads(args.source.read_text()), archive_sha256
    )
    for key in (
        "source",
        "nutrient_precedence",
        "ingredient_count",
        "ingredients",
        "ingredient_table_sha256",
    ):
        if replayed_curation[key] != curation[key]:
            raise ValueError(f"curation replay differs at {key}")
    review_sha256 = _audit_manual_review(review, curation)

    split_audits = {}
    fingerprints = {}
    expected_dataset_splits = {
        "train": "train",
        "dev": "multi_answer",
        "eval": "multi_answer",
    }
    for split, dataset_split in expected_dataset_splits.items():
        dataset = load_from_disk(str(args.data_root / split))
        if set(dataset) != {dataset_split}:
            raise ValueError(f"{split} dataset names differ from contract")
        audit = _audit_rows(
            dataset[dataset_split].to_list(),
            split=split,
            ingredient_table_sha256=curation["ingredient_table_sha256"],
        )
        fingerprints[split] = set(audit.pop("fingerprints"))
        split_audits[split] = audit
    overlap = (
        fingerprints["train"] & fingerprints["dev"]
        or fingerprints["train"] & fingerprints["eval"]
        or fingerprints["dev"] & fingerprints["eval"]
    )
    if overlap:
        raise ValueError("PantryPlan v2 specifications overlap across splits")

    expected_identity = {
        "split_rows": {
            split: split_audits[split]["rows"] for split in split_audits
        },
        "split_rows_sha256": {
            split: split_audits[split]["rows_sha256"] for split in split_audits
        },
        "family_counts": {
            split: split_audits[split]["family_counts"] for split in split_audits
        },
        "minimum_exact_support_count": min(
            audit["minimum_exact_support_count"]
            for audit in split_audits.values()
        ),
        "maximum_exact_support_count": max(
            audit["maximum_exact_support_count"]
            for audit in split_audits.values()
        ),
        "ingredient_curation_sha256": _canonical_sha256(curation),
        "ingredient_table_sha256": curation["ingredient_table_sha256"],
        "source_archive_sha256": archive_sha256,
        "split_overlap_count": 0,
        "v1_rows_copied": 0,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"dataset identity mismatch at {key}")
    if identity.get("split_rows") != {"train": 384, "dev": 64, "eval": 128}:
        raise ValueError("PantryPlan v2 row counts differ from preregistration")

    payload = {
        "schema_version": "pantry-plan-modebench-admission-audit-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": "admitted_for_0.5b_development_viability_sampling",
        "support_semantics": "exact_ingredient_support",
        "source_replay": {
            "archive_sha256": archive_sha256,
            "curation_sha256": _canonical_sha256(curation),
            "ingredient_table_sha256": curation["ingredient_table_sha256"],
            "manual_review_sha256": review_sha256,
        },
        "splits": split_audits,
        "split_overlap_count": 0,
        "v1_rows_copied": 0,
        "dataset_identity_sha256": _canonical_sha256(identity),
        "audit_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        "[pantry-audit-v2] status=pass "
        f"train={split_audits['train']['rows']} "
        f"dev={split_audits['dev']['rows']} "
        f"eval={split_audits['eval']['rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
