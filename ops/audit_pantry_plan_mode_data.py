#!/usr/bin/env python3
"""Replay the full PantryPlan source, split, verifier, and mode-count audit."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OPS = ROOT / "ops"
for path in (SRC, OPS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from curate_pantry_plan_source import (  # noqa: E402
    ARCHIVE_SHA256,
    SELECTED,
    build_curation,
)
from make_pantry_plan_mode_data import enumerate_pantry_supports  # noqa: E402
from oat_drgrpo.pantry_plan import parse_pantry_plan_spec  # noqa: E402


DEFAULT_DATA = ROOT / "var/data/pantry_plan_modebench_v1"
DEFAULT_CURATION = ROOT / "var/data/pantry_plan_v1/ingredients.json"
DEFAULT_SOURCE = (
    ROOT
    / "var/source_data/usda_fdc_foundation_2026_04_30"
    / "FoodData_Central_foundation_food_json_2026-04-30.json"
)
DEFAULT_ARCHIVE = DEFAULT_SOURCE.with_suffix(".zip")
DEFAULT_REVIEW = ROOT / "var/artifacts/pantry_plan_v1_manual_source_review.json"
DEFAULT_OUTPUT = ROOT / "var/artifacts/pantry_plan_modebench_v1_admission_audit.json"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(data_root: Path) -> tuple[list[dict], list[dict]]:
    train = load_from_disk(str(data_root / "train"))
    evaluation = load_from_disk(str(data_root / "eval"))
    if set(train) != {"train"} or set(evaluation) != {"multi_answer"}:
        raise ValueError("PantryPlan split names differ from the frozen contract")
    return train["train"].to_list(), evaluation["multi_answer"].to_list()


def _audit_manual_review(
    review: Mapping[str, Any], curation: Mapping[str, Any]
) -> str:
    if review.get("schema_version") != "pantry-plan-manual-source-review-v1":
        raise ValueError("manual source review schema mismatch")
    if review.get("status") != "complete":
        raise ValueError("manual source review is not complete")
    if review.get("decision") != "accepted_for_benchmark_generation":
        raise ValueError("manual source review did not admit the table")
    if review.get("curation_sha256") != _canonical_sha256(curation):
        raise ValueError("manual review does not bind the frozen curation")
    if review.get("ingredient_table_sha256") != curation["ingredient_table_sha256"]:
        raise ValueError("manual review ingredient-table hash mismatch")
    reviewed = review.get("reviewed_ingredients")
    if not isinstance(reviewed, list):
        raise ValueError("manual review lacks reviewed ingredients")
    expected = [
        {
            "id": row["id"],
            "fdc_id": row["fdc_id"],
            "description": row["description"],
            "tags": row["tags"],
        }
        for row in curation["ingredients"]
    ]
    if reviewed != expected:
        raise ValueError("manual reviewed ingredient ledger differs from curation")
    if review.get("allowed_benchmark_tags") != ["peanut", "tree_nut"]:
        raise ValueError("manual review tag vocabulary mismatch")
    if not str(review.get("safety_boundary", "")).strip():
        raise ValueError("manual review omits its safety boundary")
    return _canonical_sha256(review)


def _audit_rows(
    rows: list[dict],
    *,
    split: str,
    ingredient_table_sha256: str,
) -> dict[str, Any]:
    fingerprints: set[str] = set()
    families: Counter[str] = Counter()
    mode_counts: list[int] = []
    instance_ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("answer_mode_split") != split:
            raise ValueError(f"{split} row {index} has wrong split marker")
        prompt = str(row.get("problem", ""))
        if any(
            hidden in prompt
            for hidden in (
                "certified_mode_count",
                "certified_support_sha256",
                "instance_fingerprint",
            )
        ):
            raise ValueError(f"{split} row {index} leaks certification metadata")
        spec = json.loads(row["answer"])
        parse_pantry_plan_spec(spec)
        if spec.get("source_ingredient_table_sha256") != ingredient_table_sha256:
            raise ValueError(f"{split} row {index} ingredient hash mismatch")
        instance_id = str(spec.get("instance_id", ""))
        if not instance_id:
            raise ValueError(f"{split} row {index} lacks an instance ID")
        if instance_id in instance_ids:
            raise ValueError(f"{split} instance IDs are not unique")
        instance_ids.add(instance_id)

        fingerprint_payload = dict(spec)
        fingerprint_payload.pop("instance_id", None)
        fingerprint = _canonical_sha256(fingerprint_payload)
        if row.get("instance_fingerprint") != fingerprint:
            raise ValueError(f"{split} row {index} fingerprint mismatch")
        if fingerprint in fingerprints:
            raise ValueError(f"{split} contains duplicate specifications")
        fingerprints.add(fingerprint)

        supports = enumerate_pantry_supports(spec)
        support_keys = ["+".join(support) for support in sorted(supports)]
        support_sha256 = hashlib.sha256(
            "\n".join(support_keys).encode("utf-8")
        ).hexdigest()
        if len(supports) != int(row["answer_mode_count"]):
            raise ValueError(f"{split} row {index} mode count failed replay")
        if len(supports) != int(spec["certified_mode_count"]):
            raise ValueError(f"{split} row {index} spec count failed replay")
        if support_sha256 != spec["certified_support_sha256"]:
            raise ValueError(f"{split} row {index} support hash failed replay")
        if not 8 <= len(supports) <= 64:
            raise ValueError(f"{split} row {index} violates the mode-count gate")
        if not all(math.isfinite(float(value)) for value in mode_counts + [len(supports)]):
            raise ValueError("nonfinite mode count")

        family = str(row["answer_mode_family"])
        if spec.get("family") != family:
            raise ValueError(f"{split} row {index} family mismatch")
        families[family] += 1
        mode_counts.append(len(supports))
    return {
        "rows": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "fingerprints": sorted(fingerprints),
        "family_counts": dict(sorted(families.items())),
        "minimum_exact_support_count": min(mode_counts),
        "maximum_exact_support_count": max(mode_counts),
        "total_exact_supports": sum(mode_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = json.loads((args.data_root / "identity.json").read_text())
    curation = json.loads(args.curation.read_text())
    review = json.loads(args.manual_review.read_text())
    archive_sha256 = _sha256_file(args.archive)
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError("USDA source archive hash mismatch")
    replayed_curation = build_curation(json.loads(args.source.read_text()), archive_sha256)
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

    train_rows, eval_rows = _load_rows(args.data_root)
    train = _audit_rows(
        train_rows,
        split="train",
        ingredient_table_sha256=curation["ingredient_table_sha256"],
    )
    evaluation = _audit_rows(
        eval_rows,
        split="eval",
        ingredient_table_sha256=curation["ingredient_table_sha256"],
    )
    overlap = set(train.pop("fingerprints")) & set(evaluation.pop("fingerprints"))
    if overlap:
        raise ValueError("PantryPlan train/eval specifications overlap")
    expected_identity = {
        "train_rows": train["rows"],
        "eval_rows": evaluation["rows"],
        "train_rows_sha256": train["rows_sha256"],
        "eval_rows_sha256": evaluation["rows_sha256"],
        "family_counts": {
            "train": train["family_counts"],
            "eval": evaluation["family_counts"],
        },
        "minimum_exact_support_count": min(
            train["minimum_exact_support_count"],
            evaluation["minimum_exact_support_count"],
        ),
        "maximum_exact_support_count": max(
            train["maximum_exact_support_count"],
            evaluation["maximum_exact_support_count"],
        ),
        "ingredient_curation_sha256": _canonical_sha256(curation),
        "ingredient_table_sha256": curation["ingredient_table_sha256"],
        "source_archive_sha256": archive_sha256,
        "train_eval_overlap_count": 0,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"dataset identity mismatch at {key}")

    payload = {
        "schema_version": "pantry-plan-modebench-admission-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": "admitted_for_0.5b_viability_sampling",
        "support_semantics": "exact_ingredient_support",
        "source_replay": {
            "archive_sha256": archive_sha256,
            "food_rows": curation["source"]["physical_rows"],
            "food_object_rows": curation["source"]["food_object_rows"],
            "null_placeholder_rows": curation["source"]["null_placeholder_rows"],
            "curation_sha256": _canonical_sha256(curation),
            "ingredient_table_sha256": curation["ingredient_table_sha256"],
            "manual_review_sha256": review_sha256,
        },
        "train": train,
        "eval": evaluation,
        "train_eval_overlap_count": 0,
        "dataset_identity_sha256": _canonical_sha256(identity),
        "audit_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        "[pantry-audit] "
        f"status=pass train={train['rows']} eval={evaluation['rows']} "
        f"supports={train['total_exact_supports'] + evaluation['total_exact_supports']}"
    )


if __name__ == "__main__":
    main()
