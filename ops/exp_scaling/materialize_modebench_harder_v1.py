#!/usr/bin/env python3
"""Materialize a verifier-compatible harder evaluation mode for E117 tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "ops", ROOT / "src", ROOT / "ops" / "exp_scaling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_exact_answer_mode_data import _synthetic_graph_rows
from make_exact_countdown_mode_data import _synthetic_countdown_rows
from make_mathir_action_menu_data import _build_rows as _build_mathir_rows, _family_support
from make_python_factor_mode_data import _build_rows as _build_python_rows
from make_pantry_plan_mode_data import (
    FAMILY_POOLS as PANTRY_FAMILIES,
    _build_row as _build_pantry_row,
    _canonical_sha256,
)
from materialize_e117_evaluation_reserves import (
    DOMAIN_ORDER,
    SOURCE_ROOTS,
    _identities,
    _load_source_rows,
)

SCHEMA = "modebench_harder_v1"
OUTPUT_ROOT = ROOT / "var/data/modebench_harder_v1"
ROWS_PER_DOMAIN = 128
DOMAIN_ORDER = tuple(DOMAIN_ORDER) + ("pantry",)
SEEDS = {
    "countdown": 3117100,
    "graph_coloring": 3117200,
    "python_factors": 3117300,
    "mathir": 3117400,
    "pantry": 3117500,
}
HARD_MATHIR_FAMILIES = {
    "ax_plus_b_eq_dx_plus_c",
    "ax_plus_b_eq_c_minus_dx",
}


def _rows_hash(rows: list[dict[str, Any]]) -> str:
    text = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(text.encode()).hexdigest()


def _existing_ids(domain: str) -> set[tuple[Any, ...]]:
    if domain == "pantry":
        root = ROOT / "var/data/pantry_plan_modebench_v2"
        fingerprints: set[str] = set()
        for partition, split in (("train", "train"), ("dev", "multi_answer"), ("eval", "multi_answer")):
            rows = load_from_disk(str(root / partition))[split]
            fingerprints |= {str(row["instance_fingerprint"]) for row in rows}
        return {("pantry", fingerprint) for fingerprint in fingerprints}
    ids = _identities(domain, _load_source_rows(domain))
    reserve = ROOT / "var/data/e117_evaluation_reserve_v1"
    for block in ("development", "confirmation"):
        dataset = load_from_disk(str(reserve / block / domain / "eval"))["multi_answer"]
        ids |= _identities(domain, [dict(row) for row in dataset])
    return ids


def build_rows(domain: str, excluded: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
    tag = "harder_v1"
    seed = SEEDS[domain]
    generator_excluded = {identity[1:] for identity in excluded}
    if domain == "countdown":
        return _synthetic_countdown_rows(
            ROWS_PER_DOMAIN, seed=seed, split_tag=tag, number_count=4,
            max_value=10, min_modes=4, max_modes=64,
            exclude=generator_excluded,
        )
    if domain == "graph_coloring":
        return _synthetic_graph_rows(
            ROWS_PER_DOMAIN, seed=seed, split_tag=tag, hidden_count=4,
            min_completions=4, max_completions=32, min_solutions=4,
            max_n=6, max_edges=10, prompt_style="original",
            balance_hidden_color=False, exclude=generator_excluded,
        )
    if domain == "python_factors":
        return _build_python_rows(
            ROWS_PER_DOMAIN, seed=seed, split_tag=tag, case_count=6,
            max_value=192, min_modes=64,
            excluded={identity[1] for identity in excluded},
        )
    if domain == "mathir":
        # Build a balanced superset, then retain only the two families that
        # require moving x-terms across the equality before isolation.
        rows = _build_mathir_rows(
            ROWS_PER_DOMAIN * 2, seed=seed, split_tag=tag,
            family_support=_family_support(),
            excluded={(identity[1], identity[2]) for identity in excluded},
        )
        selected = [row for row in rows if row["mathir_family"] in HARD_MATHIR_FAMILIES]
        if len(selected) != ROWS_PER_DOMAIN:
            raise RuntimeError(f"hard MathIR family balance yielded {len(selected)} rows")
        return selected
    if domain == "pantry":
        curation = json.loads((ROOT / "var/data/pantry_plan_v1/ingredients.json").read_text())
        source_by_id = {row["id"]: row for row in curation["ingredients"]}
        blocked = {identity[1] for identity in excluded}
        rng = random.Random(seed)
        rows: list[dict[str, Any]] = []
        counts: defaultdict[str, int] = defaultdict(int)
        attempts = 0
        while any(counts[family] < 32 for family in PANTRY_FAMILIES):
            attempts += 1
            if attempts > 200_000:
                raise RuntimeError("could not build balanced scarce-support Pantry rows")
            choices = [family for family in PANTRY_FAMILIES if counts[family] < 32]
            family = rng.choice(choices)
            row = _build_pantry_row(
                family=family, split=tag, seed=seed, index=counts[family], rng=rng,
                source_by_id=source_by_id,
                ingredient_table_sha256=str(curation["ingredient_table_sha256"]),
            )
            if row is None or not 8 <= int(row["answer_mode_count"]) <= 12:
                continue
            spec = json.loads(row["answer"])
            fingerprint_spec = dict(spec)
            fingerprint_spec.pop("instance_id", None)
            fingerprint = _canonical_sha256(fingerprint_spec)
            if fingerprint in blocked:
                continue
            row["instance_fingerprint"] = fingerprint
            blocked.add(fingerprint)
            rows.append(row)
            counts[family] += 1
        rng.shuffle(rows)
        return rows
    raise ValueError(domain)


def _validate(domain: str, rows: list[dict[str, Any]], excluded: set[tuple[Any, ...]]) -> None:
    identities = (
        {("pantry", row["instance_fingerprint"]) for row in rows}
        if domain == "pantry" else _identities(domain, rows)
    )
    if len(rows) != ROWS_PER_DOMAIN or len(identities) != ROWS_PER_DOMAIN:
        raise RuntimeError(f"{domain}: row count or uniqueness failure")
    if identities & excluded:
        raise RuntimeError(f"{domain}: overlap with train/easy/reserved prompts")
    specs = [json.loads(row["answer"]) for row in rows]
    if domain == "countdown" and not all(len(s["numbers"]) == 4 for s in specs):
        raise RuntimeError("Countdown difficulty drift")
    if domain == "graph_coloring" and not all(sum(v is None for v in s["partial_colors"]) == 4 for s in specs):
        raise RuntimeError("graph difficulty drift")
    if domain == "python_factors" and not all(len(s["cases"]) == 6 for s in specs):
        raise RuntimeError("Python difficulty drift")
    if domain == "mathir" and {s["family"] for s in specs} != HARD_MATHIR_FAMILIES:
        raise RuntimeError("MathIR difficulty drift")
    if domain == "pantry":
        if not all(8 <= int(s["certified_mode_count"]) <= 12 for s in specs):
            raise RuntimeError("Pantry difficulty drift")
        if {s["family"] for s in specs} != set(PANTRY_FAMILIES):
            raise RuntimeError("Pantry family coverage drift")


def materialize(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    built: dict[str, list[dict[str, Any]]] = {}
    records: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        excluded = _existing_ids(domain)
        rows = build_rows(domain, excluded)
        _validate(domain, rows, excluded)
        built[domain] = rows
        records[domain] = {
            "rows": len(rows), "seed": SEEDS[domain], "rows_sha256": _rows_hash(rows),
            "excluded_identity_count": len(excluded),
        }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".modebench_harder_v1.", dir=output_root.parent))
    try:
        for domain, rows in built.items():
            DatasetDict({"harder": Dataset.from_list(rows)}).save_to_disk(str(temporary / domain / "eval"))
        manifest = {
            "schema": SCHEMA, "rows_per_domain": ROWS_PER_DOMAIN,
            "domains": records,
            "difficulty": {
                "countdown": "4 numbers (easy uses 3)",
                "graph_coloring": "4 hidden vertices (easy uses 3)",
                "python_factors": "6 test cases (easy uses 4)",
                "mathir": "variable terms on both equation sides (easy mixes four families)",
                "pantry": "8-12 feasible supports, balanced over four families (easy allows up to 45)",
            },
            "split": "harder", "verifier_contract": "unchanged",
        }
        (temporary / "identity.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
