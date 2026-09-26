#!/usr/bin/env python3
"""Build disjoint, support-matched Level 3 development candidate pools."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "ops", ROOT / "ops/exp_scaling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from materialize_modebench_harder_v2 import (
    DOMAINS, LEVEL1, SPLITS, existing_ids, identity_set, load_rows, modes, row_hash,
)

REFERENCE = ROOT / "var/data/modebench_harder_v2_matched_r5"
DEFAULT_OUTPUT = ROOT / "var/data/modebench_level3_calibration_v1"
SEEDS = {domain: 6317100 + 10000 * i for i, domain in enumerate(DOMAINS)}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def historical_ids(domain: str) -> set:
    blocked = existing_ids(domain)
    data_root = ROOT / "var/data"
    prior_roots = set(data_root.glob("modebench_harder*")) | set(data_root.glob("modebench_level3*"))
    for root in sorted(prior_roots):
        if not root.is_dir():
            continue
        for split, (_, dataset_split) in SPLITS.items():
            path = root / domain / split
            if path.is_dir() and (path / "dataset_dict.json").exists():
                from datasets import load_from_disk
                dataset = load_from_disk(str(path))
                for subset in dataset.values():
                    blocked |= identity_set(domain, [dict(row) for row in subset])
    return blocked


def reference_rows(domain: str, split: str) -> list[dict]:
    return load_rows(REFERENCE / domain / split, SPLITS[split][1])


def generator(domain: str):
    if domain == "graph_coloring":
        from modebench_level3_graph_v6 import build_pool
    elif domain == "python_factors":
        from modebench_level3_python_v4 import build_pool
    elif domain == "mathir":
        from modebench_level3_mathir_sign_v1 import build_pool
    elif domain == "countdown":
        from modebench_level3_countdown_v3 import build_pool
    else:
        from modebench_level3_constraints import build_pool
    return build_pool


def verify_rows(domain: str, rows: list[dict], reference: list[dict],
                target: Counter, blocked: set) -> dict:
    ids = identity_set(domain, rows)
    def contract(row):
        return (row["modebench_task"], json.loads(row["answer"])["verifier"])
    checks = {
        "row_count": len(rows) == sum(target.values()),
        "unique_identities": len(ids) == len(rows),
        "historical_and_cross_split_disjointness": not (ids & blocked),
        "exact_support_histogram": modes(rows) == target,
        "unchanged_verifier_contract": {contract(r) for r in rows} ==
                                       {contract(r) for r in reference},
    }
    if not all(checks.values()):
        raise RuntimeError(f"{domain} structural checks failed: {checks}")
    return checks


def build_development_pool(domain: str, difficulty: int, output: Path,
                           multiplier: int = 1) -> dict:
    directory = output / "pools" / domain
    rows_path = directory / f"difficulty_{difficulty}.jsonl"
    manifest_path = rows_path.with_suffix(".identity.json")
    if rows_path.exists() or manifest_path.exists():
        raise FileExistsError(rows_path)
    reference = reference_rows(domain, "dev")
    target = Counter({k: v * multiplier for k, v in modes(reference).items()})
    blocked = historical_ids(domain)
    pool_directories = {directory}
    pool_directories.update(root / "pools" / domain for root in (ROOT / "var/data").glob("modebench_level3_calibration*"))
    for pool_directory in sorted(pool_directories):
        for path in sorted(pool_directory.glob("*.jsonl")):
            blocked |= identity_set(domain, [json.loads(line) for line in path.read_text().splitlines()])
    seed = SEEDS[domain] + 1000 * difficulty
    extra = {}
    if domain == "pantry":
        extra["joint_target"] = Counter({k: v * multiplier for k, v in
            Counter((int(r["answer_mode_count"]), r["answer_mode_family"]) for r in reference).items()})
    rows = generator(domain)(domain, target, blocked, seed, "level3_development_pool",
                             difficulty, multiplier=1, **extra)
    checks = verify_rows(domain, rows, reference, target, blocked)
    if domain == "pantry":
        checks["exact_family_histogram"] = Counter((int(r["answer_mode_count"]), r["answer_mode_family"]) for r in rows) == extra["joint_target"]
        if not checks["exact_family_histogram"]:
            raise RuntimeError("Pantry family histogram drift")
    directory.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    record = {
        "schema": "modebench_level3_development_pool_v1", "domain": domain,
        "difficulty": difficulty, "seed": seed, "rows": len(rows),
        "rows_sha256": row_hash(rows), "checks": checks,
        "support_histogram": dict(sorted(target.items())),
        "source_sha256": hashlib.sha256(Path(sys.modules[generator(domain).__module__].__file__).read_bytes()).hexdigest(),
        "information_boundary": "development candidates only; no confirmation model outcomes used",
    }
    write_json(manifest_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--difficulty", required=True, type=int, choices=range(4))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--multiplier", type=int, default=1)
    args = parser.parse_args()
    if args.multiplier < 1:
        parser.error("multiplier must be positive")
    print(json.dumps(build_development_pool(args.domain, args.difficulty,
                                           args.output_root, args.multiplier), sort_keys=True))


if __name__ == "__main__":
    main()
