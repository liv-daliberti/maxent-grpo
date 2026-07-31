#!/usr/bin/env python3
"""Trim the generated PointMaze v2 splits to the common 384/64/128 design.

Certification yield is not controllable in advance, so the generator attempts a
surplus of geometries and sizes the splits to whatever survives. This step cuts
that surplus back to the exact shape every other ModeBench domain uses, so
PointMaze is statistically comparable rather than merely present.

Geometries are kept in the generator's assignment order, which interleaves
difficulty, so trimming preserves the difficulty mix instead of lopping off one
end of it. Whole geometries are kept or dropped together -- never a subset of a
geometry's four rotations -- so every retained map still has all four rotations
in its split and the holdout stays at geometry granularity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
TARGET_GEOMETRIES = {"train": 96, "dev": 16, "eval": 32}   # x4 rotations
SPLIT_KEY = {"train": "train", "dev": "multi_answer", "eval": "multi_answer"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=ROOT / "var/data/point_maze_modebench_v2")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "var/data/point_maze_modebench_v2_384")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    identity = json.loads((args.root / "identity.json").read_text())
    cert = identity["certification"]

    # Families in first-appearance order per split == generator assignment order.
    keep: dict[str, list[str]] = {}
    short: dict[str, int] = {}
    for split, want in TARGET_GEOMETRIES.items():
        seen: list[str] = []
        for entry in cert:
            if entry["split"] == split and entry["family"] not in seen:
                seen.append(entry["family"])
        keep[split] = seen[:want]
        if len(seen) < want:
            short[split] = want - len(seen)

    if short:
        print("WARNING: yield short of target, keeping everything available:")
        for split, n in short.items():
            print(f"  {split}: short {n} geometries "
                  f"({len(keep[split])}/{TARGET_GEOMETRIES[split]})")

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite")
        import shutil
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    kept_cert = [e for e in cert if e["family"] in keep[e["split"]]]
    rows_written = {}
    for split in TARGET_GEOMETRIES:
        src = load_from_disk(str(args.root / split))
        table = src[SPLIT_KEY[split]]
        wanted = set(keep[split])
        rows = [r for r in table if r["answer_mode_family"] in wanted]
        # every retained geometry must still carry all four rotations
        per_family: dict[str, int] = {}
        for r in rows:
            per_family[r["answer_mode_family"]] = per_family.get(
                r["answer_mode_family"], 0) + 1
        bad = {f: n for f, n in per_family.items() if n != 4}
        if bad:
            raise SystemExit(f"{split}: geometries without 4 rotations: {bad}")
        DatasetDict({SPLIT_KEY[split]: Dataset.from_list(rows)}).save_to_disk(
            str(args.out / split))
        rows_written[split] = len(rows)

    counts = [len(e["routes"]) for e in kept_cert]
    trimmed = {
        **{k: v for k, v in identity.items() if k != "certification"},
        "schema_version": "point-maze-modebench-data-v2-384",
        "trimmed_from": str(args.root.relative_to(ROOT)),
        "split_rows": rows_written,
        "distinct_families": {s: len(keep[s]) for s in TARGET_GEOMETRIES},
        "routes_per_map": {"min": min(counts), "max": max(counts),
                           "mean": sum(counts) / len(counts)},
        "distinct_program_sets": len({
            tuple(sorted(r["program_sha256"] for r in e["routes"]))
            for e in kept_cert}),
        "map_count": len(kept_cert),
        "target_shortfall": short,
        "certification": kept_cert,
    }
    (args.out / "identity.json").write_text(
        json.dumps(trimmed, indent=2, sort_keys=True) + "\n")
    print(f"rows: {rows_written}")
    print(f"geometries: {trimmed['distinct_families']}")
    print(f"routes/map: {trimmed['routes_per_map']}")
    print(f"distinct program sets: {trimmed['distinct_program_sets']}"
          f"/{trimmed['map_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
