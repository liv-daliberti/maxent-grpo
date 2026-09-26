#!/usr/bin/env python3
"""Write an identity record for a released ModeBench dataset that lacks one.

Four of the five Level-1 domains ship a construction record beside their rows.
Countdown does not: ``var/data/exact_countdown_easy3_probe`` holds only its two
split directories, so nothing beside the data pins what was released. This
writes the missing record from the rows themselves.

It records only what the released bytes establish -- a content digest, split
sizes, support statistics, split disjointness, and the structural parameters
the rows exhibit. It does **not** reconstruct the generator invocation. The
shipped splits do not match the generator's defaults, so the arguments that
produced them are not recoverable from the release, and the record says that
rather than guessing. The seed is an exception: the generator writes it into
every ``instance_id``, so it is read back rather than assumed.

Refuses to overwrite an existing record: the other four domains' identities
were written by their generators at construction time and are the stronger
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "var/data/exact_countdown_easy3_probe"
GENERATOR = ROOT / "ops/make_exact_countdown_mode_data.py"


def tree_digest(root: Path) -> str:
    """Content digest over the dataset's files, independent of walk order."""
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "identity.json":
            continue
        entries.append(
            f"{path.relative_to(root).as_posix()}:"
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def split_facts(root: Path, split: str) -> tuple[dict, set[str], set[int]]:
    from datasets import load_from_disk

    loaded = load_from_disk(str(root / split))
    names = list(loaded.keys()) if hasattr(loaded, "keys") else [None]
    facts, problems, seeds = {}, set(), set()
    for name in names:
        subset = loaded[name] if name else loaded
        counts = list(subset["answer_mode_count"])
        numbers, targets = [], []
        for raw in subset["answer"]:
            meta = json.loads(raw)
            numbers.append(meta["numbers"])
            targets.append(meta["target"])
            # The generator stamps its seed into every identifier; read it back
            # rather than assuming the default.
            parts = str(meta["instance_id"]).rsplit("-", 2)
            if len(parts) == 3 and parts[1].isdigit():
                seeds.add(int(parts[1]))
        problems.update(subset["problem"])
        facts[name or split] = {
            "rows": len(subset),
            "support": {"mean": statistics.mean(counts),
                        "min": min(counts), "max": max(counts)},
            "operands_per_prompt": sorted({len(v) for v in numbers}),
            "operand_value_range": [min(min(v) for v in numbers),
                                    max(max(v) for v in numbers)],
            "target_range": [min(targets), max(targets)],
        }
    return facts, problems, seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing record (refused by default)")
    args = parser.parse_args()
    target = args.dataset / "identity.json"
    if target.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite the existing record at {target}")

    train, train_problems, train_seeds = split_facts(args.dataset, "train")
    evaluation, eval_problems, eval_seeds = split_facts(args.dataset, "eval")
    # The generator offsets the evaluation seed by 10,000 from the base it was
    # given, so the smallest stamped value is the base.
    stamped = sorted(train_seeds | eval_seeds)

    record = {
        "schema": "modebench_countdown_identity_v1",
        "reconstructed": {
            "by": "ops/build_modebench_identity_record.py",
            "why": "the release shipped without a construction record; this one "
                   "is derived from the released rows, not from the run that "
                   "produced them.",
        },
        "data_tree_sha256": tree_digest(args.dataset),
        "split_rows": {"train": train, "eval": evaluation},
        "split_overlap_count": {
            "train_vs_eval_problems": len(train_problems & eval_problems)},
        "seed_stamps": stamped,
        "provenance": {
            "generator": str(GENERATOR.relative_to(ROOT)),
            "generator_sha256_at_record_time":
                hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
                if GENERATOR.is_file() else None,
            "arguments_recoverable": False,
            "note": "the shipped split sizes and support bounds differ from the "
                    "generator's defaults, so the invocation that produced this "
                    "release is not recoverable from the release alone.",
        },
    }
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "written", "path": str(target),
                      "data_tree_sha256": record["data_tree_sha256"],
                      "overlap": record["split_overlap_count"]}))


if __name__ == "__main__":
    main()
