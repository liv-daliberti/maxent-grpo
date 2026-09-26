#!/usr/bin/env python3
"""Build reviewed MultiRC tasks without fabricating or relabeling any answer.

Usage: PYTHONPATH=src python ops/noncoding_multi_answer_prepare.py \
  --review var/artifacts/noncoding_multi_answer_20260921/noncoding_multi_answer_review.json
Raw official archives must already be downloaded; this builder never uses a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import zipfile

from oat_drgrpo.noncoding_multi_answer import validate_record


ROOT = Path("var/artifacts/noncoding_multi_answer_20260921")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_passage(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text))).strip()


def source_records(archive: Path):
    with zipfile.ZipFile(archive) as z:
        for member, official_split in (("splitv2/train_456-fixedIds.json", "train"), ("splitv2/dev_83-fixedIds.json", "validation")):
            data = json.loads(z.read(member))["data"]
            for p in data:
                passage_hash = sha(clean_passage(p["paragraph"]["text"]).encode())
                for q in p["paragraph"]["questions"]:
                    yield {
                        "id": p["id"] + "#" + q["idx"],
                        "source_passage_id": p["id"],
                        "source_question_id": q["idx"],
                        "source_official_split": official_split,
                        "source_archive_member": member,
                        "source_passage_sha256": passage_hash,
                        "passage": clean_passage(p["paragraph"]["text"]),
                        "question": q["question"],
                        "source_options": [{"text": a["text"], "label": int(a["isAnswer"])} for a in q["answers"]],
                        "source_multisent": q["multisent"],
                        "source_sentences_used": q["sentences_used"],
                    }


def assigned_split(record):
    if record["source_official_split"] == "validation":
        return "test"
    # Passage ID rather than question ensures all sibling questions stay together.
    return "dev" if int(sha(record["source_passage_id"].encode())[:8], 16) % 5 == 0 else "train"


def build_record(raw, review, position_counts):
    gold = {i for i, a in enumerate(raw["source_options"]) if a["label"]}
    mapped = [i for c in review["clusters"] for i in c["option_indices"]]
    if len(mapped) != len(set(mapped)) or set(mapped) != gold:
        raise ValueError(f"Review must partition every and only original gold option: {raw['id']}")
    if len(review["clusters"]) < 2:
        raise ValueError("At least two distinct reviewed semantic modes are required")
    split = assigned_split(raw)
    n = len(raw["source_options"])
    order = sorted(range(n), key=lambda i: sha(f"noncoding_multi_answer_order_v1:{raw['id']}:{i}".encode()))
    counts = position_counts[(split, n)]
    if not counts:
        counts.extend([0] * n)
    rotations = [order[k:] + order[:k] for k in range(n)]
    order = min(rotations, key=lambda perm: sum((counts[pos] + int(i in gold)) ** 2 for pos, i in enumerate(perm)))
    options = []
    for pos, i in enumerate(order):
        source = raw["source_options"][i]
        counts[pos] += source["label"]
        options.append({"id": f"answer_{i}", "source_index": i, "display": chr(65 + pos), **source})
    mode_map = {}
    mode_definitions = {}
    for k, c in enumerate(review["clusters"]):
        key = f"{raw['id']}:semantic_{k}"
        mode_definitions[key] = c["meaning"]
        for i in c["option_indices"]:
            mode_map[f"answer_{i}"] = key
    return {
        **raw,
        "split": split,
        "options": options,
        "gold_mode_map": mode_map,
        "mode_definitions": mode_definitions,
        "review_status": "assistant_audited_source_labels_and_semantic_modes",
        "review_rationale": review["rationale"],
        "option_order_policy": "fixed_sha256_permutation_then_gold_position_balance_rotation_v1",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    archive = args.root / "raw/original_multirc_v2.zip"
    raw = list(source_records(archive))
    by_id = {r["id"]: r for r in raw}
    assert len(by_id) == len(raw)
    review_data = json.loads(args.review.read_text())
    review = review_data["approved"]
    ids = [r["id"] for r in review]
    assert len(set(ids)) == len(ids)
    counts = defaultdict(list)
    records = [build_record(by_id[r["id"]], r, counts) for r in sorted(review, key=lambda r: r["id"])]
    groups = defaultdict(set)
    content_groups = defaultdict(set)
    for r in records:
        validate_record(r)
        groups[r["source_passage_id"]].add(r["split"])
        content_groups[r["source_passage_sha256"]].add(r["split"])
        assert sorted(r["options"], key=lambda o: o["source_index"]) == [
            {**a, "id": f"answer_{i}", "source_index": i, "display": next(o["display"] for o in r["options"] if o["source_index"] == i)}
            for i, a in enumerate(r["source_options"])
        ]
    assert all(len(s) == 1 for s in groups.values()), "passage ID leakage"
    assert all(len(s) == 1 for s in content_groups.values()), "duplicate passage leakage"
    def write_jsonl(name, rows):
        path = args.root / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
        return {"path": str(path), "sha256": sha(path.read_bytes()), "records": len(rows)}
    outputs = [write_jsonl("pilot_records.jsonl", records)]
    # Reserved final evaluation data are not loadable as reviewed tasks until audited.
    reserved = [r for r in raw if r["source_official_split"] == "validation" and 2 <= sum(a["label"] for a in r["source_options"]) < len(r["source_options"])]
    outputs.append(write_jsonl("reserved_official_validation_unreviewed.jsonl", reserved))
    manifest = {
        "dataset": "MultiRC original v2 one-valid-option adaptation",
        "source_url": "https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip",
        "raw_sha256": sha(archive.read_bytes()),
        "review_sha256": sha(args.review.read_bytes()),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "source_question_counts": dict(Counter(r["source_official_split"] for r in raw)),
        "reviewed_split_counts": dict(Counter(r["split"] for r in records)),
        "reviewed_passage_counts": {s: len({r["source_passage_id"] for r in records if r["split"] == s}) for s in ("train", "dev", "test")},
        "semantic_mode_counts": dict(Counter(len(set(r["gold_mode_map"].values())) for r in records)),
        "gold_position_counts_by_split_and_option_count": {str(k): v for k, v in counts.items()},
        "reserved_final_test_status": "Official validation; structural metadata only; no model evaluation; semantic review required before loading",
        "human_review": "None claimed. Source labels are human-created; subset curation and mode clustering are assistant-assisted and reviewable.",
        "outputs": outputs,
    }
    (args.root / "noncoding_multi_answer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
