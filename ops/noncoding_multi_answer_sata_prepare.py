#!/usr/bin/env python3
"""Freeze a native-label Reuters subset of the human-validated SATA release."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from oat_drgrpo.noncoding_multi_answer_sata import VERSION, validate_record

RAW_SHA = "d9809889057a6f37bf0dd35b371a4cf3f3a23ff431f9fc69201220598e950112"
HF_REVISION = "ba43a7ab537adfa3498e3a160a6d1eafbefc95c1"
QUESTION = "What topics are related to the document above?"
ROOT = Path("var/artifacts/noncoding_multi_answer_sata_20260921")


def sha(s):
    return hashlib.sha256(s if isinstance(s, bytes) else s.encode()).hexdigest()


def normalize(s):
    return re.sub(r"\W+", "", s.casefold())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    raw_path = args.root / "raw/hf_data_main.json"
    if sha(raw_path.read_bytes()) != RAW_SHA:
        raise ValueError("Unrecognized release bytes")
    rows = json.loads(raw_path.read_text())
    exclusions, eligible = [], []
    for i, r in enumerate(rows):
        if QUESTION not in r["question"]:
            continue
        if r["question"].strip() != QUESTION:
            exclusions.append({"row": i, "reason": "question_contains_extra_nonwhitespace_text", "question": r["question"]})
            continue
        positives, negatives = r["answer groups"], r["distractor groups"]
        labels = positives + negatives
        if len(labels) != len(set(normalize(s) for s in labels)):
            exclusions.append({"row": i, "reason": "duplicate_or_conflicting_label"})
            continue
        if not 2 <= len(positives) < len(labels) <= 26:
            exclusions.append({"row": i, "reason": "invalid_support_size"})
            continue
        eligible.append((i, r))
    # Inputs inspected during source selection are forced to train, before split freeze.
    inspected_rows = {583, 584, 585, 600, 750}
    old = args.root / "raw/sata_bench_final_2025.json"
    inspected_texts = set()
    if old.exists():
        oldrows = [json.loads(l) for l in old.read_text().splitlines()]
        inspected_texts = {normalize(oldrows[i]["paragraph"]) for i in (750, 900, 1000)}
    # Group near-duplicate documents by character-shingle Jaccard >= .8.
    texts = [normalize(r["paragraph"]) for _, r in eligible]
    shingles = [{s[k:k+5] for k in range(max(0, len(s)-4))} for s in texts]
    parent = list(range(len(eligible)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    duplicate_edges = []
    for i, a in enumerate(shingles):
        for j in range(i):
            b = shingles[j]
            if min(len(a), len(b)) / max(1, max(len(a), len(b))) < .8:
                continue
            intersection = len(a & b)
            score = intersection / max(1, len(a) + len(b) - intersection)
            if score >= .8:
                parent[find(i)] = find(j)
                duplicate_edges.append({"rows": [eligible[i][0], eligible[j][0]], "jaccard": score})
    groups = {}
    for k in range(len(eligible)):
        groups.setdefault(find(k), []).append(k)
    group_ids = {g: sha("|".join(sorted(sha(texts[k]) for k in members))) for g, members in groups.items()}
    forced = {g for g, members in groups.items() if any(eligible[k][0] in inspected_rows or texts[k] in inspected_texts for k in members)}
    assigned, ntrain, ndev = {}, 0, 0
    for g in sorted(forced):
        assigned[g] = "train"
        ntrain += len(groups[g])
    for g in sorted(groups, key=lambda g: sha("noncoding_multi_answer_split_v1:" + group_ids[g])):
        if g in assigned:
            continue
        size = len(groups[g])
        if ntrain + size <= 128:
            assigned[g], ntrain = "train", ntrain + size
        elif ndev + size <= 32:
            assigned[g], ndev = "dev", ndev + size
        else:
            assigned[g] = "test"
    if (ntrain, ndev) != (128, 32):
        raise ValueError("Unable to allocate requested grouped train/dev sizes")
    records, position_counts = [], {s: [0] * 6 for s in ("train", "dev", "test")}
    for k, (i, raw) in enumerate(eligible):
        g, split = find(k), assigned[find(k)]
        task_id = "sata_news_" + sha(raw["paragraph"])[:16]
        positives, negatives = raw["answer groups"], raw["distractor groups"]
        ordered = sorted(positives + negatives, key=lambda topic: sha("noncoding_multi_answer_order_v1:" + task_id + ":" + topic))
        rotations = [ordered[j:] + ordered[:j] for j in range(len(ordered))]
        counts = position_counts[split]
        ordered = min(rotations, key=lambda ls: sum((counts[j] + int(s in positives)) ** 2 for j, s in enumerate(ls)))
        options = []
        for j, topic in enumerate(ordered):
            label = int(topic in positives)
            counts[j] += label
            options.append({"id": "reuters_topic:" + topic, "display": chr(65+j), "text": topic, "label": label})
        record = {
            "id": task_id, "schema_version": VERSION, "domain": "news", "split": split,
            "paragraph": raw["paragraph"], "source_question": raw["question"],
            "source_row_index": i, "source_answer_groups": positives, "source_distractor_groups": negatives,
            "source_record_sha256": sha(json.dumps(raw, sort_keys=True, ensure_ascii=False)),
            "document_group": group_ids[g], "options": options,
            "gold_topic_ids": sorted("reuters_topic:" + s for s in positives),
        }
        validate_record(record)
        records.append(record)
    records.sort(key=lambda r: r["id"])
    path = args.root / "noncoding_multi_answer_sata_records.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records))
    manifest = {
        "schema_version": VERSION,
        "source": {
            "dataset": "sata-bench/sata-bench", "revision": HF_REVISION,
            "url": f"https://huggingface.co/datasets/sata-bench/sata-bench/resolve/{HF_REVISION}/data_main.json",
            "raw_sha256": RAW_SHA, "license": "CC-BY-NC-4.0",
            "paper": "https://arxiv.org/abs/2506.00643",
            "upstream": "Reuters-21578 human newswire articles and topic taxonomy",
            "upstream_url": "https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection",
            "upstream_license_reported": "CC-BY-4.0 (UCI)",
            "label_provenance": "SATA release; paper reports correction by three human annotators and unanimous agreement filtering",
        },
        "records_sha256": sha(path.read_bytes()),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "source_total_rows": len(rows), "source_news_rows": len(eligible) + len(exclusions),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "gold_support_counts": dict(Counter(len(r["gold_topic_ids"]) for r in records)),
        "num_document_groups": len(groups), "near_duplicate_edges": duplicate_edges,
        "exclusions": exclusions,
        "inspected_rows_forced_train": sorted(eligible[k][0] for g in forced for k in groups[g]),
        "split_policy": "Document groups fixed before model evaluation; 128 train,32 dev,remainder reserved test; inspected rows forced train; group order SHA256; no correctness-based selection",
        "option_policy": "All unchanged source labels; deterministic SHA256 ordering with fixed rotation to balance gold display positions within split; never resampled",
        "gold_position_counts": position_counts,
        "test_status": "Frozen and not model-evaluated; test loading requires allow_test=true",
        "semantic_limit": "Known support is the released annotated topic set within each menu; not a claim that every conceivable relevant topic is exhaustively annotated",
        "adaptation": "Select any one correct listed topic; official SATA asks to select all; mode is native topic identity",
        "excluded_domains": {"multirc": "source label contradictions and semantic aliases", "biomed": "observed plausible missing disease label; deferred pending ontology audit"},
    }
    (args.root / "noncoding_multi_answer_sata_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Review packet only contains train/dev items; no test examples exposed.
    review = [r for r in records if r["split"] == "train"][:12] + [r for r in records if r["split"] == "dev"][:4]
    (args.root / "noncoding_multi_answer_sata_review_packet.json").write_text(json.dumps(review, indent=2) + "\n")
    config = {"records_path": str(path), "records_sha256": sha(path.read_bytes()), "splits": ["train", "dev"]}
    (args.root / "noncoding_multi_answer_sata_adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"records_path": str(path), "records_sha256": sha(path.read_bytes()), "split_counts": manifest["split_counts"], "groups": len(groups), "exclusions": len(exclusions), "gold_position_counts": position_counts}, indent=2))


if __name__ == "__main__":
    main()
