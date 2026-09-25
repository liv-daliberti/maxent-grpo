#!/usr/bin/env python3
"""Compare released SATA news text/labels with the public original Reuters archive.

This is a source provenance check, not model evaluation or a claim of an
independent semantic annotation. No question is excluded based on model output.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import html
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", html.unescape(text).lower())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path):
    release = root / "raw/hf_data_main.json"
    archive = root / "provenance_audit/reuters_uci.zip"
    raw = json.loads(release.read_text())
    with zipfile.ZipFile(archive) as z:
        data = z.read("reuters21578.tar.gz")
        (root / "provenance_audit/Reuters_original_README.txt").write_bytes(z.read("README.txt"))
    corpus = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        for member in t.getmembers():
            if not member.name.endswith(".sgm"):
                continue
            text = t.extractfile(member).read().decode("latin1")
            for m in re.finditer(r"<REUTERS\b([^>]*)>(.*?)</REUTERS>", text, re.S):
                body = re.search(r"<BODY>(.*?)</BODY>", m[2], re.S)
                if body is None or not normalize(body[1]):
                    continue
                topics = re.search(r"<TOPICS>(.*?)</TOPICS>", m[2], re.S)
                nid = re.search(r'NEWID="(\d+)"', m[1])
                corpus.append({"id": nid[1], "body": normalize(body[1]), "raw_body_sha256": hashlib.sha256(body[1].encode()).hexdigest(), "topics": re.findall(r"<D>(.*?)</D>", topics[1]) if topics else []})
    rows, article_to_rows, shingles = [], defaultdict(list), {}
    for i, r in enumerate(raw):
        if "topics are related to the document above" not in r["question"]:
            continue
        p = normalize(r["paragraph"])
        matches = [x for x in corpus if p in x["body"] or (len(x["body"].removesuffix("reuter")) >= .9 * len(p) and x["body"].removesuffix("reuter") in p)]
        words = re.findall(r"[a-z0-9]+", r["paragraph"].lower())
        shingles[i] = {tuple(words[j:j + 5]) for j in range(max(0, len(words) - 4))}
        out = {"source_index": i, "paragraph_sha256": hashlib.sha256(r["paragraph"].encode()).hexdigest(), "match_count": len(matches), "sata_gold": r["answer groups"], "sata_distractors": r["distractor groups"], "matches": []}
        for x in matches:
            article_to_rows[x["id"]].append(i)
            out["matches"].append({"reuters_id": x["id"], "original_topics": x["topics"], "original_body_sha256": x["raw_body_sha256"], "original_positive_as_sata_negative": sorted(set(x["topics"]) & set(r["distractor groups"])), "sata_positive_not_original": sorted(set(r["answer groups"]) - set(x["topics"]))})
        rows.append(out)
    similar = []
    keys = sorted(shingles)
    for offset, a in enumerate(keys):
        for b in keys[offset + 1:]:
            union = shingles[a] | shingles[b]
            score = len(shingles[a] & shingles[b]) / len(union) if union else 0
            if score >= .6:
                similar.append({"source_indices": [a, b], "five_word_shingle_jaccard": score})
    summary = {
        "schema": "sata-reuters-independent-source-comparison-v3",
        "source_urls": ["https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection", "https://huggingface.co/datasets/sata-bench/sata-bench"],
        "archive_sha256": digest(archive), "sata_release_sha256": digest(release),
        "original_articles_with_body": len(corpus), "sata_news_rows": len(rows),
        "uniquely_matched": sum(r["match_count"] == 1 for r in rows),
        "unmatched": [r["source_index"] for r in rows if r["match_count"] == 0],
        "multiple_matches": [r["source_index"] for r in rows if r["match_count"] > 1],
        "original_gold_used_as_distractor": [r["source_index"] for r in rows if any(m["original_positive_as_sata_negative"] for m in r["matches"])],
        "sata_additional_positive": [r["source_index"] for r in rows if any(m["sata_positive_not_original"] for m in r["matches"])],
        "shared_original_article_groups": {k: sorted(set(v)) for k, v in article_to_rows.items() if len(set(v)) > 1},
        "similar_sata_news_pairs": similar,
        "matching_rule": "paragraph lowercase alphanumeric text occurs in original body, or original body minus final Reuter covers at least 90% of paragraph and occurs in it; original records and hashes retained",
        "rows": rows,
    }
    (root / "provenance_audit/news_source_comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
