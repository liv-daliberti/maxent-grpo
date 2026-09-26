#!/usr/bin/env python3
"""Convert the frozen ModeBench datasets to verl's DAPO parquet contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd
from transformers import AutoTokenizer

import e113r4_official_dapo_common as common

import sys

sys.path.insert(0, str(common.ROOT / "src"))
from oat_drgrpo import templates  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    if path.is_file():
        value.update(path.read_bytes())
        return value.hexdigest()
    for member in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if "__pycache__" in member.parts or member.suffix == ".pyc":
            continue
        value.update(str(member.relative_to(path)).encode("utf-8"))
        value.update(b"\0")
        value.update(member.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def select_split(value: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(value, Dataset):
        return value
    wanted = "train" if split == "train" else "multi_answer"
    if wanted not in value:
        raise RuntimeError(f"dataset lacks frozen {wanted!r} split: {list(value)}")
    return value[wanted]


def prompt_messages(domain: str, question: str) -> list[dict[str, str]]:
    if domain == "pantry_plan":
        system = templates._PANTRY_SUPPORT_MASK_SYSTEM
        user = templates._pantry_support_mask_user(
            question, "e113r4_official_dapo"
        )
    else:
        system = templates._BOXED_SYSTEM
        user = question
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def rows(domain: str, split: str) -> list[dict[str, Any]]:
    source = select_split(load_from_disk(str(common.DATA_ROOTS[domain] / split)), split)
    if split == "train" and len(source) != common.GEN_PROMPT_BATCH:
        raise RuntimeError(
            f"{domain}: expected exactly {common.GEN_PROMPT_BATCH} train rows, "
            f"found {len(source)}"
        )
    required = {"problem", "answer"}
    if not required.issubset(source.column_names):
        raise RuntimeError(f"{domain}/{split}: missing columns {required - set(source.column_names)}")
    converted: list[dict[str, Any]] = []
    for index, source_row in enumerate(source):
        question = str(source_row["problem"])
        answer = source_row["answer"]
        converted.append(
            {
                "data_source": f"modebench/{domain}",
                "prompt": prompt_messages(domain, question),
                "ability": domain,
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "domain": domain,
                    "problem": question,
                },
            }
        )
    return converted


def audit_rendering(payloads: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for family in common.FAMILIES:
        tokenizer = AutoTokenizer.from_pretrained(
            common.MODEL_ROOTS[family], local_files_only=True
        )
        family_report: dict[str, Any] = {}
        for domain in common.DOMAINS:
            template_name = common.PROMPT_TEMPLATES[family][domain]
            template = templates.TEMPLATE_FACTORY[template_name]
            max_tokens = 0
            for split in ("train", "eval"):
                for index, row in enumerate(payloads[(domain, split)]):
                    observed = tokenizer.apply_chat_template(
                        row["prompt"], tokenize=False, add_generation_prompt=True
                    )
                    expected = template(row["extra_info"]["problem"])
                    if observed != expected:
                        raise RuntimeError(
                            f"{family}/{domain}/{split}/{index}: tokenizer chat "
                            "rendering differs from the frozen comparator prompt"
                        )
                    token_count = len(tokenizer.encode(observed, add_special_tokens=False))
                    max_tokens = max(max_tokens, token_count)
                    if token_count > common.PROMPT_LENGTHS[domain]:
                        raise RuntimeError(
                            f"{family}/{domain}/{split}/{index}: {token_count} tokens "
                            f"exceeds frozen limit {common.PROMPT_LENGTHS[domain]}"
                        )
            family_report[domain] = {
                "prompt_template": template_name,
                "maximum_rendered_tokens": max_tokens,
                "maximum_allowed_tokens": common.PROMPT_LENGTHS[domain],
            }
        report[family] = family_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    for model in common.MODEL_ROOTS.values():
        if not (model / "config.json").is_file():
            raise SystemExit(f"missing frozen model snapshot: {model}")
    payloads = {
        (domain, split): rows(domain, split)
        for domain in common.DOMAINS
        for split in ("train", "eval")
    }
    render_audit = audit_rendering(payloads)

    expected_outputs = {
        (domain, split): common.parquet_path(domain, split)
        for domain in common.DOMAINS
        for split in ("train", "eval")
    }
    if not args.check:
        common.DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
        for key, path in expected_outputs.items():
            frame = pd.DataFrame(payloads[key])
            temporary = path.with_name(f".{path.name}.tmp")
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)

    missing = [str(path) for path in expected_outputs.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing converted parquet files: " + ", ".join(missing))
    output_records = []
    for (domain, split), path in expected_outputs.items():
        observed = pd.read_parquet(path)
        if len(observed) != len(payloads[(domain, split)]):
            raise RuntimeError(f"row-count drift after parquet round trip: {path}")
        output_records.append(
            {
                "domain": domain,
                "split": split,
                "path": str(path),
                "rows": len(observed),
                "sha256": digest(path),
                "source": str(common.DATA_ROOTS[domain] / split),
                "source_sha256": digest(common.DATA_ROOTS[domain] / split),
            }
        )
    manifest = {
        "schema": "e113r4_official_verl_dapo_data_v1",
        "train_rows": common.GEN_PROMPT_BATCH,
        "prompt_rendering_exactly_matches_frozen_comparators": True,
        "render_audit": render_audit,
        "outputs": output_records,
    }
    if args.check:
        current = json.loads(common.DATA_MANIFEST.read_text(encoding="utf-8"))
        if current != manifest:
            raise SystemExit("E113-R4 data manifest drifted")
    else:
        atomic_json(common.DATA_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

