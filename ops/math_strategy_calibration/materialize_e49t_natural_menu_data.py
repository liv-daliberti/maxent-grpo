#!/usr/bin/env python3
"""Materialize E49T natural-derivation prompts from the certified E49S toy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from oat_drgrpo.math_strategy_menu import (
    MENU_END,
    parse_strategy_menu,
    strategy_menu_natural_response_instructions,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49t_menu_inferred_math_haarnoja_05b.md"
)
SOURCE = ROOT / "var/data/e49s_deterministic_mathir_toy"
CERTIFICATION = (
    ROOT
    / "var/artifacts/e49s_deterministic_mathir_repairs_v1/"
    "advancement_decision.json"
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _rewrite(problem: str, expected_menu_sha256: str) -> str:
    menu = parse_strategy_menu(problem)
    if menu is None or menu.sha256 != expected_menu_sha256:
        raise RuntimeError("E49T source menu identity mismatch")
    if problem.count(MENU_END) != 1:
        raise RuntimeError("E49T source prompt has malformed menu boundary")
    menu_stop = problem.index(MENU_END) + len(MENU_END)
    rewritten = (
        problem[:menu_stop]
        + strategy_menu_natural_response_instructions(menu)
    )
    if (
        "<action_trace>" in rewritten
        or "<action_step" in rewritten
        or "MUST begin at its first character" in rewritten
    ):
        raise RuntimeError("E49T strict-format instructions survived rewrite")
    reparsed = parse_strategy_menu(rewritten)
    if reparsed is None or reparsed.sha256 != menu.sha256:
        raise RuntimeError("E49T rewrite changed the finite menu")
    return rewritten


def materialize(output_root: pathlib.Path) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"fresh E49T data root required: {output_root}")
    decision = json.loads(CERTIFICATION.read_text(encoding="utf-8"))
    if (
        not decision.get("advance_to_matched_toy_training")
        or decision.get("support_counts", {}).get("train_multi") != 10
        or decision.get("support_counts", {}).get("eval_multi") != 10
    ):
        raise RuntimeError("E49S certification gate is not frozen and passing")

    output_root.mkdir(parents=True)
    counts: dict[str, int] = {}
    multi_counts: dict[str, int] = {}
    for split in ("train", "eval"):
        source = load_from_disk(str(SOURCE / split))
        source_name = next(iter(source))
        rows = []
        multi = 0
        for raw in source[source_name]:
            row = dict(raw)
            expected_hash = str(row["strategy_menu_sha256"])
            row["problem"] = _rewrite(str(row["problem"]), expected_hash)
            menu = parse_strategy_menu(row["problem"])
            assert menu is not None
            multi += int(len(menu.strategies) >= 2)
            rows.append(row)
        target_name = "train" if split == "train" else "math"
        DatasetDict({target_name: Dataset.from_list(rows)}).save_to_disk(
            str(output_root / split)
        )
        counts[split] = len(rows)
        multi_counts[split] = multi

    if counts != {"train": 50, "eval": 50} or multi_counts != {
        "train": 10,
        "eval": 10,
    }:
        raise RuntimeError("E49T materialized support differs from E49S")
    manifest = {
        "schema": "e49t_natural_menu_materialization_v1",
        "menu_count": 100,
        "row_counts": counts,
        "multi_support_counts": multi_counts,
        "train_tree_sha256": _tree_sha256(output_root / "train"),
        "eval_tree_sha256": _tree_sha256(output_root / "eval"),
        "source_manifest_sha256": _sha256(
            SOURCE / "MATERIALIZATION_MANIFEST.json"
        ),
        "e49s_certification_sha256": _sha256(CERTIFICATION),
        "protocol_sha256": _sha256(PROTOCOL),
        "materializer_sha256": _sha256(SCRIPT),
    }
    _write(output_root / "MATERIALIZATION_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.output_root.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
