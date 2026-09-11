#!/usr/bin/env python3
"""Materialize the E49X hard-MATH toy from empirically executable routes."""

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
    MENU_START,
    MathStrategyMenu,
    parse_strategy_menu,
    strategy_menu_natural_response_instructions,
)
from oat_drgrpo.templates import apply_qwen_math_template


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/e49x_accessible_bottom_up_math_toy_20260726.md"
)
E49T_DATA = ROOT / "var/data/e49t_natural_menu_math_toy"
E49T_MANIFEST = E49T_DATA / "MATERIALIZATION_MANIFEST.json"
E49U = (
    ROOT
    / "var/artifacts/e49u_05b_route_capability_calibration_v1/result.json"
)
E49W = (
    ROOT
    / "var/artifacts/e49w_bottom_up_05b_route_calibration_v1/result.json"
)
CALIBRATION_RESULT = E49W
CALIBRATION_SCHEMA = "e49w_bottom_up_05b_route_calibration_v1"
OUTPUT_SCHEMA = "e49x_accessible_bottom_up_math_toy_v1"
OUTPUT_LABEL = "E49X"
CALIBRATION_ORIGIN = "e49w_bottom_up_bidirectional"
SELECTED_IDS_FIELD = "selected_problem_ids"
PROBLEM_ID_FIELD = "problem_id"
WRITE_TRAIN_ROUTE_PROBE = False
NEUTRAL_ROUTE_CHOICE = False
BASE_SCRIPT: pathlib.Path | None = None
FULL = ROOT / "var/data/math12k_384_math500"
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
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


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _menu_from_payload(payload: dict[str, Any]) -> MathStrategyMenu:
    embedded = (
        f"x\n{MENU_START}\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + f"\n{MENU_END}"
    )
    menu = parse_strategy_menu(embedded)
    if menu is None:
        raise RuntimeError("E49X menu payload disappeared during parsing")
    return menu


def _prune(menu: MathStrategyMenu, retained_strategy_id: str) -> MathStrategyMenu:
    retained = menu.strategy(retained_strategy_id)
    if retained is None:
        raise RuntimeError("E49X tried to prune an unknown strategy")
    used = set(retained.action_ids)
    actions = [action for action in menu.actions if action.action_id in used]
    action_map = {
        action.action_id: f"A{index}"
        for index, action in enumerate(actions, start=1)
    }
    return _menu_from_payload(
        {
            "schema": "math_strategy_action_menu_v1",
            "actions": [
                {
                    "action_id": action_map[action.action_id],
                    "operation": action.operation,
                }
                for action in actions
            ],
            "strategies": [
                {
                    "strategy_id": "S1",
                    "action_ids": [
                        action_map[action_id]
                        for action_id in retained.action_ids
                    ],
                    "plan": retained.plan,
                }
            ],
        }
    )


def _embed(original_problem: str, menu: MathStrategyMenu) -> str:
    neutral_choice = ""
    if NEUTRAL_ROUTE_CHOICE and len(menu.strategies) > 1:
        neutral_choice = (
            "\n\nStrategy IDs and listing order are labels only; neither "
            "strategy is preferred. Choose independently on each attempt."
        )
    return (
        original_problem.rstrip()
        + f"\n\n{MENU_START}\n"
        + menu.canonical_json
        + f"\n{MENU_END}"
        + neutral_choice
        + strategy_menu_natural_response_instructions(menu)
    )


def _load_split(root: pathlib.Path, split: str) -> Any:
    dataset = load_from_disk(str(root / split))
    return dataset[next(iter(dataset))]


def materialize(output_root: pathlib.Path) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(
            f"fresh {OUTPUT_LABEL} output required: {output_root}"
        )
    e49w = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    if (
        e49w.get("schema") != CALIBRATION_SCHEMA
        or e49w.get("pass") is not True
        or len(e49w.get(SELECTED_IDS_FIELD) or []) != 10
        or e49w.get("bidirectionally_executable_count", 0) < 10
    ):
        raise RuntimeError(
            f"{CALIBRATION_SCHEMA} has not passed its ten-problem route gate"
        )
    selected_ids = list(e49w[SELECTED_IDS_FIELD])
    by_problem = {
        row[PROBLEM_ID_FIELD]: row for row in e49w.get("problems", [])
    }
    if any(
        problem_id not in by_problem
        or by_problem[problem_id].get("bidirectionally_executable") is not True
        for problem_id in selected_ids
    ):
        raise RuntimeError(
            f"{CALIBRATION_SCHEMA} selected IDs are not all bidirectional"
        )

    e49u = json.loads(E49U.read_text(encoding="utf-8"))
    if (
        e49u.get("schema") != "e49u_05b_route_capability_calibration_v1"
        or e49u.get("problem_count") != 20
        or e49u.get("sample_count_per_route") != 8
    ):
        raise RuntimeError("E49U capability result is not the frozen audit")
    e49t_manifest = json.loads(E49T_MANIFEST.read_text(encoding="utf-8"))
    if (
        e49t_manifest.get("schema")
        != "e49t_natural_menu_materialization_v1"
        or e49t_manifest.get("row_counts") != {"train": 50, "eval": 50}
        or e49t_manifest.get("multi_support_counts")
        != {"train": 10, "eval": 10}
    ):
        raise RuntimeError("E49T source manifest drifted")

    full_train = _load_split(FULL, "train")
    full_by_id = {str(row["unique_id"]): dict(row) for row in full_train}
    selected_row_ids = {
        str(by_problem[problem_id]["row_id"]) for problem_id in selected_ids
    }
    old_train = _load_split(E49T_DATA, "train")
    singleton_rows: list[dict[str, Any]] = []
    dual_rows: list[tuple[int, dict[str, Any], MathStrategyMenu]] = []
    for row_index, raw in enumerate(old_train):
        menu = parse_strategy_menu(str(raw["problem"]))
        if menu is None:
            raise RuntimeError("E49T train row lost its menu")
        if len(menu.strategies) == 1:
            singleton_rows.append(dict(raw))
        else:
            dual_rows.append((row_index, dict(raw), menu))
    if len(singleton_rows) != 40 or len(dual_rows) != 10:
        raise RuntimeError("E49X expected 40 singleton and 10 dual train rows")

    pruned_train_fillers: list[dict[str, Any]] = []
    if WRITE_TRAIN_ROUTE_PROBE:
        retained_train = [
            row
            for row in singleton_rows
            if str(row["unique_id"]) not in selected_row_ids
        ]
        retained_e49t_train_singletons = len(retained_train)
        needed = 40 - len(retained_train)
        e49u_train = {
            int(problem["row_index"]): problem
            for problem in e49u["problems"]
            if problem["split"] == "train"
        }
        for row_index, raw, menu in dual_rows:
            if needed == 0:
                break
            if str(raw["unique_id"]) in selected_row_ids:
                continue
            capability = e49u_train.get(row_index)
            if capability is None:
                raise RuntimeError("E49U omitted an E49T dual train row")
            counts = {
                str(strategy_id): int(count)
                for strategy_id, count in capability[
                    "route_success_counts"
                ].items()
            }
            chosen = sorted(
                counts,
                key=lambda strategy_id: (-counts[strategy_id], strategy_id),
            )[0]
            selected_menu = _prune(menu, chosen)
            row = dict(raw)
            original_problem = str(row["original_problem"])
            row["problem"] = _embed(original_problem, selected_menu)
            row["strategy_menu_sha256"] = selected_menu.sha256
            row["strategy_menu_origin"] = (
                f"e49u_train_filler_pruned_from_{chosen}"
            )
            retained_train.append(row)
            pruned_train_fillers.append(
                {
                    "row_index": row_index,
                    "row_id": str(row["unique_id"]),
                    "retained_original_strategy_id": chosen,
                    "route_success_counts": counts,
                    "source_menu_sha256": menu.sha256,
                    "retained_menu_sha256": selected_menu.sha256,
                }
            )
            needed -= 1
        if needed:
            raise RuntimeError(
                "not enough nonoverlapping E49T rows for train fillers"
            )
    else:
        retained_train = singleton_rows
        retained_e49t_train_singletons = 40

    new_train = list(retained_train)
    route_probe_rows: list[dict[str, Any]] = []
    selected_menu_hashes = []
    for problem_id in selected_ids:
        result = by_problem[problem_id]
        row_id = str(result["row_id"])
        if row_id not in full_by_id:
            raise RuntimeError(
                f"calibration row is absent from exact OAT train: {row_id}"
            )
        row = dict(full_by_id[row_id])
        menu = _menu_from_payload(result["menu"])
        if len(menu.strategies) != 2 or menu.sha256 != result["menu_sha256"]:
            raise RuntimeError("selected calibration menu identity mismatch")
        original_problem = str(row["problem"])
        row["problem"] = _embed(original_problem, menu)
        row["original_problem"] = original_problem
        row["strategy_menu_sha256"] = menu.sha256
        row["strategy_menu_origin"] = CALIBRATION_ORIGIN
        new_train.append(row)
        route_probe_rows.append(dict(row))
        selected_menu_hashes.append(menu.sha256)
    if len(new_train) != 50:
        raise RuntimeError("E49X train row count drifted")
    if WRITE_TRAIN_ROUTE_PROBE and len(
        {str(row["unique_id"]) for row in new_train}
    ) != 50:
        raise RuntimeError("route-probe toy train rows are not unique")

    e49u_eval = {
        int(problem["row_index"]): problem
        for problem in e49u["problems"]
        if problem["split"] == "eval"
    }
    e49u_cases = {
        (int(case["row_index"]), str(case["strategy_id"])): case
        for case in e49u["cases"]
        if case["split"] == "eval"
    }
    old_eval = _load_split(E49T_DATA, "eval")
    new_eval: list[dict[str, Any]] = []
    pruned_eval: list[dict[str, Any]] = []
    retained_eval_dual = 0
    for row_index, raw in enumerate(old_eval):
        row = dict(raw)
        menu = parse_strategy_menu(str(row["problem"]))
        if menu is None:
            raise RuntimeError("E49T eval row lost its menu")
        selected_menu = menu
        origin = str(row["strategy_menu_origin"])
        if len(menu.strategies) == 2:
            capability = e49u_eval.get(row_index)
            if capability is None:
                raise RuntimeError("E49U omitted an E49T dual eval row")
            if capability["bidirectionally_executable"]:
                retained_eval_dual += 1
            else:
                counts = {
                    strategy.strategy_id: int(
                        e49u_cases[
                            (row_index, strategy.strategy_id)
                        ]["forced_route_success_count"]
                    )
                    for strategy in menu.strategies
                }
                chosen = sorted(
                    counts, key=lambda strategy_id: (-counts[strategy_id], strategy_id)
                )[0]
                selected_menu = _prune(menu, chosen)
                origin = f"e49u_pruned_from_{chosen}"
                pruned_eval.append(
                    {
                        "row_index": row_index,
                        "retained_original_strategy_id": chosen,
                        "route_success_counts": counts,
                        "source_menu_sha256": menu.sha256,
                        "retained_menu_sha256": selected_menu.sha256,
                    }
                )
        original_problem = str(row["original_problem"])
        row["problem"] = _embed(original_problem, selected_menu)
        row["strategy_menu_sha256"] = selected_menu.sha256
        row["strategy_menu_origin"] = origin
        new_eval.append(row)
    if retained_eval_dual != 1 or len(pruned_eval) != 9:
        raise RuntimeError("E49X evaluation pruning differs from E49U")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    max_prompt_tokens = 0
    for row in [*new_train, *new_eval]:
        prompt = apply_qwen_math_template(str(row["problem"]))
        token_count = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        max_prompt_tokens = max(max_prompt_tokens, token_count)
        if token_count > 2048:
            raise RuntimeError(
                f"E49X prompt exceeds 2048 tokens: {token_count}"
            )

    output_root.mkdir(parents=True)
    DatasetDict({"train": Dataset.from_list(new_train)}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict({"math": Dataset.from_list(new_eval)}).save_to_disk(
        str(output_root / "eval")
    )
    route_probe_hash = None
    if WRITE_TRAIN_ROUTE_PROBE:
        if len(route_probe_rows) != 10:
            raise RuntimeError("route probe must contain ten selected menus")
        DatasetDict(
            {"route_probe": Dataset.from_list(route_probe_rows)}
        ).save_to_disk(str(output_root / "route_probe"))
        route_probe_hash = _tree_sha256(output_root / "route_probe")
    train_multi = sum(
        len(parse_strategy_menu(str(row["problem"])).strategies) >= 2
        for row in new_train
    )
    eval_multi = sum(
        len(parse_strategy_menu(str(row["problem"])).strategies) >= 2
        for row in new_eval
    )
    if (train_multi, eval_multi) != (10, 1):
        raise RuntimeError("E49X final support counts drifted")
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "row_counts": {
            "train": 50,
            "eval": 50,
            **({"route_probe": 10} if WRITE_TRAIN_ROUTE_PROBE else {}),
        },
        "multi_support_counts": {"train": train_multi, "eval": eval_multi},
        "retained_e49t_train_singletons": (
            retained_e49t_train_singletons
        ),
        "pruned_train_fillers": pruned_train_fillers,
        "selected_calibration_problem_ids": selected_ids,
        "selected_calibration_menu_sha256": selected_menu_hashes,
        "pruned_eval": pruned_eval,
        "max_formatted_prompt_tokens": max_prompt_tokens,
        "train_tree_sha256": _tree_sha256(output_root / "train"),
        "eval_tree_sha256": _tree_sha256(output_root / "eval"),
        "route_probe_tree_sha256": route_probe_hash,
        "e49t_manifest_sha256": _sha256(E49T_MANIFEST),
        "e49u_result_sha256": _sha256(E49U),
        "calibration_schema": CALIBRATION_SCHEMA,
        "calibration_result_sha256": _sha256(CALIBRATION_RESULT),
        "full_source_manifest_sha256": _sha256(
            FULL / "MATERIALIZATION_MANIFEST.json"
        ),
        "protocol_sha256": _sha256(PROTOCOL),
        "materializer_sha256": _sha256(SCRIPT),
        "base_materializer_sha256": (
            _sha256(BASE_SCRIPT) if BASE_SCRIPT is not None else None
        ),
    }
    _write_json(output_root / "MATERIALIZATION_MANIFEST.json", manifest)
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
