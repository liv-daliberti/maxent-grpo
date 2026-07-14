#!/usr/bin/env python3
"""Build a small exact-answer-mode graph-coloring benchmark."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict

from make_modebench_data import (
    _graph_prompt,
    _graph_color_string,
    _valid_graph_colorings,
)


def _local_neighbor_graph_prompt(
    n: int,
    edges: list[list[int]],
    partial: list[int | None],
    *,
    state_missing_rule: bool = False,
) -> str:
    hidden = [index for index, color in enumerate(partial) if color is None]
    if len(hidden) != 1:
        return _graph_prompt(n, edges, partial)

    hidden_index = hidden[0]
    hidden_vertex = hidden_index + 1
    neighbor_bits = []
    for u, v in edges:
        other = None
        if u == hidden_vertex:
            other = v
        elif v == hidden_vertex:
            other = u
        if other is None:
            continue
        other_color = partial[other - 1]
        if other_color is not None:
            neighbor_bits.append(f"vertex {other} has color {int(other_color)}")
    neighbor_text = "; ".join(neighbor_bits) if neighbor_bits else "none are colored"
    edge_text = ", ".join(f"({u},{v})" for u, v in edges) if edges else "none"
    partial_text = _graph_color_string(partial)
    return (
        f"Color this graph with vertices 1 through {n}. The undirected edges are: "
        f"{edge_text}. A partial coloring is {partial_text}, where ? means the one "
        f"uncolored vertex. The uncolored vertex is {hidden_vertex}. Its colored "
        f"neighbors are: {neighbor_text}. Choose a color from {{1, 2, 3}} for "
        f"vertex {hidden_vertex} so no edge has equal colors at both endpoints. "
        + (
            "Because this is the only uncolored vertex, the correct answer is the "
            "color from {1, 2, 3} that is missing from those neighbor colors. "
            if state_missing_rule
            else ""
        )
        + "Return exactly one final answer and no explanation: exactly 1 digit, "
        "inside \\boxed{}."
    )


def _prompt_for_style(
    n: int,
    edges: list[list[int]],
    partial: list[int | None],
    *,
    prompt_style: str,
) -> str:
    if prompt_style == "original":
        return _graph_prompt(n, edges, partial)
    if prompt_style == "local_neighbors":
        return _local_neighbor_graph_prompt(n, edges, partial)
    if prompt_style == "local_missing_rule":
        return _local_neighbor_graph_prompt(
            n, edges, partial, state_missing_rule=True
        )
    raise ValueError(
        f"Unknown prompt_style={prompt_style!r}; "
        "use original, local_neighbors, or local_missing_rule."
    )


def _unique_hidden_completion_color(
    n: int, edges: list[list[int]], partial: list[int | None]
) -> int | None:
    hidden = [index for index, color in enumerate(partial) if color is None]
    if len(hidden) != 1:
        return None
    matches = [
        colors[hidden[0]]
        for colors in _valid_graph_colorings(n, edges)
        if all(color is None or colors[index] == color for index, color in enumerate(partial))
    ]
    if len(matches) != 1:
        return None
    return int(matches[0])


def _choose_partial_with_completion_range(
    *,
    n: int,
    edges: list[list[int]],
    target: list[int],
    rng: random.Random,
    hidden_count: int,
    min_completions: int,
    max_completions: int | None,
) -> tuple[list[int | None], int] | None:
    valid_colorings = _valid_graph_colorings(n, edges)
    if len(valid_colorings) < min_completions:
        return None
    hidden_count = max(1, min(int(hidden_count), n))
    indices = list(range(n))
    for _ in range(128):
        hidden = set(rng.sample(indices, hidden_count))
        partial: list[int | None] = [
            None if index in hidden else int(target[index]) for index in indices
        ]
        completion_count = sum(
            all(
                partial[index] is None or colors[index] == partial[index]
                for index in indices
            )
            for colors in valid_colorings
        )
        if completion_count < min_completions:
            continue
        if max_completions is not None and completion_count > max_completions:
            continue
        return partial, completion_count
    return None


def _synthetic_graph_rows(
    limit: int,
    *,
    seed: int,
    hidden_count: int,
    min_completions: int,
    max_completions: int | None,
    max_n: int,
    max_edges: int,
    min_solutions: int,
    split_tag: str,
    prompt_style: str,
    balance_hidden_color: bool,
    exclude: set[tuple[int, tuple[tuple[int, int], ...], str]] | None = None,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    # Seed the dedup set with prior splits' identities so evaluation prompts
    # are genuinely held out from training.
    seen: set[tuple[int, tuple[tuple[int, int], ...], str]] = (
        set(exclude) if exclude else set()
    )
    hidden_color_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    hidden_color_limits: dict[int, int] = {}
    if balance_hidden_color:
        base = int(limit) // 3
        remainder = int(limit) % 3
        hidden_color_limits = {
            color: base + (1 if color <= remainder else 0) for color in (1, 2, 3)
        }
    max_n = max(4, min(int(max_n), 6))
    max_possible_attempts = max(10_000, int(limit) * 1_000)
    for _ in range(max_possible_attempts):
        n = rng.randint(4, max_n)
        all_edges = [(u, v) for u in range(1, n + 1) for v in range(u + 1, n + 1)]
        edge_upper = min(int(max_edges), len(all_edges))
        edge_lower = min(edge_upper, max(1, n - 2))
        edge_count = rng.randint(edge_lower, edge_upper)
        edges = [list(edge) for edge in sorted(rng.sample(all_edges, edge_count))]
        valid_colorings = _valid_graph_colorings(n, edges)
        if len(valid_colorings) < int(min_solutions):
            continue
        target = rng.choice(valid_colorings)
        partial_bundle = _choose_partial_with_completion_range(
            n=n,
            edges=edges,
            target=target,
            rng=rng,
            hidden_count=hidden_count,
            min_completions=min_completions,
            max_completions=max_completions,
        )
        if partial_bundle is None:
            continue
        partial, completion_count = partial_bundle
        hidden_color = _unique_hidden_completion_color(n, edges, partial)
        if balance_hidden_color and hidden_color is not None:
            if hidden_color_counts[hidden_color] >= hidden_color_limits[hidden_color]:
                continue
        key = (n, tuple(tuple(edge) for edge in edges), _graph_color_string(partial))
        if key in seen:
            continue
        seen.add(key)
        if hidden_color is not None:
            hidden_color_counts[hidden_color] += 1
        spec = {
            "verifier": "graph_coloring",
            "n": n,
            "edges": edges,
            "partial_colors": partial,
            "source": "exact_answer_mode_graph_coloring",
            "instance_id": f"{split_tag}-{seed}-{len(rows)}",
            "num_solutions": len(valid_colorings),
            "num_completions": completion_count,
        }
        rows.append(
            {
                "problem": _prompt_for_style(
                    n, edges, partial, prompt_style=prompt_style
                ),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": "graph_coloring",
                "answer_mode_count": int(completion_count),
                "answer_mode_split": split_tag,
            }
        )
        if len(rows) >= int(limit):
            return rows
    raise RuntimeError(
        f"Only built {len(rows)} {split_tag} rows; requested {limit}. "
        "Relax graph filters or completion-count bounds."
    )


def _answer_mode_count_hist(rows: list[dict[str, str]]) -> dict[int, int]:
    counts = Counter(int(row["answer_mode_count"]) for row in rows)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--eval-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--max-edges", type=int, default=8)
    parser.add_argument("--multi-hidden-count", type=int, default=3)
    parser.add_argument("--multi-min-completions", type=int, default=4)
    parser.add_argument("--multi-max-completions", type=int, default=24)
    parser.add_argument("--unique-hidden-count", type=int, default=2)
    parser.add_argument(
        "--eval-splits",
        default="multi_answer,unique_answer",
        help=(
            "Comma-separated eval splits to write. Use 'multi_answer' for a "
            "multi-only validation set."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=["original", "local_neighbors", "local_missing_rule"],
        default="original",
    )
    parser.add_argument("--balance-hidden-color", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_size <= 0:
        raise SystemExit("--train-size must be positive")
    if args.eval_size <= 0:
        raise SystemExit("--eval-size must be positive")

    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    common = {
        "max_n": args.max_n,
        "max_edges": args.max_edges,
        "prompt_style": args.prompt_style,
        "balance_hidden_color": args.balance_hidden_color,
    }
    train_rows = _synthetic_graph_rows(
        args.train_size,
        seed=args.seed,
        hidden_count=args.multi_hidden_count,
        min_completions=args.multi_min_completions,
        max_completions=args.multi_max_completions,
        min_solutions=args.multi_min_completions,
        split_tag="train_multi_answer",
        **common,
    )
    def _row_identity(row: dict[str, str]) -> tuple:
        spec = json.loads(row["answer"])
        return (
            int(spec["n"]),
            tuple(tuple(edge) for edge in spec["edges"]),
            _graph_color_string(spec["partial_colors"]),
        )

    train_identities = {_row_identity(row) for row in train_rows}
    eval_multi_rows = _synthetic_graph_rows(
        args.eval_size,
        seed=args.seed + 10_000,
        hidden_count=args.multi_hidden_count,
        min_completions=args.multi_min_completions,
        max_completions=args.multi_max_completions,
        min_solutions=args.multi_min_completions,
        split_tag="eval_multi_answer",
        exclude=train_identities,
        **common,
    )
    eval_splits = {
        split.strip()
        for split in str(args.eval_splits).split(",")
        if split.strip()
    }
    allowed_splits = {"multi_answer", "unique_answer"}
    unknown_splits = eval_splits - allowed_splits
    if unknown_splits:
        raise SystemExit(
            f"Unknown --eval-splits values: {sorted(unknown_splits)}. "
            f"Allowed: {sorted(allowed_splits)}"
        )
    if not eval_splits:
        raise SystemExit("--eval-splits must include at least one split")
    eval_unique_rows = (
        _synthetic_graph_rows(
            args.eval_size,
            seed=args.seed + 20_000,
            hidden_count=args.unique_hidden_count,
            min_completions=1,
            max_completions=1,
            min_solutions=1,
            split_tag="eval_unique_answer",
            exclude=train_identities,
            **common,
        )
        if "unique_answer" in eval_splits
        else []
    )

    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        str(output_root / "train")
    )
    eval_dict = {}
    if "multi_answer" in eval_splits:
        eval_dict["multi_answer"] = Dataset.from_list(eval_multi_rows)
    if "unique_answer" in eval_splits:
        eval_dict["unique_answer"] = Dataset.from_list(eval_unique_rows)
    DatasetDict(eval_dict).save_to_disk(str(output_root / "eval"))
    print(
        "wrote train={} eval={} rows={}/{}/{}".format(
            output_root / "train",
            output_root / "eval",
            len(train_rows),
            len(eval_multi_rows),
            len(eval_unique_rows),
        )
    )
    print(
        "[exact-answer-mode] answer_mode_count_hist "
        f"train={_answer_mode_count_hist(train_rows)} "
        f"eval_multi={_answer_mode_count_hist(eval_multi_rows)} "
        f"eval_unique={_answer_mode_count_hist(eval_unique_rows)}"
    )
    if max(_answer_mode_count_hist(eval_multi_rows), default=0) <= 1:
        print(
            "[exact-answer-mode] WARNING: eval split named multi_answer has no "
            "multi-completion rows."
        )


if __name__ == "__main__":
    main()
