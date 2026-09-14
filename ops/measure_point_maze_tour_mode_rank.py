#!/usr/bin/env python3
"""Which certified tours does a policy actually keep?

PointMaze Tour's modes are not exchangeable: every certified tour carries a
measured step cost, so they admit a cost ordering. Graph colorings do not --
one valid colouring is as arbitrary as another. If a policy concentrates on the
cheapest tours, that ordering is an attractor, and breadth on this domain is
intrinsically harder to hold than on a domain whose modes are symmetric.

This records, for each successful sample, the cost rank of the tour it produced
within that map's execution-certified list (rank 1 = cheapest). It reports the
rank distribution rather than a count, so "which modes survive" is measured
instead of assumed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1]))
for path in (ROOT / "src", ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def rank_table(identity: dict, split: str) -> dict[str, dict[str, int]]:
    """map_id -> {order string: 1-based rank by measured step cost}."""

    table = {}
    for entry in identity["splits"][split]["maps"]:
        ordered = sorted(entry["certified_orders"], key=lambda row: row["steps"])
        table[entry["map_id"]] = {
            ">".join(row["order"]): index + 1 for index, row in enumerate(ordered)
        }
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import train_point_maze_tour as tour
    from oat_drgrpo.point_maze_tour_policy import (
        POINT_TOUR_LABELS,
        render_point_tour_terminal_padding_prompt,
    )
    from oat_drgrpo.point_maze_tour_process import PointMazeTourProcess

    identity = json.loads((args.data_root / "identity.json").read_text())
    ranks = rank_table(identity, args.split)
    rows = load_from_disk(str(args.data_root))[args.split].to_list()[: args.prompts]
    horizon = len(json.loads(rows[0]["answer"])["landmarks"])

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_token_ids = {}
    for label in POINT_TOUR_LABELS[:horizon]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_token_ids[label] = int(ids[0])
    padding_token_ids = tokenizer.encode(
        render_point_tour_terminal_padding_prompt(), add_special_tokens=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False

    counts: Counter = Counter()
    successes = 0
    with PointMazeTourProcess(worker_python=args.worker_python) as worker:
        for index, row in enumerate(rows):
            _e, _s, outcomes, _d = tour.run_episodes(
                model=model, tokenizer=tokenizer, worker=worker, row=row,
                sample_count=8, session_prefix=f"rank-{args.label}-m{index}",
                seed_base=args.seed + index * 10_000,
                label_token_ids=label_token_ids,
                padding_token_ids=padding_token_ids,
                max_length=1536, record_decisions=False,
                expected_horizon=horizon,
            )
            map_id = json.loads(row["answer"])["map_id"]
            for key in outcomes:
                if key is None:
                    continue
                successes += 1
                order = str(key).rsplit(":", 1)[-1]
                counts[ranks.get(map_id, {}).get(order, 0)] += 1

    total = sum(counts.values())
    payload = {
        "schema": "point-maze-tour-mode-rank-v1",
        "label": args.label,
        "model": str(args.model),
        "split": args.split,
        "prompts": len(rows),
        "successful_samples": successes,
        "rank_counts": {str(k): v for k, v in sorted(counts.items())},
        "share_rank1": counts[1] / total if total else 0.0,
        "share_top3": sum(counts[r] for r in (1, 2, 3)) / total if total else 0.0,
        "mean_rank": (
            sum(r * n for r, n in counts.items() if r) / sum(n for r, n in counts.items() if r)
            if any(r for r in counts) else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("label", "successful_samples", "share_rank1", "share_top3", "mean_rank")}))


if __name__ == "__main__":
    main()
