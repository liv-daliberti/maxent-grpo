#!/usr/bin/env python3
"""Can a 0.5B policy solve the redesigned PointMaze maps at all?

Binary-reward RL has nothing to optimise from a policy that never succeeds: with
zero verified episodes every group advantage is zero and Dr.GRPO makes no update.
So before committing a training run, measure what the starting policy actually
does on the v2 evaluation split.

Reports greedy pass@1, sampled pass@8, and distinct@8 over the same evaluation
machinery the Stage-B trainer uses, so the numbers are directly comparable to
the v1 campaign. Reads the warmstart checkpoint the PointMaze campaigns start
from -- it was SFT'd on v1 geometries, so this also measures whether that
warmstart transfers to the new maps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path,
                    default=ROOT / "var/models/point_maze_interactive_warmstart_v3")
    ap.add_argument("--data-root", type=Path,
                    default=ROOT / "var/data/point_maze_modebench_v2_384")
    ap.add_argument("--worker-python", type=Path,
                    default=ROOT / "var/maze_runtime/venv/bin/python")
    ap.add_argument("--rows", type=int, default=16,
                    help="evaluation maps to probe (0 = all 128)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "var/artifacts/point_maze_v2_zero_shot_probe.json")
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import train_point_maze_stage_b_05b_12pass as T
    from oat_drgrpo.point_maze_interactive_policy import POINT_POLICY_LABELS
    from oat_drgrpo.point_maze_interactive_process import PointMazeInteractiveProcess

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("probe requires a BF16 GPU")

    rows = load_from_disk(str(args.data_root / "eval"))["multi_answer"].to_list()
    if args.rows:
        rows = rows[: args.rows]
    print(f"probing {len(rows)} v2 evaluation maps from {args.data_root.name}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token_ids = []
    for label in POINT_POLICY_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"label not a single token: {label}")
        action_token_ids.append(int(ids[0]))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.eval()

    # evaluate() asserts v1's fixed trajectory count (4 prompts x 33). The
    # arithmetic is identical for any row count, so scale the expectation rather
    # than being pinned to a four-map evaluation split.
    T.EVAL_PROMPTS = len(rows)
    T.EVAL_TRAJECTORIES = len(rows) * (1 + T.EVAL_DRAWS * T.EVAL_K)

    worker = PointMazeInteractiveProcess(worker_python=args.worker_python)
    try:
        with torch.no_grad():
            result = T.evaluate(
                model=model, tokenizer=tokenizer, worker=worker, rows=rows,
                arm="grpo", seed=43, update=0,
                action_token_ids=action_token_ids,
                max_length=args.max_length, batch_size=args.batch_size,
            )
    finally:
        worker.close()

    payload = {
        "schema": "point-maze-v2-zero-shot-probe-v1",
        "model": str(args.model.relative_to(ROOT)),
        "data_root": str(args.data_root.relative_to(ROOT)),
        "rows_probed": len(rows),
        "metrics": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    keys = ("greedy", "pass8", "mean8", "distinct8")
    print("\n=== zero-shot on PointMaze v2 ===")
    for k in keys:
        if k in result:
            print(f"  {k:10s} {result[k]:.4f}")
    verdict = (result.get("pass8") or 0) > 0
    print(f"\n  verified episodes present: {verdict}")
    print("  -> Dr.GRPO has a gradient to work with" if verdict else
          "  -> ZERO successes: binary reward gives no signal; training cannot start here")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
