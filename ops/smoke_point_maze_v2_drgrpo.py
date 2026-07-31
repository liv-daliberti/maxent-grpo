#!/usr/bin/env python3
"""Short Dr.GRPO smoke on PointMaze v2: does a sparse signal compound or stall?

The zero-shot probe found the v1 warmstart succeeds on ~3.5% of rollouts over v2
maps -- enough that binary reward has non-zero group variance, but close to the
floor. This runs plain Dr.GRPO (no xGRPO terms) for a couple of passes and
evaluates periodically, to see whether that sparse signal compounds.

It reuses the frozen Stage-B rollout, backward and evaluation code rather than
reimplementing them, so the dynamics are the campaign's, not a lookalike. What
it deliberately does NOT reuse is the campaign identity/qualification gating,
which is bound to the v1 cohort -- this is a diagnostic, not a registered run,
and it writes no campaign artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
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
    ap.add_argument("--train-prompts", type=int, default=24)
    ap.add_argument("--eval-prompts", type=int, default=8)
    ap.add_argument("--updates", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=24)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--learning-rate", type=float, default=2e-7)
    ap.add_argument("--microbatch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "var/artifacts/point_maze_v2_drgrpo_smoke.json")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="save/resume model+optimizer here; required for multi-day runs")
    ap.add_argument("--checkpoint-every", type=int, default=48)
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import train_point_maze_interactive_paired_smoke_v1 as base
    import train_point_maze_stage_b_05b_12pass as T
    from oat_drgrpo.point_maze_interactive_policy import POINT_POLICY_LABELS
    from oat_drgrpo.point_maze_interactive_process import PointMazeInteractiveProcess
    from oat_drgrpo.interactive_episode_replay import VerifiedInteractiveReplayBank
    from oat_drgrpo.interactive_episode_objective import drgrpo_task_advantages

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("smoke requires a BF16 GPU")

    train_rows = load_from_disk(str(args.data_root / "train"))["train"].to_list()
    eval_rows = load_from_disk(str(args.data_root / "eval"))["multi_answer"].to_list()
    train_rows = train_rows[: args.train_prompts]
    eval_rows = eval_rows[: args.eval_prompts]
    # scale the frozen evaluator's v1 split cardinality to this probe's size
    T.EVAL_PROMPTS = len(eval_rows)
    T.EVAL_TRAJECTORIES = len(eval_rows) * (1 + T.EVAL_DRAWS * T.EVAL_K)
    T.TRAIN_PROMPTS = len(train_rows)

    print(f"Dr.GRPO smoke: {len(train_rows)} train / {len(eval_rows)} eval maps, "
          f"{args.updates} updates, seed {args.seed}", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token_ids = []
    for label in POINT_POLICY_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"label not one token: {label}")
        action_token_ids.append(int(ids[0]))
    padding_token_ids = tokenizer.encode(T.POINT_TERMINAL_PADDING_PROMPT,
                                         add_special_tokens=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate),
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    replay_bank = VerifiedInteractiveReplayBank(capacity=base.REPLAY_CAPACITY)

    # A 4,608-update run spans days and will outlive at least one wall limit, so
    # checkpoint the full training state and resume from it rather than restart.
    start_update = 0
    curve, train_log = [], []
    ckpt = args.checkpoint_dir
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)
        latest = ckpt / "latest.pt"
        if latest.is_file():
            state = torch.load(latest, map_location="cuda", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_update = int(state["update"])
            curve = state.get("curve", [])
            train_log = state.get("train_log", [])
            print(f"resumed from update {start_update}", flush=True)

    def save_checkpoint(update):
        if ckpt is None:
            return
        tmp = ckpt / "latest.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "update": update, "curve": curve, "train_log": train_log}, tmp)
        tmp.replace(ckpt / "latest.pt")
        print(f"  [checkpoint @ {update}]", flush=True)

    with PointMazeInteractiveProcess(worker_python=args.worker_python) as worker:
        def run_eval(update):
            model.eval()
            with torch.no_grad():
                r = T.evaluate(model=model, tokenizer=tokenizer, worker=worker,
                               rows=eval_rows, arm="grpo", seed=args.seed,
                               update=update, action_token_ids=action_token_ids,
                               max_length=args.max_length, batch_size=16)
            row = {k: r.get(k) for k in ("greedy", "mean8", "pass8", "distinct8")}
            row["update"] = update
            curve.append(row)
            print(f"  [eval @ {update:>3}] greedy {row['greedy']:.4f}  "
                  f"mean@8 {row['mean8']:.4f}  pass@8 {row['pass8']:.4f}  "
                  f"distinct@8 {row['distinct8']:.4f}", flush=True)

        if start_update == 0:
            run_eval(0)
        verified_total = sum(r["verified_episodes"] for r in train_log)
        for update_index in range(start_update + 1, args.updates + 1):
            row_index = (update_index - 1) % len(train_rows)
            row = train_rows[row_index]
            model.eval()
            episodes, policy_slots, rollout = base._rollout_group(
                model=model, tokenizer=tokenizer, worker=worker, row=row,
                row_index=row_index, update_index=update_index, arm="grpo",
                seed=args.seed, action_token_ids=action_token_ids,
                padding_token_ids=padding_token_ids, max_length=args.max_length,
            )
            verified = int(rollout.get("verified_episodes", 0))
            verified_total += verified
            rewards = torch.tensor([[e.task_reward for e in episodes]],
                                   dtype=torch.float32)
            task_advantages = drgrpo_task_advantages(rewards).flatten()
            decision_counts = [len(e.decisions) for e in episodes]
            # plain Dr.GRPO: task advantage only, no exploration term
            for slot in policy_slots:
                if slot["active"]:
                    i = int(slot["episode"])
                    slot["weight"] = (float(task_advantages[i].item())
                                      / decision_counts[i] / base.SAMPLES)
            replay_bank.observe_group(episodes)
            replay_group = replay_bank.schedule_one_global_round_robin()
            replay_slots, _ = base._replay_slots(
                group=replay_group, padding_token_ids=padding_token_ids,
                action_token_ids=action_token_ids, compute_only=True,
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            base._backward_fixed_slots(
                model=model, tokenizer=tokenizer,
                slots=[*policy_slots, *replay_slots],
                action_token_ids=action_token_ids,
                microbatch_size=args.microbatch_size,
            )
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(gn.detach().item())):
                raise RuntimeError("nonfinite gradient")
            optimizer.step()
            train_log.append({"update": update_index, "verified_episodes": verified,
                              "task_advantage_rms": float(torch.sqrt(
                                  torch.mean(task_advantages.square())).item()),
                              "grad_norm": float(gn.detach().item())})
            if update_index % 8 == 0:
                recent = train_log[-8:]
                print(f"  [train @ {update_index:>3}] verified/16 last8="
                      f"{sum(r['verified_episodes'] for r in recent)}/{8*base.SAMPLES}"
                      f"  adv_rms {recent[-1]['task_advantage_rms']:.4f}", flush=True)
            if update_index % args.eval_every == 0:
                run_eval(update_index)
            if update_index % args.checkpoint_every == 0:
                save_checkpoint(update_index)
        save_checkpoint(args.updates)

    payload = {
        "schema": "point-maze-v2-drgrpo-smoke-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.relative_to(ROOT)),
        "data_root": str(args.data_root.relative_to(ROOT)),
        "arm": "grpo (plain Dr.GRPO, no xGRPO terms)",
        "train_prompts": len(train_rows), "eval_prompts": len(eval_rows),
        "updates": args.updates, "seed": args.seed,
        "verified_episodes_total": verified_total,
        "verified_rollout_fraction": verified_total / (args.updates * base.SAMPLES),
        "curve": curve, "training": train_log,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\n=== curve ===")
    for r in curve:
        print(f"  update {r['update']:>3}: greedy {r['greedy']:.4f}  "
              f"mean@8 {r['mean8']:.4f}  pass@8 {r['pass8']:.4f}  "
              f"distinct@8 {r['distinct8']:.4f}")
    first, last = curve[0], curve[-1]
    d = last["pass8"] - first["pass8"]
    print(f"\nverified rollouts: {verified_total}/{args.updates*base.SAMPLES} "
          f"({100*payload['verified_rollout_fraction']:.1f}%)")
    print(f"pass@8 {first['pass8']:.4f} -> {last['pass8']:.4f} ({d:+.4f})")
    print("  -> signal COMPOUNDS" if d > 0.02 else
          "  -> STALLED (no material improvement at this budget)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
