#!/usr/bin/env python3
"""What RLEP-Dr actually replays: the mode composition of its experience pools.

RLEP's protocol harvests its experience pool from a policy that has already
been trained by RL (its "seed policy"), keeping verified answers sampled at
temperature .7 and requiring at least two per question.  In this benchmark the
seed policy is the paired terminal Dr.GRPO control, so the pool inherits that
policy's collapse.  This generator measures the consequence directly from the
frozen collection sidecars and the learner telemetry of every RLEP-Dr cell:

* how many of the 384 training prompts had no verified trajectory at all
  (those prompts never receive a replay row);
* among replay-eligible prompts, how many canonical solution modes the pool
  holds, and the probability that a two-row replay draw is two copies of the
  same mode;
* the fraction of optimizer updates on which replay rows were actually mixed
  in, and the realized replay advantage ``1 - mean(mixed reward)`` on those
  updates, which is the coefficient RLEP's protocol assigns to its replay rows.

The Re:Dr buffer occupancy from the within-cohort E132 cells is reported
beside it, from the same ``banked_modes`` telemetry, so the reader can see
what an online buffer holds on the same prompts and seeds.

The pool composition is the decisive quantity for breadth: no update rule can
rehearse a second mode from a pool that holds one.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEDGERS = {
    "Qwen2.5-0.5B": [
        ROOT / "var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json",
        ROOT / "var/artifacts/e116_sparse_rlep_qwen05b_domain_extension_jobs.json",
    ],
    "Falcon3-1B": [
        ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json",
    ],
}
REDR_LEDGER = ROOT / "var/artifacts/e132_matched_redr_05b_jobs.json"
TRAIN_PROMPTS = 384
FRESH_ROWS = 16
REPLAY_ROWS = 2
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
LABEL = {
    "graph_coloring": r"\texttt{Graph}",
    "countdown": r"\texttt{Countdown}",
    "python_factors": r"\texttt{Python}",
    "mathir": r"\texttt{MathIR}",
    "pantry_plan": r"\texttt{PantryPlan}",
}
SCALE_TAG = {"Qwen2.5-0.5B": "qwen", "Falcon3-1B": "falcon"}
SCHEMA = "paper-rlep-pool-composition-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float, places: int = 3) -> str:
    text = f"{value:.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def pct(value: float) -> str:
    return f"{round(100 * value)}\\%"


def pool_composition(sidecar: Path) -> dict:
    """Verified trajectories per prompt and the canonical keys they carry."""

    verified: dict[int, list[str]] = collections.defaultdict(list)
    draws = 0
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("draw_index") is None:
            continue
        draws += 1
        for row in record["prompts"]:
            keys, rewards = row["answer_keys"], row["rewards"]
            if len(keys) != len(rewards):
                raise ValueError(f"{sidecar}: keys and rewards misaligned")
            for key, reward in zip(keys, rewards):
                if float(reward) > 0.0:
                    verified[int(row["prompt_index"])].append(str(key))
    if draws != 4:
        raise ValueError(f"{sidecar}: expected four collection draws, found {draws}")
    eligible = [keys for keys in verified.values() if len(keys) >= REPLAY_ROWS]
    distinct = [len(set(keys)) for keys in eligible]
    same_pair = []
    dominant = []
    for keys in eligible:
        counts = collections.Counter(keys)
        m = len(keys)
        same_pair.append(sum(c * (c - 1) for c in counts.values()) / (m * (m - 1)))
        dominant.append(max(counts.values()) / m)
    return {
        "prompts_with_no_verified_trajectory": TRAIN_PROMPTS - len(verified),
        "prompts_with_one_verified_trajectory": sum(
            1 for keys in verified.values() if len(keys) == 1
        ),
        "eligible_prompts": len(eligible),
        "eligible_prompts_with_one_key": sum(1 for d in distinct if d == 1),
        "keys_per_eligible_prompt": st.fmean(distinct) if distinct else float("nan"),
        "max_keys_on_any_eligible_prompt": max(distinct) if distinct else 0,
        "p_same_key_pair": st.fmean(same_pair) if same_pair else float("nan"),
        "dominant_key_share": st.fmean(dominant) if dominant else float("nan"),
        "verified_trajectories": sum(len(keys) for keys in verified.values()),
    }


def replay_telemetry(run_dir: Path) -> dict:
    """Realized replay eligibility and advantage over the terminal attempt."""

    complete = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    metrics = Path(complete["terminal_attempt"]) / "train_metrics.jsonl"
    eligible: list[float] = []
    advantage: list[float] = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        flag = row.get("train/rlep_replay_eligible")
        if flag is None:
            continue
        eligible.append(float(flag))
        if flag:
            advantage.append(float(row["train/rlep_replay_advantage"]))
            if int(row["train/rlep_replay_rows"]) != REPLAY_ROWS:
                raise ValueError(f"{metrics}: replay dose differs from {REPLAY_ROWS}")
    # A row per optimizer update, less the handful of evaluation-only rows.
    if len(eligible) < 0.99 * complete["terminal_step"]:
        raise ValueError(f"{metrics}: telemetry shorter than the terminal step")
    return {
        "updates": len(eligible),
        "replay_update_fraction": st.fmean(eligible),
        "mean_replay_advantage_when_eligible": (
            st.fmean(advantage) if advantage else float("nan")
        ),
        "terminal_step": complete["terminal_step"],
    }


def redr_bank_occupancy() -> dict[str, dict]:
    ledger = json.loads(REDR_LEDGER.read_text(encoding="utf-8"))
    per_domain: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for run in ledger["runs"]:
        complete = json.loads(
            (Path(run["run_dir"]) / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
        )
        metrics = Path(complete["terminal_attempt"]) / "train_metrics.jsonl"
        banked: list[float] = []
        for line in metrics.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get("train/canonical_replay_banked_modes")
            if value is not None:
                banked.append(float(value))
        per_domain[run["domain"]].append(
            (st.fmean(banked), sum(1 for b in banked if b >= 2) / len(banked))
        )
    return {
        domain: {
            "cells": len(values),
            "mean_banked_modes": st.fmean(v[0] for v in values),
            "fraction_of_updates_with_two_or_more_modes": st.fmean(v[1] for v in values),
        }
        for domain, values in per_domain.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    parser.add_argument("--stamp", default=dt.date.today().isoformat().replace("-", ""))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    provenance: list[dict] = []
    for scale, ledgers in LEDGERS.items():
        for ledger_path in ledgers:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            provenance.append({"ledger": str(ledger_path), "sha256": sha256(ledger_path)})
            for run in ledger["runs"]:
                if run["arm"] != "rlep_dr_sparse":
                    raise SystemExit(f"{ledger_path}: unexpected arm {run['arm']}")
                sidecars = sorted(Path(run["pool_root"]).glob("**/eval_mode_coverage_draws.jsonl"))
                if len(sidecars) != 1:
                    raise SystemExit(
                        f"{run['run_stamp']}: expected one pool sidecar, found {len(sidecars)}"
                    )
                cells.append(
                    {
                        "scale": scale,
                        "domain": run["domain"],
                        "seed": int(run["seed"]),
                        "run_stamp": run["run_stamp"],
                        "pool_sidecar": str(sidecars[0]),
                        "pool": pool_composition(sidecars[0]),
                        "telemetry": replay_telemetry(Path(run["run_dir"])),
                    }
                )

    def mean_over(rows: list[dict], group: str, key: str) -> float:
        values = [r[group][key] for r in rows if r[group][key] == r[group][key]]
        return st.fmean(values) if values else float("nan")

    domains: dict[str, dict[str, dict]] = {}
    for scale in LEDGERS:
        domains[scale] = {}
        for domain in DOMAINS:
            rows = [c for c in cells if c["scale"] == scale and c["domain"] == domain]
            if not rows:
                continue
            domains[scale][domain] = {
                "cells": len(rows),
                "seeds": sorted(r["seed"] for r in rows),
                "no_verified_fraction": mean_over(rows, "pool", "prompts_with_no_verified_trajectory")
                / TRAIN_PROMPTS,
                "eligible_fraction": mean_over(rows, "pool", "eligible_prompts") / TRAIN_PROMPTS,
                "single_key_fraction": st.fmean(
                    r["pool"]["eligible_prompts_with_one_key"] / r["pool"]["eligible_prompts"]
                    for r in rows
                ),
                "keys_per_eligible_prompt": mean_over(rows, "pool", "keys_per_eligible_prompt"),
                "p_same_key_pair": mean_over(rows, "pool", "p_same_key_pair"),
                "replay_update_fraction": mean_over(rows, "telemetry", "replay_update_fraction"),
                "mean_replay_advantage": mean_over(
                    rows, "telemetry", "mean_replay_advantage_when_eligible"
                ),
            }

    redr = redr_bank_occupancy()

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "seed_policy": "paired terminal Dr.GRPO control of the same seed",
            "collection": "64 candidates per prompt (four draws of 16), temperature .7, top-p .95",
            "eligibility": f"at least {REPLAY_ROWS} verified trajectories on the prompt",
            "update": f"{FRESH_ROWS} fresh + {REPLAY_ROWS} replay rows under one common baseline",
            "replay_row_coefficient": "(1 - mean mixed reward) / (fresh + replay rows)",
        },
        "provenance": provenance + [{"ledger": str(REDR_LEDGER), "sha256": sha256(REDR_LEDGER)}],
        "domains": domains,
        "redr_bank_occupancy_e132": redr,
        "cells": cells,
    }
    (out / f"rlep_pool_composition_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    qwen = domains["Qwen2.5-0.5B"]
    falcon = domains["Falcon3-1B"]
    qwen_cells = [c for c in cells if c["scale"] == "Qwen2.5-0.5B"]
    falcon_cells = [c for c in cells if c["scale"] == "Falcon3-1B"]
    qwen_single = all(
        c["pool"]["eligible_prompts_with_one_key"] == c["pool"]["eligible_prompts"]
        for c in qwen_cells
    )
    multi = ("graph_coloring", "countdown", "pantry_plan")
    macros = [
        "% Generated by ops/build_paper_rlep_pool_composition.py; do not hand edit.",
        rf"\newcommand{{\RPqwencells}}{{{len(qwen_cells)}}}",
        rf"\newcommand{{\RPfalconcells}}{{{len(falcon_cells)}}}",
        rf"\newcommand{{\RPqwenmaxkeys}}{{{max(c['pool']['max_keys_on_any_eligible_prompt'] for c in qwen_cells)}}}",
        rf"\newcommand{{\RPqwensinglekeyall}}{{{'every' if qwen_single else 'not every'}}}",
        rf"\newcommand{{\RPqwensinglekeyshare}}{{{pct(sum(c['pool']['eligible_prompts_with_one_key'] for c in qwen_cells) / sum(c['pool']['eligible_prompts'] for c in qwen_cells))}}}",
        rf"\newcommand{{\RPfalconsinglekeyshare}}{{{pct(sum(c['pool']['eligible_prompts_with_one_key'] for c in falcon_cells) / sum(c['pool']['eligible_prompts'] for c in falcon_cells))}}}",
        rf"\newcommand{{\RPqweneligibleprompts}}{{{sum(c['pool']['eligible_prompts'] for c in qwen_cells)}}}",
        rf"\newcommand{{\RPqwenmultikeyprompts}}{{{sum(c['pool']['eligible_prompts'] - c['pool']['eligible_prompts_with_one_key'] for c in qwen_cells)}}}",
        rf"\newcommand{{\RPqwensamepairlo}}{{{fmt(min(v['p_same_key_pair'] for v in qwen.values()), 2)}}}",
        rf"\newcommand{{\RPqwennoverifiedlo}}{{{pct(min(v['no_verified_fraction'] for v in qwen.values()))}}}",
        rf"\newcommand{{\RPqwennoverifiedhi}}{{{pct(max(v['no_verified_fraction'] for v in qwen.values()))}}}",
        rf"\newcommand{{\RPqwenupdatelo}}{{{pct(min(v['replay_update_fraction'] for v in qwen.values()))}}}",
        rf"\newcommand{{\RPqwenupdatehi}}{{{pct(max(v['replay_update_fraction'] for v in qwen.values()))}}}",
        rf"\newcommand{{\RPqwenadvlo}}{{{fmt(min(v['mean_replay_advantage'] for v in qwen.values()), 2)}}}",
        rf"\newcommand{{\RPqwenadvhi}}{{{fmt(max(v['mean_replay_advantage'] for v in qwen.values()), 2)}}}",
        rf"\newcommand{{\RPfalconsamepairlo}}{{{fmt(min(v['p_same_key_pair'] for v in falcon.values()), 2)}}}",
        rf"\newcommand{{\RPfalconsamepairhi}}{{{fmt(max(v['p_same_key_pair'] for v in falcon.values()), 2)}}}",
        rf"\newcommand{{\RPfalconkeyshi}}{{{fmt(max(v['keys_per_eligible_prompt'] for v in falcon.values()), 2)}}}",
        rf"\newcommand{{\RPredrbanklo}}{{{fmt(min(redr[d]['mean_banked_modes'] for d in multi), 1)}}}",
        rf"\newcommand{{\RPredrbankhi}}{{{fmt(max(redr[d]['mean_banked_modes'] for d in multi), 1)}}}",
    ]
    for scale, tag in SCALE_TAG.items():
        for domain, v in domains[scale].items():
            dtag = domain.split("_")[0]
            macros.append(rf"\newcommand{{\RP{tag}{dtag}keys}}{{{fmt(v['keys_per_eligible_prompt'], 2)}}}")
            macros.append(rf"\newcommand{{\RP{tag}{dtag}update}}{{{pct(v['replay_update_fraction'])}}}")
    (out / f"rlep_pool_composition_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    body = ["% Generated by ops/build_paper_rlep_pool_composition.py; do not hand edit."]
    for scale in LEDGERS:
        body.append(rf"  \multicolumn{{9}}{{@{{}}l}}{{\textit{{{scale}}}}} \\")
        for domain in DOMAINS:
            v = domains[scale].get(domain)
            if v is None:
                continue
            bank = redr.get(domain) if scale == "Qwen2.5-0.5B" else None
            body.append(
                f"  {LABEL[domain]} & {v['cells']}"
                f" & {pct(v['no_verified_fraction'])}"
                f" & {pct(v['eligible_fraction'])}"
                f" & {fmt(v['keys_per_eligible_prompt'], 2)}"
                f" & {fmt(v['p_same_key_pair'], 2)}"
                f" & {pct(v['replay_update_fraction'])}"
                f" & {fmt(v['mean_replay_advantage'], 2)}"
                + (f" & {fmt(bank['mean_banked_modes'], 2)}" if bank else r" & \textemdash{}")
                + r" \\"
            )
    body.append(r"  \bottomrule")
    (out / f"rlep_pool_composition_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )

    print(f"wrote rlep_pool_composition_{args.stamp}.{{json,_macros.tex,_table_body.tex}}")
    for scale in LEDGERS:
        print(f"  {scale}")
        for domain, v in domains[scale].items():
            print(
                f"    {domain:16s} n={v['cells']} no-verified {pct(v['no_verified_fraction'])}"
                f" eligible {pct(v['eligible_fraction'])} keys/eligible {fmt(v['keys_per_eligible_prompt'],2)}"
                f" P(same pair) {fmt(v['p_same_key_pair'],2)} replay updates {pct(v['replay_update_fraction'])}"
                f" adv {fmt(v['mean_replay_advantage'],2)}"
            )
    print("  Re:Dr (E132) mean banked modes: " + ", ".join(
        f"{d} {fmt(redr[d]['mean_banked_modes'],2)}" for d in DOMAINS if d in redr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
