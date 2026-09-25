#!/usr/bin/env python3
"""The four-arm coding campaign (2026-09-24), paired seed by seed within cohort.

Twenty runs on the hardened CodeContests slate: Dr.GRPO, Re:Dr, MaxRL and
Re:Max, five seeds each, Qwen2.5-Coder-7B-Instruct with a rank-16 LoRA, 128
optimizer updates of one sixteen-rollout group over eight training tasks, and
a final eight-draw evaluation on thirteen held-out development tasks. All four
arms share seeds, prompts, placement, runtime and evaluator, so every contrast
is a within-cohort paired quantity, as E132 made the Level-1 contrasts.

Two things this builder settles, and one it cannot.

**The replay factor is what the logs say it is.** The result receipt's
``objective`` string is hardcoded to the MaxRL advantage for every arm, so the
objective is keyed on ``arm`` and the replay factor is read from
``canonical_replay_compute_only`` at every logged update: zero means the
replay derivative was live. A cell whose log disagrees with its arm is refused.

**Replay raises fresh success during training.** Over the 2,048 fresh
rollouts each run draws, the replay arm verifies more of them than its
matched control in nine of ten pairs; that difference is emitted paired, with
a Student-t interval over the five seeds.

**The development endpoint is at floor and cannot separate the arms.** With
eight draws per task, every arm on every seed solves exactly one of the
thirteen development tasks, the same task, and no task reaches the two
verified draws PCMD needs. The earlier pilot obtained its endpoints from a
separate 128-draw native evaluation of the saved checkpoints
(``ops/prepare_real_domains_native_endpoints_20260922.py``); that pass has not
been prepared for this campaign, and this builder does not stand in for it.
The endpoint is emitted as measured, with its eligibility count, so the
appendix can say what was and was not observed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "var/artifacts/real_domains_pilot_20260921/code_4arm_execution_20260924"
MONITOR = ROOT / "ops/monitor_code4arm_campaign_20260924.py"

ARMS = ("drgrpo", "redr", "maxrl", "remax")
PAIRS = (("redr", "drgrpo"), ("remax", "maxrl"))
#: arm -> (fresh objective, replay derivative live). The objective is not
#: recoverable from the result receipt; see the module docstring.
MATRIX = {"drgrpo": ("drgrpo", False), "redr": ("drgrpo", True),
          "maxrl": ("maxrl", False), "remax": ("maxrl", True)}
LABEL = {"drgrpo": "Dr.GRPO", "redr": "Re:Dr \\textit{(ours)}",
         "maxrl": "MaxRL", "remax": "Re:Max \\textit{(ours)}"}
UPDATES = 128
GROUP = 16
SCHEMA = "paper-code-4arm-campaign-v1"
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float, places: int = 3) -> str:
    text = f"{value:.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def signed(value: float, places: int = 3) -> str:
    return ("+" if value >= 0 else "-") + fmt(abs(value), places)


def paired(deltas: list[float]) -> dict:
    n = len(deltas)
    mean = st.fmean(deltas)
    half = T95[n] * st.stdev(deltas) / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": n,
            "positive": sum(1 for d in deltas if d > 0),
            "by_seed": deltas, "interval_type": "paired Student-t 95%"}


def read_cell(run_dir: Path) -> dict:
    arm, seed = run_dir.name.replace("training_", "").rsplit("_s", 1)
    result = json.loads((run_dir / "training" / "result.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if result["arm"] != arm:
        raise SystemExit(f"{run_dir.name}: receipt arm {result['arm']} differs from directory")
    if result["completed_updates"] != UPDATES or config["updates"] != UPDATES:
        raise SystemExit(f"{run_dir.name}: {result['completed_updates']} of {UPDATES} updates")
    if config["group_size"] != GROUP:
        raise SystemExit(f"{run_dir.name}: group size {config['group_size']}")

    rows = [json.loads(line) for line in
            (run_dir / "training" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != UPDATES:
        raise SystemExit(f"{run_dir.name}: {len(rows)} logged updates")
    # An update with no replay group logs no flag; that happens only before
    # the first verified success is banked, so require a group on all but the
    # earliest updates and read the flag wherever a group was scheduled.
    unflagged = [i for i, row in enumerate(rows) if row.get("replay_task_id") is None]
    if any(i >= 8 for i in unflagged):
        raise SystemExit(f"{run_dir.name}: no replay group at updates {unflagged}")
    live = {row.get("canonical_replay_compute_only") for row in rows
            if row.get("replay_task_id") is not None}
    if live != {0.0 if MATRIX[arm][1] else 1.0}:
        raise SystemExit(f"{run_dir.name}: replay flag {live} disagrees with arm {arm}")
    bank = json.loads((run_dir / "training" / "checkpoint-128" / "bank.json").read_text(
        encoding="utf-8"))["entries"]
    modes_per_task = {task: len(entry["modes"]) for task, entry in bank.items()}

    final = result["evaluation"]["final"]
    if len(final) != len(config["eval_ids"]) or {t["samples"] for t in final} != {config["eval_samples"]}:
        raise SystemExit(f"{run_dir.name}: evaluation shape differs from config")
    return {
        "arm": arm, "seed": int(seed), "run_dir": str(run_dir.relative_to(ROOT)),
        "fresh_objective": MATRIX[arm][0], "replay_live": MATRIX[arm][1],
        "eval_seed": config["eval_seed"], "eval_samples": config["eval_samples"],
        "train_tasks": list(config["train_ids"]), "eval_tasks": list(config["eval_ids"]),
        "fresh_rollouts": UPDATES * GROUP,
        "fresh_successes": sum(int(r["fresh_successes"]) for r in rows),
        "mixed_groups": sum(1 for r in rows if 0 < int(r["fresh_successes"]) < GROUP),
        "all_correct_groups": sum(1 for r in rows if int(r["fresh_successes"]) == GROUP),
        "all_wrong_groups": sum(1 for r in rows if int(r["fresh_successes"]) == 0),
        "bank_modes": int(rows[-1]["bank_modes"]),
        "bank_tasks": len(bank),
        "train_tasks_with_two_modes": sum(1 for m in modes_per_task.values() if m >= 2),
        "modes_per_train_task": modes_per_task,
        "dev_pass1": st.fmean(t["pass_at_1"] for t in final),
        "dev_pass8": st.fmean(t["pass_at_8"] for t in final),
        "dev_distinct8": st.fmean(t["distinct_valid_at_8"] for t in final),
        "dev_accepted": sum(int(t["accepted"]) for t in final),
        "dev_solved_tasks": sorted(t["task_id"] for t in final if t["accepted"]),
        "dev_pcmd_eligible": sum(1 for t in final if t["pcmd_eligible"]),
        "gpu_hours": result["allocated_gpu_hours_during_runner"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    parser.add_argument("--stamp", default=dt.date.today().isoformat().replace("-", ""))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cells: dict[tuple[str, int], dict] = {}
    for run_dir in sorted(EXE.glob("training_*")):
        if not run_dir.is_dir() or run_dir.name.endswith("_work") or run_dir.name == "training_requests":
            continue
        cell = read_cell(run_dir)
        cells[(cell["arm"], cell["seed"])] = cell
    seeds = sorted({s for _a, s in cells})
    missing = [(a, s) for a in ARMS for s in seeds if (a, s) not in cells]
    if missing:
        raise SystemExit(f"campaign is short {len(missing)} cells: {missing}")
    if len({c["eval_seed"] for c in cells.values()}) != 1:
        raise SystemExit("evaluation seeds differ across arms; the endpoint is not paired")
    if len({tuple(c["eval_tasks"]) for c in cells.values()}) != 1:
        raise SystemExit("evaluation task lists differ across arms")

    arms = {}
    for arm in ARMS:
        v = [cells[(arm, s)] for s in seeds]
        arms[arm] = {
            "label": LABEL[arm], "fresh_objective": MATRIX[arm][0], "replay_live": MATRIX[arm][1],
            "seeds": seeds,
            "fresh_success_rate": st.fmean(c["fresh_successes"] / c["fresh_rollouts"] for c in v),
            "fresh_successes": st.fmean(c["fresh_successes"] for c in v),
            "mixed_group_fraction": st.fmean(c["mixed_groups"] / UPDATES for c in v),
            "all_wrong_group_fraction": st.fmean(c["all_wrong_groups"] / UPDATES for c in v),
            "bank_modes": st.fmean(c["bank_modes"] for c in v),
            "train_tasks_with_two_modes": st.fmean(c["train_tasks_with_two_modes"] for c in v),
            "dev_pass1": st.fmean(c["dev_pass1"] for c in v),
            "dev_pass8": st.fmean(c["dev_pass8"] for c in v),
            "dev_distinct8": st.fmean(c["dev_distinct8"] for c in v),
            "dev_accepted": st.fmean(c["dev_accepted"] for c in v),
            "dev_solved_tasks": sorted({t for c in v for t in c["dev_solved_tasks"]}),
            "dev_pcmd_eligible": sum(c["dev_pcmd_eligible"] for c in v),
            "gpu_hours": st.fmean(c["gpu_hours"] for c in v),
        }

    contrasts = {}
    for replay, control in PAIRS:
        contrasts[f"{replay}_minus_{control}"] = {
            "fresh_success_rate": paired([
                (cells[(replay, s)]["fresh_successes"] - cells[(control, s)]["fresh_successes"])
                / cells[(replay, s)]["fresh_rollouts"] for s in seeds]),
            "bank_modes": paired([cells[(replay, s)]["bank_modes"] - cells[(control, s)]["bank_modes"]
                                  for s in seeds]),
            "dev_accepted": paired([cells[(replay, s)]["dev_accepted"] - cells[(control, s)]["dev_accepted"]
                                    for s in seeds]),
            "dev_pass8": paired([cells[(replay, s)]["dev_pass8"] - cells[(control, s)]["dev_pass8"]
                                 for s in seeds]),
        }
    pooled_fresh = paired([
        (cells[(r, s)]["fresh_successes"] - cells[(c, s)]["fresh_successes"]) / (UPDATES * GROUP)
        for r, c in PAIRS for s in seeds])

    n_dev = len(next(iter(cells.values()))["eval_tasks"])
    endpoint = {
        "draws_per_task": next(iter(cells.values()))["eval_samples"],
        "development_tasks": n_dev,
        "solved_task_sets_identical": len({tuple(c["dev_solved_tasks"]) for c in cells.values()}) == 1,
        "solved_tasks": sorted({t for c in cells.values() for t in c["dev_solved_tasks"]}),
        "pass8_identical_across_cells": len({round(c["dev_pass8"], 9) for c in cells.values()}) == 1,
        "pcmd_eligible_cells": sum(1 for c in cells.values() if c["dev_pcmd_eligible"]),
        "separates_arms": False,
        "deep_endpoint": "not prepared for this campaign; the earlier pilot's 128-draw "
                         "native evaluation of saved checkpoints is the pass that would "
                         "measure pass@32, distinct modes and PCMD eligibility",
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "campaign": EXE.name,
        "design": {
            "model": "Qwen2.5-Coder-7B-Instruct, rank-16 LoRA", "updates": UPDATES,
            "group_size": GROUP, "train_tasks": next(iter(cells.values()))["train_tasks"],
            "eval_tasks": next(iter(cells.values()))["eval_tasks"], "seeds": seeds,
            "arms": {a: {"fresh_objective": MATRIX[a][0], "replay_live": MATRIX[a][1]} for a in ARMS},
            "replay_factor_verified_from": "canonical_replay_compute_only at every logged update",
            "objective_receipt_caveat": "result.json's objective string is the same for every arm; "
                                        "arms are keyed on the run's arm field and directory name",
        },
        "cells": len(cells),
        "sources": {
            "campaign_root": str(EXE.relative_to(ROOT)),
            "monitor": {"path": str(MONITOR.relative_to(ROOT)), "sha256": sha256(MONITOR)},
            "result_receipts": {f"{a}/s{s}": sha256(ROOT / c["run_dir"] / "training" / "result.json")
                                for (a, s), c in sorted(cells.items())},
        },
        "arms": arms,
        "paired_contrasts": contrasts,
        "pooled_fresh_success_contrast": pooled_fresh,
        "development_endpoint": endpoint,
        "registered_predictions": None,
        "cells_measured": {f"{a}/s{s}": c for (a, s), c in sorted(cells.items())},
    }
    (out / f"code_4arm_campaign_{args.stamp}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    rd, rm = contrasts["redr_minus_drgrpo"], contrasts["remax_minus_maxrl"]
    macros = [
        "% Generated by ops/build_paper_code_4arm_campaign.py; do not hand edit.",
        rf"\newcommand{{\CCAcells}}{{{len(cells)}}}",
        rf"\newcommand{{\CCAseeds}}{{{len(seeds)}}}",
        rf"\newcommand{{\CCAupdates}}{{{UPDATES}}}",
        rf"\newcommand{{\CCAfresh}}{{{UPDATES * GROUP:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\CCAtraintasks}}{{{len(next(iter(cells.values()))['train_tasks'])}}}",
        rf"\newcommand{{\CCAdevtasks}}{{{n_dev}}}",
        rf"\newcommand{{\CCAdevdraws}}{{{endpoint['draws_per_task']}}}",
    ]
    for arm in ARMS:
        tag = arm
        a = arms[arm]
        macros += [
            rf"\newcommand{{\CCA{tag}fresh}}{{{fmt(a['fresh_success_rate'])}}}",
            rf"\newcommand{{\CCA{tag}mixed}}{{{fmt(a['mixed_group_fraction'])}}}",
            rf"\newcommand{{\CCA{tag}bank}}{{{a['bank_modes']:.1f}}}",
            rf"\newcommand{{\CCA{tag}devpass}}{{{fmt(a['dev_pass8'])}}}",
            rf"\newcommand{{\CCA{tag}devaccepted}}{{{a['dev_accepted']:.1f}}}",
        ]
    macros += [
        rf"\newcommand{{\CCAredrfreshgain}}{{{signed(rd['fresh_success_rate']['mean'])}}}",
        rf"\newcommand{{\CCAredrfreshlo}}{{{signed(rd['fresh_success_rate']['ci95'][0])}}}",
        rf"\newcommand{{\CCAredrfreshhi}}{{{signed(rd['fresh_success_rate']['ci95'][1])}}}",
        rf"\newcommand{{\CCAredrfreshpos}}{{{rd['fresh_success_rate']['positive']}}}",
        rf"\newcommand{{\CCAremaxfreshgain}}{{{signed(rm['fresh_success_rate']['mean'])}}}",
        rf"\newcommand{{\CCAremaxfreshlo}}{{{signed(rm['fresh_success_rate']['ci95'][0])}}}",
        rf"\newcommand{{\CCAremaxfreshhi}}{{{signed(rm['fresh_success_rate']['ci95'][1])}}}",
        rf"\newcommand{{\CCAremaxfreshpos}}{{{rm['fresh_success_rate']['positive']}}}",
        rf"\newcommand{{\CCApooledfreshgain}}{{{signed(pooled_fresh['mean'])}}}",
        rf"\newcommand{{\CCApooledfreshlo}}{{{signed(pooled_fresh['ci95'][0])}}}",
        rf"\newcommand{{\CCApooledfreshhi}}{{{signed(pooled_fresh['ci95'][1])}}}",
        rf"\newcommand{{\CCApooledfreshpos}}{{{pooled_fresh['positive']}}}",
        rf"\newcommand{{\CCApooledpairs}}{{{pooled_fresh['n']}}}",
        rf"\newcommand{{\CCAdevsolved}}{{{len(endpoint['solved_tasks'])}}}",
        rf"\newcommand{{\CCAdevpass}}{{{fmt(arms['drgrpo']['dev_pass8'])}}}",
        rf"\newcommand{{\CCAdeveligible}}{{{endpoint['pcmd_eligible_cells']}}}",
    ]
    (out / f"code_4arm_campaign_{args.stamp}_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8")

    body = ["% Generated by ops/build_paper_code_4arm_campaign.py; do not hand edit."]
    for arm in ARMS:
        a = arms[arm]
        body.append(
            f"  {LABEL[arm]} & {fmt(a['fresh_success_rate'])} & {fmt(a['mixed_group_fraction'])}"
            f" & {a['bank_modes']:.1f} & {a['train_tasks_with_two_modes']:.1f}"
            f" & {fmt(a['dev_pass8'])} & {a['dev_accepted']:.1f} & {a['dev_pcmd_eligible']} \\\\"
        )
        if arm in ("redr", "remax"):
            body.append(r"  \addlinespace[2pt]")
    body[-1:] = [r"  \bottomrule"]
    (out / f"code_4arm_campaign_{args.stamp}_table_body.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8")

    print(f"wrote code_4arm_campaign_{args.stamp}.{{json,_macros.tex,_table_body.tex}} -> {out.relative_to(ROOT)}")
    print(f"  cells={len(cells)} seeds={seeds} updates={UPDATES} group={GROUP} dev_tasks={n_dev} draws={endpoint['draws_per_task']}")
    for arm in ARMS:
        a = arms[arm]
        print(f"  {arm:7s} fresh {fmt(a['fresh_success_rate'])}  mixed {fmt(a['mixed_group_fraction'])}"
              f"  bank {a['bank_modes']:.1f}  dev pass@8 {fmt(a['dev_pass8'])} accepted {a['dev_accepted']:.1f}"
              f"  pcmd-eligible {a['dev_pcmd_eligible']}")
    for name, c in contrasts.items():
        f = c["fresh_success_rate"]
        print(f"  {name}: fresh success {signed(f['mean'])} [{signed(f['ci95'][0])},{signed(f['ci95'][1])}]"
              f" positive {f['positive']}/{f['n']}; dev accepted {signed(c['dev_accepted']['mean'], 1)}")
    print(f"  pooled fresh success {signed(pooled_fresh['mean'])} [{signed(pooled_fresh['ci95'][0])},"
          f"{signed(pooled_fresh['ci95'][1])}] positive {pooled_fresh['positive']}/{pooled_fresh['n']}")
    print(f"  development endpoint: solved {endpoint['solved_tasks']} in every cell = {endpoint['solved_task_sets_identical']};"
          f" pass@8 identical = {endpoint['pass8_identical_across_cells']}; PCMD-eligible cells = {endpoint['pcmd_eligible_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
