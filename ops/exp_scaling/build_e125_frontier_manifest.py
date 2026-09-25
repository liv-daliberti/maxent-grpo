#!/usr/bin/env python3
"""Freeze the E125 decoding-frontier source inventory.

E125 re-measures the *current* eight-pass terminal checkpoints under changed
decoding settings, so that the decoding appendix stops resting on the earlier
twelve-pass E72 cohort whose replay arm predates the Re:Dr objective contract.
The grid itself is unchanged: this script only says which checkpoints it runs
on and under which inherited evaluation configuration.

Where each field comes from, and why it is not retyped here:

* the checkpoint, from each run's ``TRAINING_COMPLETE.json`` terminal marker;
* the evaluation configuration, from ``e72_frontier_source_runs.json`` --- not
  because these are E72 checkpoints, but because that frozen manifest is the
  artifact ``launch_e78_verified_replay_only_05b.py`` itself read when it built
  these runs' environments, so it is the configuration they trained and were
  published under. The E78 learner logs have since been reaped, and re-deriving
  the configuration from anywhere else would be a second source of truth;
* the published terminal endpoint, recomputed from each run's retained per-draw
  records by the same reader ``build_paper_core_terminal_endpoints.py`` uses,
  so the reproduction gate compares the sweep against the number the paper
  prints rather than against a parallel recomputation;
* the training node, from the E78 ledger, cross-checked against Slurm
  accounting. Every cell is pinned to the GPU model that trained it, because
  GPU model changes floating-point reduction order and therefore the sampled
  tokens.

All fifty control/replay pairs of this cohort trained on two hosts --- node302
(a100) and node105 (a5000) --- and both arms of every pair trained on the same
one. The sweep is therefore fully pinned and every paired contrast is
hardware-matched, which is what lets E125 drop the unpinned compromise the
preregistration had registered for the four-arm population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_paper_core_terminal_endpoints as endpoints  # noqa: E402

MODEL_TAG = "qwen25_0p5b_instruct"
MODEL = "Qwen2.5-0.5B-Instruct"

LEDGER = "var/artifacts/e78_verified_replay_only_05b_jobs.json"
EVAL_CONFIG_SOURCE = "var/artifacts/e72_frontier_source_runs.json"
OUTPUT = "var/artifacts/e125_frontier_source_runs.json"

DOMAINS: tuple[str, ...] = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
ARMS: tuple[str, ...] = ("control", "replay")
SEEDS: tuple[int, ...] = (43, 44, 45, 46, 47)

#: The arm of the E72 manifest whose per-(domain, seed) template E78 read. Both
#: E78 arms were built from this same template; they differ by objective flags,
#: never by an evaluation field.
EVAL_TEMPLATE_ARM = "xgrpo"

#: Evaluation fields carried into the sweep. A cell that changes any of these
#: is measuring a different quantity, so the set is closed and checked.
INHERITED_EVAL_KEYS: tuple[str, ...] = (
    "canonical_action_task",
    "canonical_graph_action_count",
    "canonical_graph_actions",
    "canonical_graph_fixed_shape_sampling",
    "canonical_graph_learner_sampling",
    "collocate",
    "eval_batch_size",
    "eval_data",
    "eval_generate_max_length",
    "eval_input_key",
    "eval_mode_coverage_draws",
    "eval_mode_coverage_k",
    "eval_mode_coverage_seed",
    "eval_mode_coverage_temperature",
    "eval_output_key",
    "eval_temperature",
    "generate_max_length",
    "max_model_len",
    "max_train",
    "num_samples",
    "pretrain",
    "prompt_data",
    "prompt_max_length",
    "prompt_template",
    "rollout_batch_size",
    "test_split",
    "train_batch_size_per_device",
    "verifier_version",
    "vllm_gpu_ratio",
    "zero_stage",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sacct_nodes(job_ids: list[str]) -> dict[str, str]:
    """Recover each job's node from Slurm accounting, one batched query.

    The ledger records a source node too. Reading both and requiring agreement
    catches a ledger written from the submission request rather than from where
    the job actually landed, which is the failure that would silently unpin the
    sweep.
    """

    if not job_ids:
        return {}
    result = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "-X", "-n", "-P",
         "--format=JobID,NodeList,State"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"sacct failed: {result.stderr.strip()}")
    nodes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        job_id, node_list, state = parts
        if state != "COMPLETED":
            raise SystemExit(f"job {job_id} accounts as {state}, not COMPLETED")
        nodes[job_id] = node_list
    return nodes


def published_terminal(run_dir: Path, step: int) -> dict[str, Any] | None:
    """The endpoint the paper prints for this run, plus its across-draw spread.

    ``sampled_endpoint`` returns the four-draw means the paper reports. The
    reproduction gate also needs the spread those means were taken over and the
    greedy trace, so both are read from the same records in the same pass.
    """

    means = endpoints.sampled_endpoint(run_dir, step=step)
    if means is None:
        return None

    per_draw: dict[str, list[float]] = {name: [] for name in endpoints.ENDPOINT_FIELDS}
    greedy: float | None = None
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("step") != step:
                    continue
                metrics = row.get("metrics")
                if not isinstance(metrics, dict):
                    continue
                kind = row.get("evaluation_kind")
                if kind == "fixed_seed_sampled_k_neutral":
                    for name, field in endpoints.ENDPOINT_FIELDS.items():
                        per_draw[name].append(float(metrics[field]))
                elif kind == "deterministic_greedy_trace_neutral":
                    value = float(metrics[endpoints.ENDPOINT_FIELDS["pass8"]])
                    if greedy is not None and abs(greedy - value) > 0:
                        raise SystemExit(
                            f"{run_dir}: two disagreeing greedy traces at step {step}"
                        )
                    greedy = value

    record: dict[str, Any] = {"step": step, "greedy": greedy}
    for name, values in per_draw.items():
        record[name] = means[name]
        # Standard error of the reported mean over the draws it averaged.
        record[f"{name}_draw_se"] = (
            statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        )
    return record


def eval_templates(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = load(root / EVAL_CONFIG_SOURCE)
    templates: dict[tuple[str, int], dict[str, Any]] = {}
    for run in payload["runs"]:
        if str(run.get("arm")) != EVAL_TEMPLATE_ARM:
            continue
        templates[(str(run["domain"]), int(run["seed"]))] = run["inherited_eval_config"]
    expected = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    if set(templates) != expected:
        raise SystemExit(
            f"{EVAL_CONFIG_SOURCE} does not hold exactly the 25 "
            f"{EVAL_TEMPLATE_ARM} domain/seed templates"
        )
    return templates


def build(*, root: Path) -> dict[str, Any]:
    ledger_path = root / LEDGER
    ledger = load(ledger_path)
    if ledger.get("released") is not True:
        raise SystemExit(f"{LEDGER} was not durably released")
    if int(ledger["passes"]) != 8:
        raise SystemExit(f"{LEDGER} is not the eight-pass cohort")
    target_step = int(ledger["target_steps"])

    templates = eval_templates(root)
    index = {
        (str(run["domain"]), str(run["arm"]), int(run["seed"])): run
        for run in ledger["runs"]
    }
    nodes = sacct_nodes([str(run["job_id"]) for run in ledger["runs"]])

    runs: list[dict[str, Any]] = []
    problems: list[str] = []
    for domain in DOMAINS:
        for arm in ARMS:
            for seed in SEEDS:
                cell = f"{domain}/{arm}/s{seed}"
                source = index.get((domain, arm, seed))
                if source is None:
                    problems.append(f"{cell}: absent from {LEDGER}")
                    continue

                run_dir = Path(source["run_dir"])
                marker = run_dir / "TRAINING_COMPLETE.json"
                if not marker.is_file():
                    problems.append(f"{cell}: no TRAINING_COMPLETE.json")
                    continue
                complete = load(marker)
                export = Path(complete["terminal_export"])
                terminal_step = int(complete["terminal_step"])
                if terminal_step != target_step + 1:
                    problems.append(
                        f"{cell}: terminal step {terminal_step} is not the "
                        f"export of target step {target_step}"
                    )
                    continue

                ledger_node = str(source.get("source_node") or "")
                accounted = nodes.get(str(source["job_id"]), "")
                if not ledger_node:
                    problems.append(f"{cell}: ledger records no source node")
                    continue
                if accounted and accounted != ledger_node:
                    problems.append(
                        f"{cell}: ledger node {ledger_node} but Slurm accounts "
                        f"{accounted}; the sweep would be pinned to the wrong host"
                    )
                    continue

                # Weights are retired to the model archive; the receipt is what
                # makes the restore verifiable, so a cell without one cannot be
                # measured on the checkpoint the paper reports.
                receipt = run_dir / "MODEL_ARCHIVE.json"
                weights = export / "model.safetensors"
                if weights.is_file():
                    archive: dict[str, Any] | None = None
                    weight_bytes = weights.stat().st_size
                    weight_sha: str | None = None
                elif receipt.is_file():
                    retired = load(receipt)
                    if Path(retired["original_terminal_export"]) != export:
                        problems.append(f"{cell}: receipt names another export")
                        continue
                    removed = {
                        entry["relative_path"]: entry
                        for entry in retired.get("removed_files", [])
                    }
                    if "model.safetensors" not in removed:
                        problems.append(f"{cell}: receipt retires no weight file")
                        continue
                    archive = {
                        "receipt": str(receipt),
                        "repo_id": retired["repo_id"],
                        "repo_prefix": retired["repo_prefix"],
                        "status": retired["status"],
                    }
                    weight_bytes = int(removed["model.safetensors"]["bytes"])
                    weight_sha = removed["model.safetensors"]["sha256"]
                else:
                    problems.append(f"{cell}: weights absent and no archive receipt")
                    continue

                terminal = published_terminal(run_dir, target_step)
                if terminal is None:
                    problems.append(
                        f"{cell}: no admissible sampled endpoint at step {target_step}"
                    )
                    continue
                if terminal["greedy"] is None:
                    problems.append(f"{cell}: no greedy trace at step {target_step}")
                    continue

                runs.append(
                    {
                        "arm": arm,
                        "domain": domain,
                        "seed": seed,
                        "run_stamp": source["run_stamp"],
                        "run_dir": str(run_dir),
                        "job_id": source["job_id"],
                        "source_node": ledger_node,
                        "source_node_accounted": accounted or None,
                        "export": {
                            "path": str(export),
                            "step": terminal_step,
                            "weight_bytes": weight_bytes,
                            "weight_files": ["model.safetensors"],
                            "weight_sha256": weight_sha,
                            "archive": archive,
                        },
                        "inherited_eval_config": {
                            key: templates[(domain, seed)][key]
                            for key in INHERITED_EVAL_KEYS
                        },
                        "published_terminal": terminal,
                    }
                )

    expected_runs = len(DOMAINS) * len(ARMS) * len(SEEDS)
    return {
        "schema": "e125-frontier-source-runs-v1",
        "model": MODEL,
        "model_tag": MODEL_TAG,
        "cohort": {
            "experiment": "E78",
            "passes": int(ledger["passes"]),
            "target_steps": target_step,
            "terminal_export_step": target_step + 1,
            "replay_objective": ledger.get("objective"),
            "replay_weight": ledger.get("replay_weight"),
        },
        "arms": list(ARMS),
        "domains": list(DOMAINS),
        "seeds": list(SEEDS),
        "eval_config_source": str(root / EVAL_CONFIG_SOURCE),
        "eval_config_source_sha256": digest(root / EVAL_CONFIG_SOURCE),
        "eval_config_template_arm": EVAL_TEMPLATE_ARM,
        "ledger": str(ledger_path),
        "ledger_sha256": digest(ledger_path),
        "expected_runs": expected_runs,
        "resolved_runs": len(runs),
        "problems": problems,
        "runs": runs,
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / OUTPUT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write the manifest even when some cells could not be resolved",
    )
    args = parser.parse_args()

    payload = build(root=root)
    complete = (
        payload["resolved_runs"] == payload["expected_runs"] and not payload["problems"]
    )
    if not complete and not args.allow_incomplete:
        for problem in payload["problems"][:20]:
            print(f"[e125-manifest] {problem}", file=sys.stderr)
        raise SystemExit(
            f"[e125-manifest] resolved {payload['resolved_runs']}/"
            f"{payload['expected_runs']} runs; refusing to write a partial manifest"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, args.output)

    pinned: dict[str, int] = {}
    for run in payload["runs"]:
        pinned[run["source_node"]] = pinned.get(run["source_node"], 0) + 1
    print(
        f"[e125-manifest] wrote {args.output} "
        f"({payload['resolved_runs']}/{payload['expected_runs']} runs, "
        f"pinned {pinned})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
