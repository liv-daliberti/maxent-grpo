#!/usr/bin/env python3
"""Build the exact terminal-progress record for official-verl DAPO.

The upstream validation surface is a single sampled response per evaluation
prompt.  It is retained as an implementation/training diagnostic and is never
relabelled as the paper's standardized pass@8 or breadth endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import status_e78 as shared  # noqa: E402


LEDGER = ROOT / "var/artifacts/e113r4_official_verl_dapo_jobs.json"
OUTPUT = ROOT / "paper/results/e113r4_dapo_progress.json"
TARGET_STEPS = 24
FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}
RUNNING_STATES = {"RUNNING", "CONFIGURING", "COMPLETING"}
FINAL_ACC = re.compile(
    r"'val-core/modebench/(?P<domain>[^/]+)/acc/mean@1': "
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)"
)
STEP_LINE = re.compile(r"step:(?P<step>[0-9]+) - .*")
GEN_BATCHES = re.compile(r"train/num_gen_batches:(?P<value>[0-9]+(?:\.[0-9]+)?)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _terminal_record(row: dict[str, Any]) -> tuple[dict[str, Any], set[Path]] | None:
    run_dir = Path(str(row["run_dir"]))
    receipt_path = run_dir / "TRAINING_COMPLETE.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("total_training_steps") != TARGET_STEPS:
        raise RuntimeError(
            "DAPO completion receipt has the wrong terminal step: "
            f"{receipt_path}"
        )
    expected = {
        "schema": "e113r4_official_verl_dapo_training_complete_v1",
        "family": str(row["family"]),
        "domain": str(row["domain"]),
        "seed": int(row["seed"]),
        "slurm_job_id": str(row["job_id"]),
    }
    drift = {
        key: (value, receipt.get(key))
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if drift:
        raise RuntimeError(f"DAPO completion receipt drifted: {drift}")

    checkpoint = run_dir / f"checkpoints/global_step_{TARGET_STEPS}"
    actor = checkpoint / "actor"
    latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    if not actor.is_dir():
        raise RuntimeError(f"terminal DAPO checkpoint lacks actor state: {actor}")
    if not latest.is_file() or latest.read_text(encoding="utf-8").strip() != str(
        TARGET_STEPS
    ):
        raise RuntimeError(f"terminal DAPO checkpoint pointer drifted: {latest}")

    log_path = Path(str(row["log_path"]))
    if not log_path.is_file():
        raise RuntimeError(f"terminal DAPO log is missing: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    acc_matches = [
        match
        for match in FINAL_ACC.finditer(log_text)
        if match.group("domain") == str(row["domain"])
    ]
    if len(acc_matches) < 2:
        raise RuntimeError(f"terminal DAPO log lacks initial/final acc@1: {log_path}")

    final_step_line = next(
        (
            line
            for line in reversed(log_text.splitlines())
            if f"step:{TARGET_STEPS} - " in line
        ),
        None,
    )
    if final_step_line is None or STEP_LINE.search(final_step_line) is None:
        raise RuntimeError(f"terminal DAPO log lacks step {TARGET_STEPS}: {log_path}")
    gen_match = GEN_BATCHES.search(final_step_line)
    if gen_match is None:
        raise RuntimeError(f"terminal DAPO log lacks generation-batch count: {log_path}")

    return (
        {
            "family": str(row["family"]),
            "domain": str(row["domain"]),
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "accepted_training_steps": TARGET_STEPS,
            "initial_upstream_validation_acc_at_1": float(
                acc_matches[0].group("value")
            ),
            "final_upstream_validation_acc_at_1": float(
                acc_matches[-1].group("value")
            ),
            "final_num_generation_batches": float(gen_match.group("value")),
            "completion_receipt": _relative(receipt_path),
            "terminal_checkpoint": _relative(checkpoint),
            "log": _relative(log_path),
            "standardized_pass_at_8_available": False,
            "standardized_breadth_at_8_available": False,
            "evidence_class": "terminal_upstream_acc_at_1_diagnostic",
        },
        {receipt_path, latest, log_path},
    )


def build() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if ledger.get("schema") != "e113r4_official_verl_dapo_jobs_v1":
        raise RuntimeError("unexpected E113-R4 ledger schema")
    runs = list(ledger.get("runs", []))
    if len(runs) != 50:
        raise RuntimeError(f"expected 50 DAPO science cells, found {len(runs)}")

    scheduler_states = shared.scheduler_states(
        [int(row["job_id"]) for row in runs]
    )
    terminal: list[dict[str, Any]] = []
    sources: set[Path] = {LEDGER}
    for row in runs:
        result = _terminal_record(row)
        if result is None:
            continue
        record, record_sources = result
        record["scheduler_state"] = scheduler_states.get(
            int(row["job_id"]), "UNKNOWN"
        )
        terminal.append(record)
        sources.update(record_sources)
    terminal.sort(key=lambda row: (row["family"], row["domain"], row["seed"]))
    terminal_job_ids = {int(row["job_id"]) for row in terminal}
    nonterminal_states = {
        int(row["job_id"]): scheduler_states.get(int(row["job_id"]), "UNKNOWN")
        for row in runs
        if int(row["job_id"]) not in terminal_job_ids
    }
    completed_without_receipt = sorted(
        job_id
        for job_id, state in nonterminal_states.items()
        if state == "COMPLETED"
    )
    if completed_without_receipt:
        raise RuntimeError(
            "DAPO jobs completed without a validated terminal receipt: "
            f"{completed_without_receipt}"
        )

    counts: dict[str, dict[str, int]] = {}
    for row in terminal:
        counts.setdefault(row["family"], {})
        counts[row["family"]][row["domain"]] = (
            counts[row["family"]].get(row["domain"], 0) + 1
        )

    return {
        "schema": "paper-e113r4-official-dapo-progress-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registered_science_cells": 50,
        "terminal_science_cells": len(terminal),
        "nonterminal_science_cells": 50 - len(terminal),
        "running_science_cells": sum(
            state in RUNNING_STATES for state in nonterminal_states.values()
        ),
        "pending_science_cells": sum(
            state == "PENDING" for state in nonterminal_states.values()
        ),
        "failed_science_cells": sum(
            state in FAILURE_STATES for state in nonterminal_states.values()
        ),
        "other_nonterminal_science_cells": sum(
            state not in RUNNING_STATES | FAILURE_STATES | {"PENDING"}
            for state in nonterminal_states.values()
        ),
        "scheduler_state_by_job_id": {
            str(job_id): state
            for job_id, state in sorted(
                {
                    int(row["job_id"]): (
                        "COMPLETED"
                        if int(row["job_id"]) in terminal_job_ids
                        else nonterminal_states[int(row["job_id"])]
                    )
                    for row in runs
                }.items()
            )
        },
        "terminal_counts_by_family_domain": counts,
        "records": terminal,
        "interpretation": {
            "upstream_validation": (
                "one sampled response per 128-prompt evaluation split at "
                "temperature 1 and top-p 0.7; descriptive diagnostic only"
            ),
            "standardized_endpoint": (
                "pass@8 and executed-mode breadth require a separate matched "
                "paper evaluation and are not inferred from upstream acc@1"
            ),
            "incomplete_prefix": (
                "report exact cells and values only; no mean, interval, pooled, "
                "cross-domain, or efficacy conclusion"
            ),
            "scheduler_snapshot": (
                "running, pending, failed, and other-nonterminal counts are a "
                "generated-at scheduler snapshot; terminal status additionally "
                "requires the validated 24-step completion receipt"
            ),
        },
        "input_sha256": {
            _relative(path): {
                "byte_length": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(sources)
        },
    }


def main() -> int:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(
        f"DAPO progress: {payload['terminal_science_cells']}/50 terminal; "
        f"wrote {OUTPUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
