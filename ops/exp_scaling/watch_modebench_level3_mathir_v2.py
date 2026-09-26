#!/usr/bin/env python3
"""Single-owner, MathIR-only continuation for the already registered v2 receipts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "var/artifacts/modebench_level3_v2"
SEAL = CAMPAIGN / "regular_queue_recovery/implementation_seal.json"
SEAL_SHA = "65b3990674e8560d084f328a4e189709bfc99cd012e8e6620efd81879c8d2cab"
RECEIPTS = [ROOT / "var/results/modebench_level3_v2" / name for name in (
    "calibration_05b_mathir.json", *(f"calibration_3b_mathir_d{i}.json" for i in range(4)))]
RECIPE = CAMPAIGN / "recipes/mathir.json"
REPORT = CAMPAIGN / "mathir_development_fit_report.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(source_sha):
    if digest(__file__) != source_sha or digest(SEAL) != SEAL_SHA:
        raise ValueError("MathIR watcher or registered recovery seal changed")
    seal = json.loads(SEAL.read_text())
    for path, expected in seal["files_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen scientific/execution input changed: {path}")
    for directory, expected in seal["directory_files"].items():
        actual = sorted(str(p.resolve()) for p in Path(directory).rglob("*") if p.is_file())
        if actual != sorted(expected):
            raise ValueError(f"Frozen directory inventory changed: {directory}")


def fit_once(source_sha):
    verify(source_sha)
    if RECIPE.exists() or REPORT.exists():
        raise FileExistsError("MathIR fit already published; never fit a second recipe")
    receipt_pins = {str(path): digest(path) for path in RECEIPTS}
    for directory in ("ops", "ops/exp_scaling", "src"):
        sys.path.insert(0, str(ROOT / directory))
    from fit_modebench_level3_independent import fit_recipe
    from evaluate_modebench_level3 import atomic_new
    recipe = fit_recipe(RECEIPTS[0], RECEIPTS[1:], "mathir", RECIPE)
    verify(source_sha)
    if receipt_pins != {str(path): digest(path) for path in RECEIPTS}:
        raise ValueError("MathIR receipts changed during fitting")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "domain": "mathir",
              "watcher_sha256": source_sha, "recovery_seal_sha256": SEAL_SHA,
              "receipt_sha256": receipt_pins, "recipe_path": str(RECIPE),
              "recipe_sha256": digest(RECIPE), "confirmation_outcomes_used": False,
              **{key: recipe[key] for key in ("decision", "weights", "development")}}
    atomic_new(REPORT, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    if not args.watch:
        print(json.dumps({"status": "read_only", "domain": "mathir",
                          "receipts_complete": all(p.is_file() for p in RECEIPTS),
                          "recipe_exists": RECIPE.exists()}, sort_keys=True), flush=True)
        return
    verify(args.source_sha256)
    with (CAMPAIGN / ".mathir_fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if RECIPE.exists() or REPORT.exists():
            raise FileExistsError("MathIR fit evidence already exists")
        print("Waiting for all five registered MathIR receipts; this owner fits MathIR only.", flush=True)
        while not all(path.is_file() for path in RECEIPTS):
            time.sleep(30)
        print(json.dumps(fit_once(args.source_sha256), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
