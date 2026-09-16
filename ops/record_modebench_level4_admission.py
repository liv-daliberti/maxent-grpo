#!/usr/bin/env python3
"""Admit ModeBench Level 4 with its MathIR limitation on the record.

Level 4 was stopped on 2026-09-15 at four of five domains confirmed, and the
release has said ``not_admitted`` since. What changed is not the measurement but
what is known about it: the 2026-09-16 ratio bound shows the MathIR target sits
outside the achievable set at 7B rather than merely unreached. Matching needs a
pass@8/pass@1 ratio of 5.50; the best any construction of this family attains is
4.16, against 8.0 for a homogeneous population, and a ratio of averages cannot
exceed its best component.

On 2026-09-16 the level was admitted on that basis. This writes the admission
beside the data so the release and the manuscript agree.

It does not restate any domain as difficulty-matched. MathIR is not, that stays
measured and visible, and the admission records the exception rather than
dissolving it. Prior records are preserved: the closure decision is kept under
``superseded`` rather than dropped.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "var/data/modebench_scale_release_v2/level4"
STATUS = RELEASE / "release_status.json"
ADMISSION = RELEASE / "admission.json"
README = RELEASE / "README.md"
RATIO_BOUND = ROOT / "artifacts/modebench_scale_level4_mathir_ratio_bound_20260916.json"
DECIDED = "2026-09-16"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    status = json.loads(STATUS.read_text())
    bound = json.loads(RATIO_BOUND.read_text())

    exception = {
        "domain": "mathir",
        "difficulty_matched": False,
        "why_the_target_is_unreachable": bound["why_the_target_is_unreachable"],
        "required_ratio_pass8_over_pass1": bound["required_ratio"],
        "best_construction_time_ratio": bound["best_construction_time_ratio"],
        "homogeneous_ratio": bound["homogeneous_ratio"],
        "bound_status": bound["status"],
        "evidence": {
            "path": str(RATIO_BOUND.relative_to(ROOT)),
            "sha256": digest(RATIO_BOUND),
        },
        "what_this_does_not_claim":
            "MathIR at Level 4 is not difficulty-matched to its Level-1 "
            "reference. Admission records that the target is unreachable at 7B, "
            "not that it was met.",
    }
    admission = {
        "schema": "modebench_scale_level4_admission_with_exception_v1",
        "level": "level4",
        "model_label": status["model_label"],
        "test_split": status["test_split"],
        "tolerances": status["tolerances"],
        "admitted": True,
        "admitted_on": DECIDED,
        "difficulty_matched_domains": status["confirmed_domains"],
        "difficulty_matched_all_domains": False,
        "exception": exception,
        "basis":
            "Four of five domains cleared their registered held-out gates. The "
            "fifth is unreachable by construction rather than unmet by effort, "
            "and the datasets are frozen, verified and disjoint from every other "
            "level, so the level is admitted with the exception recorded.",
        "supersedes": {
            "closure_decision": status["closure_decision"],
            "closure_record": status["closure_record"],
            "prior_status": status["status"],
        },
        "datasets": "384 train / 128 development / 128 test rows per domain; "
                    "test split named eval.",
    }
    ADMISSION.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")

    status["admitted"] = True
    status["admitted_on"] = DECIDED
    status["status"] = "admitted_with_documented_exception"
    status["admission_record"] = str(ADMISSION.relative_to(ROOT))
    # difficulty_matched stays False on purpose: it is the measured property,
    # and admission did not change it.
    status["usable"] = (
        "All five datasets are frozen at 384/128/128, verified, and disjoint "
        "from every other level. Level 4 is admitted with a recorded exception: "
        "four domains are difficulty-matched, MathIR is not, and its target is "
        "outside the achievable set at 7B. Any MathIR number must carry that "
        "non-match.")
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    text = README.read_text()
    text = text.replace(
        "# ModeBench level4 (7b) — partial release",
        "# ModeBench level4 (7b) — admitted with a recorded exception")
    text = text.replace(
        "Four of five domains passed their registered held-out gates. MathIR "
        "did not, so **this level is not admitted** and carries no "
        "`admission.json`.",
        "Four of five domains passed their registered held-out gates. MathIR "
        "did not, and its target is outside the achievable set at 7B rather "
        "than merely unreached, so **this level is admitted with that exception "
        "recorded** in `admission.json`. MathIR remains not difficulty-matched.")
    README.write_text(text)

    print(json.dumps({"event": "admitted", "level": "level4",
                      "admission": str(ADMISSION.relative_to(ROOT)),
                      "status": status["status"],
                      "exception": "mathir"}))


if __name__ == "__main__":
    main()
