#!/usr/bin/env python3
"""Install the frozen E117-A4 causal-audit estimand closure."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replace_e117r1_a2_audit_job as replacement  # noqa: E402


replacement.PROTOCOL = (
    "paper/preregistration/e117a4_causal_audit_estimand_closure_20260825.md"
)
replacement.HISTORY = "var/artifacts/e117a4_audit_job_replacement.json"
replacement.AMENDMENT_NAME = "E117-A4"
replacement.REPLACEMENT_SCHEMA = "e117a4_audit_job_replacement_v1"
replacement.AUDIT_JOB_SCHEMA = "e117r1_same_plumbing_component_preflight_audit_job_v5"
replacement.LOG_LABEL = "e117a4"


if __name__ == "__main__":
    raise SystemExit(replacement.main())
