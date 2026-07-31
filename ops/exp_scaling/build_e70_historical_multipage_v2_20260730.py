#!/usr/bin/env python3
"""Assemble the corrected three-page current + historical figure report."""

from __future__ import annotations

import json
import os
import re

import build_e70_historical_multipage_20260730 as v1


def main() -> None:
    for renderer in (
        v1.PRIMARY_RENDERER,
        v1.DIAGNOSTIC_RENDERER,
        v1.LEGACY_RENDERER,
    ):
        v1.run([v1.sys.executable, str(renderer)])
    for page in (v1.PRIMARY, v1.DIAGNOSTIC, v1.LEGACY):
        if not page.is_file():
            raise FileNotFoundError(page)
    primary_page_sha256 = v1.sha(v1.PRIMARY)
    temporary = v1.OUTPUT.with_name(
        f".{v1.OUTPUT.name}.multipage.tmp.pdf"
    )
    v1.run(
        [
            "pdfunite",
            str(v1.PRIMARY),
            str(v1.DIAGNOSTIC),
            str(v1.LEGACY),
            str(temporary),
        ]
    )
    info = v1.run(["pdfinfo", str(temporary)])
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if match is None or int(match.group(1)) != 3:
        raise RuntimeError("combined historical report is not three pages")
    os.replace(temporary, v1.OUTPUT)
    provenance = (
        json.loads(v1.SIDECAR.read_text(encoding="utf-8"))
        if v1.SIDECAR.is_file()
        else {}
    )
    provenance["pdf_report"] = {
        "schema": "e70-current-and-historical-multipage-report-v1",
        "page_count": 3,
        "pages": [
            {
                "page": 1,
                "role": "current_complete_four_metric_outcome_square",
                "source_pdf_sha256_before_assembly": primary_page_sha256,
            },
            {
                "page": 2,
                "role": "current_mechanism_diagnostics",
                "source_pdf": str(v1.DIAGNOSTIC.relative_to(v1.ROOT)),
                "source_pdf_sha256": v1.sha(v1.DIAGNOSTIC),
            },
            {
                "page": 3,
                "role": (
                    "sealed_e58_e66_e69_math500_historical_reference"
                ),
                "source_png": (
                    "paper/figures/"
                    "e68_e58_vs_grpo_05b_12ep_live.png"
                ),
                "source_pdf": str(v1.LEGACY.relative_to(v1.ROOT)),
                "source_pdf_sha256": v1.sha(v1.LEGACY),
            },
        ],
        "assembled_pdf_sha256": v1.sha(v1.OUTPUT),
    }
    v1.atomic_json(v1.SIDECAR, provenance)
    print(f"[e70-multipage-v2] wrote 3-page report {v1.OUTPUT}")


if __name__ == "__main__":
    main()
