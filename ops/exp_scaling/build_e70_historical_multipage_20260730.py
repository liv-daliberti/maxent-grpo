#!/usr/bin/env python3
"""Build the readable three-page current + historical figure report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper/figures"
PRIMARY_RENDERER = (
    ROOT
    / "ops/exp_scaling/"
    "plot_e70_primary_square_repairs_v3_20260730.py"
)
DIAGNOSTIC_RENDERER = (
    ROOT
    / "ops/exp_scaling/"
    "plot_e70_current_diagnostics_square_20260730.py"
)
LEGACY_RENDERER = (
    ROOT
    / "ops/exp_scaling/"
    "render_e68_live_historical_reference_page_20260730.py"
)
OUTPUT = (
    FIGURES
    / "e68_e58_vs_grpo_05b_12ep_terminal_provenance_"
    "historical_20260730.pdf"
)
PRIMARY = OUTPUT
DIAGNOSTIC = (
    FIGURES
    / "e68_e58_vs_grpo_05b_12ep_current_diagnostics_20260730.pdf"
)
LEGACY = (
    FIGURES
    / "e68_e58_vs_grpo_05b_12ep_live_"
    "historical_reference_20260730.pdf"
)
SIDECAR = (
    ROOT
    / "var/artifacts/"
    "e68_e58_vs_grpo_05b_12ep_terminal_provenance_"
    "historical_20260730_provenance.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    return result.stdout


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    for renderer in (
        PRIMARY_RENDERER,
        DIAGNOSTIC_RENDERER,
        LEGACY_RENDERER,
    ):
        run([sys.executable, str(renderer)])
    for page in (PRIMARY, DIAGNOSTIC, LEGACY):
        if not page.is_file():
            raise FileNotFoundError(page)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.multipage.tmp.pdf")
    run(
        [
            "pdfunite",
            str(PRIMARY),
            str(DIAGNOSTIC),
            str(LEGACY),
            str(temporary),
        ]
    )
    info = run(["pdfinfo", str(temporary)])
    match = re.search(r"^Pages:\\s+(\\d+)$", info, flags=re.MULTILINE)
    if match is None or int(match.group(1)) != 3:
        raise RuntimeError("combined historical report is not three pages")
    os.replace(temporary, OUTPUT)
    provenance = (
        json.loads(SIDECAR.read_text(encoding="utf-8"))
        if SIDECAR.is_file()
        else {}
    )
    provenance["pdf_report"] = {
        "schema": "e70-current-and-historical-multipage-report-v1",
        "page_count": 3,
        "pages": [
            {
                "page": 1,
                "role": "current_primary_square",
                "source_pdf_sha256_before_assembly": sha(PRIMARY),
            },
            {
                "page": 2,
                "role": "current_outcome_and_mechanism_diagnostics",
                "source_pdf": str(DIAGNOSTIC.relative_to(ROOT)),
                "source_pdf_sha256": sha(DIAGNOSTIC),
            },
            {
                "page": 3,
                "role": "sealed_e58_e66_e69_math500_historical_reference",
                "source_png": (
                    "paper/figures/"
                    "e68_e58_vs_grpo_05b_12ep_live.png"
                ),
                "source_pdf": str(LEGACY.relative_to(ROOT)),
                "source_pdf_sha256": sha(LEGACY),
            },
        ],
        "assembled_pdf_sha256": sha(OUTPUT),
    }
    atomic_json(SIDECAR, provenance)
    print(f"[e70-multipage] wrote 3-page report {OUTPUT}")


if __name__ == "__main__":
    main()
