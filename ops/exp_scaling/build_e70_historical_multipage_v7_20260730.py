#!/usr/bin/env python3
"""Build the complete readable current + sealed historical figure report."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper/figures"
CURRENT_RENDERER = (
    ROOT
    / "ops/exp_scaling/"
    "plot_e70_current_complete_square_pages_v7_20260730.py"
)
LEGACY_RENDERER = (
    ROOT
    / "ops/exp_scaling/"
    "render_e68_live_historical_reference_page_20260730.py"
)
OUTPUT = (
    FIGURES
    / "e68_e58_vs_grpo_05b_12ep_"
    "terminal_provenance_historical_20260730.pdf"
)
SIDECAR = (
    ROOT
    / "var/artifacts/"
    "e68_e58_vs_grpo_05b_12ep_"
    "terminal_provenance_historical_20260730_provenance.json"
)
CURRENT_PAGES = tuple(
    FIGURES / f"e68_e58_vs_grpo_05b_12ep_{slug}_20260730.pdf"
    for slug in (
        "current_outcomes",
        "current_controller_diagnostics",
        "current_replay_diagnostics",
        "current_support_diagnostics",
    )
)
LEGACY = (
    FIGURES
    / "e68_e58_vs_grpo_05b_12ep_"
    "live_historical_reference_20260730.pdf"
)
ROLES = (
    "current_complete_outcomes",
    "current_controller_diagnostics",
    "current_verified_replay_diagnostics",
    "current_verified_support_and_discovery_diagnostics",
    "sealed_e58_e66_e69_math500_historical_reference",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
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


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    _run([sys.executable, str(CURRENT_RENDERER)])
    _run([sys.executable, str(LEGACY_RENDERER)])
    pages = CURRENT_PAGES + (LEGACY,)
    for page in pages:
        if not page.is_file():
            raise FileNotFoundError(page)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.multipage.tmp.pdf")
    _run(
        [
            "pdfunite",
            *(str(page) for page in pages),
            str(temporary),
        ]
    )
    info = _run(["pdfinfo", str(temporary)])
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if match is None or int(match.group(1)) != len(pages):
        raise RuntimeError(
            f"combined report is not {len(pages)} pages"
        )
    os.replace(temporary, OUTPUT)
    page_records = []
    for page_number, (page, role) in enumerate(
        zip(pages, ROLES),
        start=1,
    ):
        record = {
            "page": page_number,
            "role": role,
            "source_pdf": str(page.relative_to(ROOT)),
            "source_pdf_sha256": _sha256(page),
        }
        if page == LEGACY:
            record["source_png"] = (
                "paper/figures/e68_e58_vs_grpo_05b_12ep_live.png"
            )
            record["source_png_sha256"] = _sha256(
                FIGURES / "e68_e58_vs_grpo_05b_12ep_live.png"
            )
        page_records.append(record)
    provenance = (
        json.loads(SIDECAR.read_text(encoding="utf-8"))
        if SIDECAR.is_file()
        else {}
    )
    provenance["pdf_report"] = {
        "schema": "e70-complete-current-and-historical-multipage-report-v2",
        "page_count": len(pages),
        "pages": page_records,
        "all_july_28_metric_families_present": True,
        "square_current_axes": True,
        "assembled_pdf_sha256": _sha256(OUTPUT),
    }
    _atomic_json(SIDECAR, provenance)
    print(
        f"[e70-multipage-v7] wrote {len(pages)}-page report {OUTPUT}"
    )


if __name__ == "__main__":
    main()
