#!/usr/bin/env python3
"""Review or synchronize current paper assets into the distinct workshop source.

TeX prose is edited separately. This copies its referenced figures/tables and
provenance, preserves superseded files, and binds the reviewed workshop sources.
Use --apply only after both manuscripts' prose and figures are ready to build.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
WORKSHOP = PAPER / "mathai2026"
SOURCES = ("main.tex", "appendix.tex", "preamble.tex")
REFERENCE = re.compile(r"\\(input|include|includegraphics|bibliography)\s*(?:\[[^\]]*\]\s*)?\{([^{}]+)\}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def referenced_assets() -> set[str]:
    assets = set()
    for name in SOURCES:
        text = re.sub(r"(?<!\\)%[^\n]*", "", (WORKSHOP / name).read_text())
        for command, raw in REFERENCE.findall(text):
            for item in raw.split(",") if command == "bibliography" else [raw]:
                path = Path(item.strip())
                if not path.suffix:
                    path = path.with_suffix(".pdf" if command == "includegraphics" else
                                            ".bib" if command == "bibliography" else ".tex")
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"reference leaves standalone workshop: {path}")
                if path.as_posix() not in SOURCES:
                    assets.add(path.as_posix())
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="analysis date YYYY-MM-DD")
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--apply", action="store_true", help="copy assets and bind sources after preserving previous versions")
    args = parser.parse_args()
    stamp = date.fromisoformat(args.date).strftime("%Y%m%d")
    snapshot_path = WORKSHOP / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    if digest(WORKSHOP / "neurips_2026.sty") != snapshot["style_sha256"]:
        raise ValueError("official workshop style differs from its recorded hash")
    names = referenced_assets()
    for name in tuple(names):
        if name.startswith("figures/") and name.endswith(".pdf"):
            for suffix in (".json", ".png"):
                sidecar = Path(name).with_suffix(suffix).as_posix()
                if (PAPER / sidecar).is_file():
                    names.add(sidecar)
    for pattern in (f"results/*{stamp}*", f"figures/current_campaign_results_{stamp}.*",
                    "figures/modebench_level_terminal_by_domain.*"):
        names.update(path.relative_to(PAPER).as_posix() for path in PAPER.glob(pattern) if path.is_file())
    names.add("results/modebench_level_comparison_snapshot.json")
    missing = [name for name in sorted(names) if not (PAPER / name).is_file()]
    if missing:
        raise ValueError(f"parent assets required by workshop are missing: {missing}")
    # Check the previous bindings before overwriting any copied asset. Workshop
    # TeX hashes may differ here because the prose update precedes synchronization.
    for name, expected in snapshot["copied_files"].items():
        if digest(WORKSHOP / name) != expected:
            raise ValueError(f"workshop asset changed outside its snapshot: {name}")
    records = [{"path": name,
                "before_sha256": digest(WORKSHOP / name) if (WORKSHOP / name).is_file() else None,
                "after_sha256": digest(PAPER / name)} for name in sorted(names)]
    report = {"schema": "paper-workshop-asset-sync-v1", "date": args.date,
              "reason": args.reason, "assets": records,
              "source_sha256_before": snapshot["source_sha256"],
              "source_sha256_after": digest(PAPER / "main.tex"),
              "workshop_sources_before": snapshot["workshop_sources"],
              "workshop_sources_after": {name: digest(WORKSHOP / name) for name in SOURCES}}
    changed = [row for row in records if row["before_sha256"] != row["after_sha256"]]
    print(json.dumps({"apply": args.apply, "assets": len(records), "changed_assets": changed}, indent=2))
    if not args.apply:
        return
    if (not changed and snapshot["snapshot_date"] == args.date
            and snapshot["source_sha256"] == report["source_sha256_after"]
            and snapshot["workshop_sources"] == report["workshop_sources_after"]
            and all(snapshot["copied_files"].get(row["path"]) == row["after_sha256"]
                    for row in records)):
        print("Workshop source and asset bindings are already current.")
        return
    audit = args.audit_directory.resolve()
    audit.mkdir(parents=True, exist_ok=True)
    if (audit / "workshop_sync.json").exists():
        raise ValueError("sync audit already exists; use a new directory to preserve history")
    backup = audit / "before"
    backup.mkdir()
    for name in ["snapshot.json", *SOURCES, "Makefile", "README.md", *[row["path"] for row in changed]]:
        source = WORKSHOP / name
        if source.is_file():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for row in records:
        name = row["path"]
        if row in changed:
            destination = WORKSHOP / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PAPER / name, destination)
        snapshot["copied_files"][name] = row["after_sha256"]
    snapshot["source_sha256"] = report["source_sha256_after"]
    snapshot["snapshot_date"] = args.date
    snapshot["workshop_sources"] = report["workshop_sources_after"]
    snapshot.setdefault("correction_history", []).append({
        **report, "audit": str(audit.relative_to(ROOT)) if audit.is_relative_to(ROOT) else str(audit),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    candidate = snapshot_path.with_suffix(".json.tmp")
    candidate.write_text(json.dumps(snapshot, indent=2) + "\n")
    candidate.replace(snapshot_path)
    (audit / "workshop_sync.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Workshop assets synchronized; rebuild with make -C paper/mathai2026 bundle.")


if __name__ == "__main__":
    main()
