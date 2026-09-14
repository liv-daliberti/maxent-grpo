#!/usr/bin/env python3
"""Display campaign status with stopped E112-R1 scale detail quiet by default.

This separate presentation entry point invokes the authoritative read-only
campaign CLI unchanged. --include-history and --json pass through verbatim.
It does not import or change any supervisor, receipt, ledger, or helper file.
"""
from __future__ import annotations
import re
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).with_name("campaign_stats.py")
LABEL = "E112-R1 all scales corrected verified-support MaxEnt relaunch"


def present(output: str, args: list[str]) -> str:
    if "--include-history" in args or "--json" in args:
        return output
    kept = []
    for line in output.splitlines(keepends=True):
        scale_detail = line.startswith(f"  SCALE  {LABEL}:") or line.startswith(f"- {LABEL}:")
        running = re.findall(r"\b(\d+) running\b", line) if scale_detail else []
        if scale_detail and running and all(int(count) == 0 for count in running):
            continue
        kept.append(line)
    return "".join(kept)


def main() -> int:
    args = sys.argv[1:]
    result = subprocess.run([sys.executable, str(SOURCE), *args], text=True, capture_output=True)
    sys.stdout.write(present(result.stdout, args) if result.returncode == 0 else result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
