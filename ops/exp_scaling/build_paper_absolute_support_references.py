#!/usr/bin/env python3
"""Build absolute distinct@8 references for the complete Level-1 core grid."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_domain_typography import format_domain_names

ROOT = Path(__file__).resolve().parents[2]
PRECHECK = ROOT / "paper/results/baseline_collapse_precheck.json"
CORE = ROOT / "paper/results/core_terminal_endpoints.json"
OUTPUT = ROOT / "paper/results/absolute_support_references.json"
TABLE_OUTPUT = ROOT / "paper/results/absolute_support_references_table_body.tex"

MODELS = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
SCALES = {"Qwen2.5-0.5B": "qwen05b", "Falcon3-1B": "falcon1b", "Qwen2.5-3B": "qwen3b"}
MODEL_LABELS = {"Qwen2.5-0.5B": r"\qwenmark{}2.5-0.5B", "Falcon3-1B": "Falcon3-1B", "Qwen2.5-3B": r"\qwenmark{}2.5-3B"}
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABELS = {"graph_coloring": "Graph", "countdown": "Countdown", "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "PantryPlan"}

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def build() -> dict:
    precheck, core = load(PRECHECK), load(CORE)
    support_means = {"graph_coloring": 6.265625, "pantry_plan": 18.1875}
    uninformed = {
        "graph_coloring": {"primary": support_means["graph_coloring"] * (1 - (26 / 27) ** 8), "policy": "uniform over all 27 three-color assignments", "candidate_count": 27},
        "pantry_plan": {"primary": support_means["pantry_plan"] * (1 - (63 / 64) ** 8), "policy": "uniform over all 64 six-bit masks", "candidate_count": 64, "instruction_following": support_means["pantry_plan"] * (1 - (49 / 50) ** 8), "instruction_following_policy": "uniform over the 50 masks selecting two to four ingredients"},
    }
    rows = []
    for model in MODELS:
        scale = SCALES[model]
        for domain in DOMAINS:
            methods = core["models"][model]["domains"][domain]["methods"]
            rows.append({
                "model": model, "scale": scale, "domain": domain,
                "frozen_distinct8": float(precheck["arms"]["drgrpo"]["scales"][scale]["domains"][domain]["endpoints"]["pass0"]["distinct8"]["mean"]),
                "drgrpo_distinct8": float(methods["control"]["summary"]["distinct8"]["mean"]),
                "replay_drgrpo_distinct8": float(methods["replay"]["summary"]["distinct8"]["mean"]),
                "uninformed_distinct8": uninformed.get(domain, {}).get("primary"),
            })
    return {"schema": "paper-absolute-support-references-v1", "generated_at": datetime.now(timezone.utc).isoformat(), "metric": "distinct@8", "reference_scope": "Analytic action-uniform references are reported only for finite fixed-width action spaces; null means no comparable uninformed generator is specified.", "support_means": support_means, "uninformed_policies": uninformed, "rows": rows}

def cell(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}".removeprefix("0")

def render(payload: dict) -> str:
    lines = []
    for model in MODELS:
        first = True
        for row in (row for row in payload["rows"] if row["model"] == model and row["uninformed_distinct8"] is not None):
            values = (row["uninformed_distinct8"], row["frozen_distinct8"], row["drgrpo_distinct8"], row["replay_drgrpo_distinct8"])
            lines.append("    " + (MODEL_LABELS[model] if first else "") + " & " + DOMAIN_LABELS[row["domain"]] + " & " + " & ".join(cell(value) for value in values) + r" \\")
            first = False
        lines.append(r"    \addlinespace[2pt]")
    return format_domain_names("\n".join(lines[:-1]) + "\n    \\bottomrule\n")

def main() -> int:
    payload = build()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    TABLE_OUTPUT.write_text(render(payload), encoding="utf-8")
    print(OUTPUT)
    print(TABLE_OUTPUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
