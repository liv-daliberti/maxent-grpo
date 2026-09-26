"""Contract checks for the paper's separated terminal comparison figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Paper figure contract failed: {message}")

def _check_records(
    stem: str,
    records: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    require(
        list(records) == expected["domains"],
        f"{stem} record order drifted",
    )
    for domain, record in records.items():
        require(
            record.get("paired_seeds_by_pass", {}).get("8.0")
            == expected["seeds"],
            f"{stem}/{domain} lacks all five paired seeds at pass 8",
        )
        semantic = record.get("semantic_seeds_by_arm", {})
        require(
            set(semantic) == expected["semantic_arms"],
            f"{stem}/{domain} has the wrong attached treatment arms",
        )
        for arm, seeds in semantic.items():
            require(
                seeds == expected["seeds"],
                f"{stem}/{domain}/{arm} lacks all five terminal seeds",
            )


def check(
    manuscript: str,
    comparison_dir: Path,
    specs: dict[str, dict[str, Any]],
) -> None:
    """Bind every compiled comparison figure to terminal five-seed JSON."""

    require(
        "figures/figure4_interim_20260806.pdf" not in manuscript,
        "the 18-panel interim wall is still compiled",
    )
    require(
        r"\label{fig:interim-replay}" not in manuscript,
        "the obsolete interim-wall label remains",
    )
    for stem, expected in specs.items():
        paths = {
            suffix: comparison_dir / f"{stem}.{suffix}"
            for suffix in ("json", "pdf", "png")
        }
        for suffix, path in paths.items():
            require(
                path.is_file() and path.stat().st_size > 0,
                f"terminal comparison {stem}.{suffix} is missing or empty",
            )
        include = rf"figures/comparisons/{stem}.pdf"
        require(
            manuscript.count(include) == 1,
            f"terminal comparison {stem} must be compiled exactly once",
        )
        require(
            manuscript.count(rf"\label{{{expected['label']}}}") == 1,
            f"terminal comparison {stem} has the wrong manuscript label",
        )

        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        if "rows" in expected:
            require(
                payload.get("schema")
                == "paper-comparison-cross-scale-figure-v1"
                and payload.get("evidence") == "terminal"
                and payload.get("status")
                == "terminal five-seed manuscript evidence by displayed row"
                and payload.get("layout")
                == "one physical line of five environments; model scales overlaid",
                f"terminal comparison {stem} has nonterminal cross-scale provenance",
            )
            require(
                payload.get("comparison") == expected["comparison"]
                and set(payload.get("methods", [])) == expected["methods"],
                f"terminal comparison {stem} cross-scale identity drifted",
            )
            actual_rows = payload.get("rows", [])
            require(
                len(actual_rows) == len(expected["rows"]),
                f"terminal comparison {stem} has the wrong displayed rows",
            )
            for actual, row_expected in zip(actual_rows, expected["rows"]):
                require(
                    actual.get("row") == row_expected["row"]
                    and actual.get("scale") == row_expected["scale"]
                    and actual.get("environment_columns")
                    == [
                        "graph_coloring", "countdown", "python_factors",
                        "mathir", "pantry_plan",
                    ]
                    and actual.get("domains") == row_expected["domains"]
                    and actual.get("missing_domains")
                    == row_expected.get("missing_domains", [])
                    and actual.get("seeds") == row_expected["seeds"]
                    and 1 <= len(actual.get("domains", [])) <= 5,
                    f"terminal comparison {stem}/{row_expected['row']} drifted",
                )
                _check_records(
                    f"{stem}/{row_expected['row']}",
                    actual.get("records", {}),
                    row_expected,
                )
            for forbidden in expected.get("forbidden_stems", ()):
                require(
                    f"figures/comparisons/{forbidden}.pdf" not in manuscript,
                    f"split factorial figure {forbidden} remains compiled",
                )
            continue
        require(
            payload.get("schema") == "paper-comparison-family-figure-v2"
            and payload.get("evidence") == "terminal"
            and payload.get("status") == "terminal five-seed manuscript evidence",
            f"terminal comparison {stem} has nonterminal provenance",
        )
        require(
            payload.get("comparison") == expected["comparison"]
            and payload.get("scale") == expected["scale"]
            and payload.get("domains") == expected["domains"]
            and set(payload.get("methods", [])) == expected["methods"],
            f"terminal comparison {stem} identity drifted",
        )
        require(
            1 <= len(payload["domains"]) <= 3,
            f"terminal comparison {stem} exceeds the three-panel limit",
        )
        _check_records(stem, payload.get("records", {}), expected)
