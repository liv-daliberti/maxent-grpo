#!/usr/bin/env python3
"""Build current pass-8 core endpoints directly from immutable run ledgers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_domain_typography import format_domain_names
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/results/core_terminal_endpoints.json"
TABLE_OUTPUT = ROOT / "paper/results/core_terminal_endpoints_table_body.tex"
EXPECTED_DRAWS = 4
ENDPOINT_FIELDS = {
    "pass8": "any_correct_at_k",
    "mean8": "mean_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
#: PCMD is not a draw field -- it is read off the mode distribution by the
#: resampling pipeline -- so the printed table joins it from the frozen curve
#: archive rather than recomputing it here.
PMD_CURVES = ROOT / "paper/results/mode_diversity_curves.json"
#: A reported PCMD value carries the whole-cell bar of 30 defined prompts, not
#: the paired bar of 20 that a difference uses: here each arm's number is read
#: on its own, so nothing cancels.
MIN_DEFINED_PROMPTS = 30
PMD_METHODS = {"control": "drgrpo", "replay": "replay_drgrpo"}
# This is an existing, source-specific scientific exclusion, not a choice
# between retry outcomes. The amendment excludes the run from every efficacy
# summary; later reuse of the comparator does not repair its source log.
APPROVED_RUN_EXCLUSIONS = {
    "var/data/xdr_falcon3_1b_instruct_verified_first_replay_"
    "rehearsal_only_e79_falcon_aligned_countdown_replay_s59": {
        "job_id": 30269051,
        "source_log": (
            "var/data/xdr_falcon3_1b_instruct_verified_first_replay_"
            "rehearsal_only_e79_falcon_aligned_countdown_replay_s59/"
            "debug_job30269051/eval_mode_coverage_draws.jsonl"
        ),
        "source_log_sha256": (
            "2f6e2ea3419c617dcedafd0ce2f7da7734ead2ad19c73056c0ae579252d64465"
        ),
        "amendment": (
            "paper/preregistration/"
            "e112r1_two_scale_49pair_terminal_integrity_amendment_20260828.md"
        ),
        "reason": "conflicting duplicate sampled rows in the registered job",
        "scope": "all efficacy endpoints and training steps for this run",
    },
}
SOURCES = {
    "Qwen2.5-0.5B": (
        "qwen05b",
        ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
    ),
    "Falcon3-1B": (
        "falcon1b",
        ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
    ),
    "Qwen2.5-3B": (
        "qwen3b",
        ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
    ),
}
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
ARMS = ("control", "replay")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    return float(value)


def approved_run_exclusion(run_dir: Path) -> dict[str, Any] | None:
    """Return an existing all-step exclusion only after validating its source.

    Independent trajectory readers may use this helper as well as the terminal
    reader. A changed or missing excluded source is a hard failure: this helper
    never silently interprets a new source as a repair of an old exclusion.
    """
    resolved = run_dir.resolve()
    for registered_dir, exclusion in APPROVED_RUN_EXCLUSIONS.items():
        if resolved != (ROOT / registered_dir).resolve():
            continue
        source = ROOT / exclusion["source_log"]
        amendment = ROOT / exclusion["amendment"]
        if not source.is_file() or not amendment.is_file():
            raise RuntimeError(f"approved exclusion source or amendment missing: {run_dir}")
        source_digest = sha256(source)
        if source_digest != exclusion["source_log_sha256"]:
            raise RuntimeError(f"approved exclusion source hash drifted: {source}")
        return {
            **exclusion,
            "run_dir": str(resolved),
            "status": "excluded",
            "source_log_sha256": source_digest,
            "amendment_sha256": sha256(amendment),
            "outcome_value_selected": False,
        }
    return None


def sampled_endpoint(
    run_dir: Path,
    *,
    step: int,
    audit: list[dict[str, Any]] | None = None,
) -> dict[str, float] | None:
    """Read one exact endpoint without choosing among conflicting retry rows.

    Identical metric payloads for the same (step, draw) are duplicate records,
    not additional samples. Any unapproved conflicting payload fails closed.
    The optional audit collector preserves the existing dict-or-None API.
    """
    exclusion = approved_run_exclusion(run_dir)
    if exclusion is not None:
        if audit is not None:
            audit.append({**exclusion, "step": step})
        return None

    records: dict[int, dict[str, Any]] = {}
    origins: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    repeated: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                digest.update(raw)
                try:
                    row = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(row, dict) or (
                    row.get("evaluation_kind") != "fixed_seed_sampled_k_neutral"
                    or row.get("step") != step
                ):
                    continue
                draw = row.get("draw_index")
                metrics = row.get("metrics")
                if not isinstance(draw, int) or not isinstance(metrics, dict):
                    raise RuntimeError(f"malformed sampled endpoint at {path}:{line_number}")
                for field in ENDPOINT_FIELDS.values():
                    finite(metrics.get(field), name=f"{path}:{line_number}:{field}")
                origin = {"path": str(path), "line": line_number, "draw_index": draw}
                if draw in records:
                    if records[draw] != metrics:
                        missing = object()
                        changed = sorted(
                            key for key in set(records[draw]) | set(metrics)
                            if records[draw].get(key, missing) != metrics.get(key, missing)
                        )
                        prior = origins[draw]
                        raise RuntimeError(
                            "conflicting duplicate sampled endpoint "
                            f"step={step}, draw={draw}: "
                            f"{prior['path']}:{prior['line']} versus "
                            f"{path}:{line_number}; differing metric fields={changed}"
                        )
                    repeated.append(origin)
                    continue
                records[draw] = metrics
                origins[draw] = origin
        sources.append({"path": str(path), "sha256": digest.hexdigest()})

    complete = sorted(records) == list(range(EXPECTED_DRAWS))
    if audit is not None:
        audit.append({
            "run_dir": str(run_dir.resolve()),
            "step": step,
            "status": "admitted" if complete else "incomplete",
            "reason": None if complete else "missing or unexpected sampled draw indices",
            "expected_draws": list(range(EXPECTED_DRAWS)),
            "observed_draws": sorted(records),
            "sources": sources,
            "unique_draw_records": [origins[draw] for draw in sorted(origins)],
            "identical_metric_repeats": repeated,
            "conflicting_retry_selected": False,
        })
    if not complete:
        return None
    return {
        output: statistics.fmean(float(records[draw][source]) for draw in range(EXPECTED_DRAWS))
        for output, source in ENDPOINT_FIELDS.items()
    }


def summarize(per_seed: dict[str, dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(per_seed)}
    for metric in ("pass8", "mean8", "distinct8"):
        values = [record[metric] for record in per_seed.values()]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "range": [min(values), max(values)],
        }
    return summary


def build() -> dict[str, Any]:
    models: dict[str, Any] = {}
    source_records: dict[str, Any] = {}
    endpoint_audit: list[dict[str, Any]] = []
    for model, (scale, path) in SOURCES.items():
        ledger = load(path)
        if ledger.get("released") is not True:
            raise RuntimeError(f"{model}: core ledger was not durably released")
        target = int(ledger["target_steps"])
        train_rows = int(ledger["train_rows"])
        if target != train_rows * int(ledger["passes"]):
            raise RuntimeError(f"{model}: target/pass geometry drifted")
        index = {
            (str(run["domain"]), str(run["arm"]), int(run["seed"])): run
            for run in ledger.get("runs", [])
            if str(run.get("domain")) in DOMAINS
            and str(run.get("arm")) in ARMS
        }
        domains: dict[str, Any] = {}
        for domain in DOMAINS:
            methods: dict[str, Any] = {}
            for arm in ARMS:
                per_seed: dict[str, dict[str, float]] = {}
                for (run_domain, run_arm, seed), run in sorted(index.items()):
                    if (run_domain, run_arm) != (domain, arm):
                        continue
                    endpoint = sampled_endpoint(
                        Path(str(run["run_dir"])),
                        step=target,
                        audit=endpoint_audit,
                    )
                    endpoint_audit[-1].update({
                        "model": model, "scale": scale, "domain": domain,
                        "arm": arm, "seed": seed, "registered_job_id": run.get("job_id"),
                    })
                    if endpoint is not None:
                        per_seed[str(seed)] = endpoint
                if per_seed:
                    methods[arm] = {
                        "per_seed": per_seed,
                        "summary": summarize(per_seed),
                    }
            if methods:
                domains[domain] = {
                    "training_pass": target / train_rows,
                    "target_step": target,
                    "methods": methods,
                }
        models[model] = {"scale": scale, "domains": domains}
        source_records[model] = {
            "path": str(path),
            "sha256": sha256(path),
        }
    return {
        "schema": "paper-core-terminal-endpoints-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "every seed with all four registered fixed-seed sampled-K draws "
            "at the exact pass-8 target; arms are retained independently; "
            "approved all-step source exclusions are honored, identical metric "
            "repeats are deduplicated, and unapproved conflicting retries abort"
        ),
        "expected_draws": EXPECTED_DRAWS,
        "domains": list(DOMAINS),
        "sources": source_records,
        "endpoint_integrity_policy": {
            "duplicate_key": ["step", "draw_index"],
            "comparison": "exact equality of the complete sampled metrics payload",
            "identical_repeats": "deduplicate; never treat as additional draws",
            "unapproved_conflicts": "abort without writing a replacement result",
            "approved_exclusions": "omit all efficacy steps; retain reason, source hash, and amendment",
        },
        "endpoint_audit": endpoint_audit,
        "exclusions": [row for row in endpoint_audit if row["status"] == "excluded"],
        "models": models,
    }


def pmd_terminal() -> dict[tuple[str, str, str, int], float]:
    """Terminal PCMD per run, keyed by scale, method, domain and seed.

    Only seeds whose terminal checkpoint defines PCMD on at least
    ``MIN_DEFINED_PROMPTS`` prompts are returned; the rest are absent, which is
    what lets a fully concentrated control print a dash instead of a zero it
    did not measure.
    """
    out: dict[tuple[str, str, str, int], float] = {}
    for curve in json.loads(PMD_CURVES.read_text())["curves"]:
        if curve["level"] != "level1":
            continue
        point = max(curve["points"], key=lambda p: p["step"])
        value, defined = point.get("pmd"), int(point.get("defined_prompts") or 0)
        if value is None or defined < MIN_DEFINED_PROMPTS:
            continue
        out[(curve["scale"], curve["method"], curve["domain"],
             int(curve["seed"]))] = float(value)
    return out


def pmd_cell(table, scale, method, domain, paired):
    """Mean terminal PCMD over the paired seeds that clear the support bar."""
    values = [table[(scale, PMD_METHODS[method], domain, int(seed))]
              for seed in paired
              if (scale, PMD_METHODS[method], domain, int(seed)) in table]
    if not values:
        return None, 0
    return statistics.fmean(values), len(values)


def _mean(records: dict[str, dict[str, float]], seeds: list[str], metric: str) -> float:
    return statistics.fmean(records[seed][metric] for seed in seeds)


#: The shared light ramp, as in the retention matrix tables: low is red, the
#: middle yellow, high green.
RAMP = ((0.00, (0xF0, 0xA5, 0x8F)),
        (0.50, (0xF8, 0xF0, 0xA6)),
        (1.00, (0xA9, 0xD6, 0xA0)))


def _ramp(position: float) -> str:
    position = min(max(position, 0.0), 1.0)
    for (low, left), (high, right) in zip(RAMP, RAMP[1:]):
        if position <= high:
            weight = 0.0 if high == low else (position - low) / (high - low)
            return "".join(f"{round(a + (b - a) * weight):02X}"
                           for a, b in zip(left, right))
    return "".join(f"{v:02X}" for v in RAMP[-1][1])


def _shade(value: float, extent: float) -> str:
    """Colour a level against the largest level of the same metric.

    These are levels, not differences, so the scale runs from zero rather than
    from a midpoint: a \pmd{} of .001 is a collapsed policy, and it should read
    as one. Both arms of a metric share the extent, which is the whole point of
    the table -- the reader is comparing Dr.GRPO against Re:Dr, and a per-column
    scale would give each arm its own meaning of green. The binary pale-green
    mark this replaces fired on almost every Re:Dr cell, so it distinguished
    nothing.
    """
    return rf"\cellcolor[HTML]{{{_ramp(value / extent if extent else 0.0)}}}"


def _cell(value: float, *, extent: float) -> str:
    formatted = f"{value:.3f}".removeprefix("0")
    return f"{_shade(value, extent)}{formatted}"


def _pmd_cell(value, seeds: int, paired: int, *, extent: float) -> str:
    """A PCMD cell, with its own seed count when it is short of the pairing.

    No seed clearing the bar is a dash, not a zero: a control concentrated onto
    one key has no measured diversity rather than a measured diversity of none,
    and an uncoloured cell says that too.
    """
    if value is None:
        return r"\textemdash{}"
    formatted = f"{value:.3f}".removeprefix("0")
    if seeds < paired:
        # Thin space: at \scriptsize a bare superscript digit runs into the
        # value, and ``.000$^{2}$`` reads as the number .0002.
        formatted += rf"$^{{\,{seeds}}}$"
    return f"{_shade(value, extent)}{formatted}"


def render_tex(payload: dict[str, Any]) -> str:
    model_labels = {
        "Qwen2.5-0.5B": r"\qwenmark{}2.5-0.5B",
        "Falcon3-1B": r"\falconmark{}3-1B",
        "Qwen2.5-3B": r"\qwenmark{}2.5-3B",
    }
    domain_labels = {
        "graph_coloring": "Graph coloring",
        "countdown": "Countdown",
        "python_factors": "Python factors",
        "mathir": "MathIR",
        "pantry_plan": "PantryPlan",
    }
    pmd = pmd_terminal()
    # One extent per metric, over both arms and every row, so a shade means the
    # same thing everywhere in the table and the two arms stay comparable.
    pass_levels, pmd_levels = [], []
    for model, model_record in payload["models"].items():
        for domain in DOMAINS:
            domain_record = model_record["domains"].get(domain)
            methods = domain_record.get("methods", {}) if domain_record else {}
            if not {"control", "replay"} <= set(methods):
                continue
            control, replay = methods["control"]["per_seed"], methods["replay"]["per_seed"]
            shared = sorted(set(control) & set(replay), key=int)
            if not shared:
                continue
            pass_levels += [_mean(control, shared, "pass8"), _mean(replay, shared, "pass8")]
            for arm in ("control", "replay"):
                value, _count = pmd_cell(pmd, SOURCES[model][0], arm, domain, shared)
                if value is not None:
                    pmd_levels.append(value)
    pass_extent = max(pass_levels) if pass_levels else 1.0
    pmd_extent = max(pmd_levels) if pmd_levels else 1.0
    lines: list[str] = []
    for model, model_record in payload["models"].items():
        first = True
        for domain in DOMAINS:
            domain_record = model_record["domains"].get(domain)
            methods = domain_record.get("methods", {}) if domain_record else {}
            if not {"control", "replay"} <= set(methods):
                continue
            control = methods["control"]["per_seed"]
            replay = methods["replay"]["per_seed"]
            paired = sorted(set(control) & set(replay), key=int)
            if not paired:
                continue
            # mean@8 and distinct@8 are dropped from the printed table: both
            # move with correctness, which is the confound PCMD exists to
            # remove, and both stay in the machine-readable record.
            control_pass = _mean(control, paired, "pass8")
            replay_pass = _mean(replay, paired, "pass8")
            scale = SOURCES[model][0]
            control_pmd, control_n = pmd_cell(pmd, scale, "control", domain, paired)
            replay_pmd, replay_n = pmd_cell(pmd, scale, "replay", domain, paired)
            model_cell = model_labels[model] if first else ""
            first = False
            values = [
                _cell(control_pass, extent=pass_extent),
                _pmd_cell(control_pmd, control_n, len(paired), extent=pmd_extent),
                _cell(replay_pass, extent=pass_extent),
                _pmd_cell(replay_pmd, replay_n, len(paired), extent=pmd_extent),
            ]
            # The paired-seed column carried one bit: every block is five
            # seeds at pass 8 except Falcon Countdown, which the caption
            # already states. It is dropped, and a block short of five prints
            # its count as a superscript beside the domain instead.
            pass_value = float(domain_record["training_pass"])
            assert abs(pass_value - 8.0) < 1e-9, "a block is not read at pass 8"
            short = "" if len(paired) == 5 else rf"$^{{\,{len(paired)}}}$"
            lines.append(
                f"    {model_cell} & {domain_labels[domain]}{short} & "
                + " & ".join(values)
                + r" \\"
            )
        lines.append(r"    \addlinespace[2pt]")
    return format_domain_names("\n".join(lines[:-1]) + "\n    \\bottomrule\n")


def main() -> int:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    TABLE_OUTPUT.write_text(render_tex(payload), encoding="utf-8")
    counts = {
        model: {
            domain: {
                arm: record["summary"]["n"]
                for arm, record in domain_record["methods"].items()
            }
            for domain, domain_record in model_record["domains"].items()
        }
        for model, model_record in payload["models"].items()
    }
    print(json.dumps(counts, indent=2))
    print(f"wrote {OUTPUT}")
    print(f"wrote {TABLE_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
