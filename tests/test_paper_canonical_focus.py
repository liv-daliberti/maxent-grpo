"""Narrative and evidence contracts for the ModeBench-centered manuscript.

These tests track the manuscript's current terminal five-domain form. They
guard properties the prose must keep rather than its exact wording: the paper
is centred on ModeBench, it reports every domain it claims, its headline design
is the preregistered one, its dataset identities match the audit files, and it
narrates design decisions rather than cluster chronology.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/main.tex"
PYTHON_IDENTITY = (
    ROOT / "paper/results/python_factor_modebench_v1_identity.json"
)

# The five executable domains the manuscript reports on one common design.
MODEBENCH_DOMAINS = (
    "Graph coloring",
    "Countdown",
    "Python factors",
    "MathIR action menu",
    "PantryPlan",
)


def _main_body() -> str:
    return MAIN.read_text(encoding="utf-8").split(
        r"\bibliographystyle", maxsplit=1
    )[0]


def _expand_vendor_macros(text: str) -> str:
    """Resolve the vendor-logo macros so assertions read the rendered name.

    Model names are typeset through icon macros, so searching the raw source
    for a plain vendor name would fail for a formatting reason rather than a
    content one.
    """

    return text.replace(r"\qwenmark{}", "Qwen").replace(r"\qwenmark", "Qwen")


def test_main_paper_centers_modebench_without_internal_experiment_ids():
    text = MAIN.read_text(encoding="utf-8")
    main_text = _main_body()

    assert r"\mb{} and x-Mode GRPO" in main_text
    assert r"\section{\mb: Executable Mode Measurement}" in main_text
    assert r"\section{\xdr: Online Verified MaxEnt-Dr.GRPO}" in main_text
    # Internal experiment identifiers (E68, E70a, ...) are repository
    # bookkeeping and must never reach the manuscript.
    assert re.search(r"\bE[0-9]+[A-Za-z-]*\b", text) is None


def test_paper_reports_exactly_the_five_executable_modebench_domains():
    prose = " ".join(MAIN.read_text(encoding="utf-8").split())

    for domain in MODEBENCH_DOMAINS:
        assert rf"\subsection{{{domain}}}" in prose, domain
    assert "five domains" in prose
    # A sixth domain subsection would mean the prompt appendix and the headline
    # table have drifted apart.
    appendix = prose.split(r"\section{Domain Prompts}", maxsplit=1)[1]
    appendix = appendix.split(r"\section{Dataset and Identity Audits}")[0]
    assert appendix.count(r"\subsection{") == len(MODEBENCH_DOMAINS)


def test_headline_rows_run_the_preregistered_terminal_design():
    prose = " ".join(_expand_vendor_macros(MAIN.read_text(encoding="utf-8")).split())

    for required in (
        "Qwen2.5-0.5B-Instruct",
        # Both base-model families must be named in the body.
        "Falcon3-1B-Instruct",
        "pinned revision",
        "seeds 43--47",
        "group size 16",
        "384 prompts",
        "128-prompt",
        "greedy decoding",
    ):
        assert required in prose, required
    # Twelve epochs is the common terminal endpoint every headline cell reports.
    assert "12 epochs" in prose


def test_python_dataset_identity_matches_the_manuscript_audit():
    identity = json.loads(PYTHON_IDENTITY.read_text(encoding="utf-8"))
    text = MAIN.read_text(encoding="utf-8")

    assert identity == {
        "case_count": 4,
        "eval_rows": 128,
        "eval_rows_sha256": (
            "be0f621c5a0ae84ca45ef4ab866ae12f4c64472eeed1a0ae4666918ab794d183"
        ),
        "external_verifier": "isolated_python_jsonl_worker",
        "max_value": 96,
        "maximum_exact_mode_count": 3600,
        "minimum_exact_mode_count": 16,
        "schema": "python_factor_modebench_v1",
        "seed": 5100,
        "support": "finite_exact",
        "train_rows": 384,
        "train_rows_sha256": (
            "bcfa9accfa3b5c7dd312c85683157af070e7f2741385e9b5a882fd57da037cf7"
        ),
    }
    # The audit table prints elided digests. The elision length is a typesetting
    # choice, but both halves must be genuine prefix/suffix of the identity file
    # rather than free-typed text.
    for digest in (identity["train_rows_sha256"], identity["eval_rows_sha256"]):
        printed = re.search(
            rf"\\texttt{{{digest[:8]}\\ldots([0-9a-f]+)}}", text
        )
        assert printed is not None, digest
        assert digest.endswith(printed.group(1)), digest
    assert "16--3,600" in " ".join(text.split())


def test_design_history_records_core_decisions_not_scheduler_chronology():
    # The component-necessity narrative lives in an appendix, so this contract
    # is read over the whole manuscript rather than the pre-bibliography body.
    text = MAIN.read_text(encoding="utf-8").lower()

    # The manuscript explains why the mechanism has the shape it does.
    for required in (
        "executable gate and canonical key",
        "separated support",
        "replay",
    ):
        assert required in text, required
    # Cluster bookkeeping is repository history, not a scientific result.
    for forbidden in (
        "slurm",
        "sbatch",
        "requeue",
        "preempt",
        "node105",
        "node302",
        "job id",
    ):
        assert forbidden not in text, forbidden
