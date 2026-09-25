"""Exercise workshop input closure and archival against isolated paper trees.

The workshop owns its TeX roots, while copied nested inputs must be traversed
from the current parent manuscript. Retiring an asset binding must preserve the
old evidence and must not accidentally copy unrelated research assets.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "ops/sync_paper_workshop_assets.py"
STAMP = "20260911"
DATE = "2026-09-11"
# Scientific companions are intentionally retained even when TeX does not
# input JSON. The fixture supplies these real contract filenames as inert data.
SCIENTIFIC_RECORDS = (
    "core_terminal_endpoints.json", "absolute_support_references.json",
    "baseline_collapse_precheck.json", "e78_python_collapse_diagnostics.json",
    "conditional_concentration_20260911.json",
    "e120_frequency_progress.json", "e120_primary_breadth.json",
    "e121_fixed_bank_survival.json", "modebench_level_comparison_snapshot.json",
    "modebench_level3_comparison_snapshot.json",
        "modebench_level_baseline_snapshot.json",
    f"training_curve_snapshot_{STAMP}.json", f"latest_results_{STAMP}.json",
    f"current_campaign_results_{STAMP}.json", f"level2_factorial_contrasts_{STAMP}.json",
    f"frontier_hosted_{STAMP}.json", f"frontier_comparison_{STAMP}.json",
    f"frontier_temperature_{STAMP}.json", f"frontier_python_retry_{STAMP}.json",
)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def sync_tree(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("workshop_asset_sync_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paper = tmp_path / "paper"
    workshop = paper / "mathai2026"
    workshop.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PAPER", paper)
    monkeypatch.setattr(module, "WORKSHOP", workshop)
    for name in module.SOURCES:
        write(workshop / name, "")
    return module


def invoke(module, monkeypatch, audit, *, apply=True):
    argv = [str(SCRIPT), "--date", DATE, "--audit-directory", str(audit),
            "--reason", "Exercise retained workshop asset contract"]
    if apply:
        argv.append("--apply")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def prepare_apply_fixture(module):
    paper, workshop = module.PAPER, module.WORKSHOP
    write(paper / "main.tex", r"Long source: \input{results/parent_only}")
    write(workshop / "main.tex", r"\title{Workshop fixture} Workshop prose: \input{appendix}\input{preamble}\bibliography{example_paper}")
    write(workshop / "appendix.tex", r"\input{sections/active}")
    write(workshop / "preamble.tex", "% Workshop-owned preamble\n")
    write(workshop / "neurips_2026.sty", "official fixture style")
    write(workshop / "README.md", "workshop documentation")
    write(workshop / "Makefile", "# workshop build recipe\n")
    sources = {
        "sections/active.tex": r"\includegraphics{figures/active}\input{results/current_table}",
        "figures/active.pdf": "current primary figure",
        "figures/active.png": "current primary preview",
        "figures/active.json": '{"source": "current primary data"}',
        "results/current_table.tex": "Current table cells",
        "example_paper.bib": "@article{current, title={Current paper}}",
    }
    for name, content in sources.items():
        write(paper / name, content)
    for name in SCIENTIFIC_RECORDS:
        write(paper / "results" / name, "{}")
    # None of these parent files is in the active workshop closure.
    write(paper / "figures/retired.pdf", "new research version of retired figure")
    write(paper / "figures/retired.json", '{"retired": "new research version"}')
    write(paper / "results/retired_progress.tex", "new obsolete progress table")
    write(paper / "results/parent_only.tex", "not in the distinct workshop prose")
    old = {
        "sections/active.tex": r"\includegraphics{figures/retired}",
        "figures/active.pdf": "previous primary figure",
        "figures/retired.pdf": "frozen retired figure",
        "results/retired_progress.tex": "frozen obsolete progress table",
    }
    for name, content in old.items():
        write(workshop / name, content)
    snapshot = {
        "snapshot_date": "2026-09-10",
        "style_sha256": module.digest(workshop / "neurips_2026.sty"),
        "source_sha256": "previous parent source hash",
        "workshop_sources": {name: "previous " + name for name in module.SOURCES},
        "copied_files": {name: module.digest(workshop / name) for name in old},
        "correction_history": [],
    }
    write(workshop / "snapshot.json", json.dumps(snapshot, indent=2) + "\n")
    expected = set(sources) | {f"results/{name}" for name in SCIENTIFIC_RECORDS}
    return old, snapshot, expected


def test_closure_uses_parent_current_nested_inputs_and_workshop_owned_roots(sync_tree):
    module = sync_tree
    write(module.WORKSHOP / "main.tex", r"\input{appendix}\input{preamble}")
    write(module.WORKSHOP / "appendix.tex", r"\input{results/current}")
    write(module.WORKSHOP / "results/current.tex", r"\includegraphics{figures/stale}")
    write(module.PAPER / "results/current.tex", r"\input{sections/deeper}")
    write(module.PAPER / "sections/deeper.tex", r"\includegraphics[width=.9\linewidth]{figures/current}\bibliography{references_a,references_b}")
    # Reading parent main/preamble instead would discover this incorrect asset.
    write(module.PAPER / "main.tex", r"\includegraphics{figures/parent_only}")
    write(module.PAPER / "preamble.tex", r"\includegraphics{figures/parent_preamble}")
    assert module.referenced_assets() == {
        "results/current.tex", "sections/deeper.tex", "figures/current.pdf",
        "references_a.bib", "references_b.bib",
    }


def test_input_cycles_terminate_and_commented_assets_are_excluded(sync_tree):
    module = sync_tree
    write(module.WORKSHOP / "main.tex", r"\input{sections/first}")
    write(module.PAPER / "sections/first.tex", r"\include{sections/second}\input{main}")
    write(module.PAPER / "sections/second.tex", "\\input{sections/first}\n% \\input{missing}\n\\includegraphics{figures/current}% \\includegraphics{figures/retired}\n")
    assert module.referenced_assets() == {
        "sections/first.tex", "sections/second.tex", "figures/current.pdf",
    }


@pytest.mark.parametrize("reference", ["../outside", "/tmp/outside"])
def test_nested_reference_cannot_escape_standalone_workshop(sync_tree, reference):
    module = sync_tree
    write(module.WORKSHOP / "main.tex", r"\input{sections/first}")
    write(module.PAPER / "sections/first.tex", "\\input{" + reference + "}")
    with pytest.raises(ValueError, match="leaves standalone workshop"):
        module.referenced_assets()


def test_apply_copies_only_active_assets_and_archives_retired_bindings(sync_tree, monkeypatch):
    module = sync_tree
    old, original, expected = prepare_apply_fixture(module)
    original_roots = {name: (module.WORKSHOP / name).read_bytes() for name in module.SOURCES}
    audit = module.ROOT / "audit"
    invoke(module, monkeypatch, audit)
    updated = json.loads((module.WORKSHOP / "snapshot.json").read_text())
    assert set(updated["copied_files"]) == expected
    for name in expected:
        assert (module.WORKSHOP / name).read_bytes() == (module.PAPER / name).read_bytes()
        assert updated["copied_files"][name] == module.digest(module.PAPER / name)
    assert all((module.WORKSHOP / name).read_bytes() == content for name, content in original_roots.items())
    assert not (module.WORKSHOP / "figures/retired.json").exists()
    assert not (module.WORKSHOP / "results/parent_only.tex").exists()
    # Retiring a binding does not overwrite or delete its old scientific file.
    retired = {name: original["copied_files"][name]
               for name in ("figures/retired.pdf", "results/retired_progress.tex")}
    for name in retired:
        assert (module.WORKSHOP / name).read_text() == old[name]
    report = json.loads((audit / "workshop_sync.json").read_text())
    assert report["archived_asset_bindings"] == retired
    assert updated["correction_history"][-1]["archived_asset_bindings"] == retired
    assert json.loads((audit / "before/snapshot.json").read_text()) == original
    assert (audit / "before/figures/active.pdf").read_text() == old["figures/active.pdf"]
    assert (audit / "before/sections/active.tex").read_text() == old["sections/active.tex"]


def test_binding_only_retirement_is_recorded_and_reapply_is_idempotent(sync_tree, monkeypatch):
    module = sync_tree
    prepare_apply_fixture(module)
    invoke(module, monkeypatch, module.ROOT / "initial_audit")
    snapshot_path = module.WORKSHOP / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    retired_name = "figures/retired.pdf"
    retired_hash = module.digest(module.WORKSHOP / retired_name)
    snapshot["copied_files"][retired_name] = retired_hash
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    audit = module.ROOT / "binding_only_audit"
    invoke(module, monkeypatch, audit)
    report = json.loads((audit / "workshop_sync.json").read_text())
    assert all(row["before_sha256"] == row["after_sha256"] for row in report["assets"])
    assert report["archived_asset_bindings"] == {retired_name: retired_hash}
    updated = json.loads(snapshot_path.read_text())
    assert retired_name not in updated["copied_files"]
    assert len(updated["correction_history"]) == 2
    before_reapply = snapshot_path.read_bytes()
    invoke(module, monkeypatch, audit)
    assert snapshot_path.read_bytes() == before_reapply
    assert len(json.loads(snapshot_path.read_text())["correction_history"]) == 2


def test_dry_run_preserves_workshop_and_does_not_create_audit(sync_tree, monkeypatch):
    module = sync_tree
    prepare_apply_fixture(module)
    before = {p.relative_to(module.WORKSHOP): p.read_bytes()
              for p in module.WORKSHOP.rglob("*") if p.is_file()}
    audit = module.ROOT / "unused_audit"
    invoke(module, monkeypatch, audit, apply=False)
    after = {p.relative_to(module.WORKSHOP): p.read_bytes()
             for p in module.WORKSHOP.rglob("*") if p.is_file()}
    assert after == before
    assert not audit.exists()


def test_retiring_a_tampered_binding_fails_before_any_copy(sync_tree, monkeypatch):
    module = sync_tree
    old, _, _ = prepare_apply_fixture(module)
    snapshot_before = (module.WORKSHOP / "snapshot.json").read_bytes()
    write(module.WORKSHOP / "figures/retired.pdf", "unreviewed local change")
    audit = module.ROOT / "rejected_audit"
    with pytest.raises(ValueError, match="asset changed outside its snapshot"):
        invoke(module, monkeypatch, audit)
    assert not audit.exists()
    assert (module.WORKSHOP / "snapshot.json").read_bytes() == snapshot_before
    assert (module.WORKSHOP / "figures/active.pdf").read_text() == old["figures/active.pdf"]
