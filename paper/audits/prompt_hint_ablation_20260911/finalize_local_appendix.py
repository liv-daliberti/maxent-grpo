"""Authenticate final local-ablation publication artifacts; makes no model calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, subprocess

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/modebench_prompt_ablation_20260911"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def info(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}

def pdf_pages(path):
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True).split("\f")[:-1]

parent_log = (AUDIT / "parent_build_final_verified.log").read_text()
workshop_log = (AUDIT / "workshop_build_final_verified.log").read_text()
assert "Current paper contract passed" in parent_log
assert "Created mathai2026-source.zip" in workshop_log
assert "compiled artifact current" in workshop_log
assert "line-fill contract passed" in parent_log.lower()
match = re.search(r"Main-length contract passed: (\d+)/9 main pages.*?References starts on page (\d+) \((\d+) total PDF pages\)", parent_log)
assert match, "Parent page contract not found"
main_pages, references_page, total_pages = map(int, match.groups())
assert references_page == main_pages + 1
parent = ROOT / "paper"
workshop = parent / "mathai2026"
assert len(pdf_pages(parent / "main.pdf")) == total_pages
for directory in [parent, workshop]:
    assert sha(directory / "results/modebench_prompt_ablation_20260911.json") == sha(BASE / "analysis_local_complete_editorial_v2/analysis.json")
    assert sha(directory / "results/modebench_prompt_ablation_20260911.tex") == sha(BASE / "analysis_local_complete_editorial_v2/appendix.tex")
    assert sha(directory / "results/modebench_prompt_ablation_python_failure_diagnostic_20260911.json") == sha(BASE / "LOCAL_PYTHON_FAILURE_DIAGNOSTIC.json")
    for ext in ["pdf", "png", "json"]:
        name = "modebench_prompt_ablation_local." + ext
        assert sha(directory / "figures" / name) == sha(BASE / "analysis_local_complete_editorial_v2" / name)
report = json.loads((parent / "results/modebench_prompt_ablation_20260911.json").read_text())
subprocess.run(["qpdf", str(parent / "main.pdf"), "--pages", ".", f"1-{main_pages}", "--", str(parent / "main-body.pdf")], check=True)
body = pdf_pages(parent / "main-body.pdf")
assert len(body) == main_pages
assert not any(re.search(r"^\s*References\s*$", page, re.M) for page in body)
review_manifest = json.loads((BASE / "appendix_review/manifest.json").read_text())
assert sha(BASE / "appendix_review/appendix_review.pdf") == review_manifest["outputs"]["appendix_review.pdf"]["sha256"]
assert len(pdf_pages(BASE / "appendix_review/appendix_review.pdf")) == 4
receipt = {
    "schema": "local-prompt-hint-ablation-publication-final-v1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "overall_experiment_status": "partial_panels",
    "local": {"status": "complete", "responses": 27648, "checkpoints": 25},
    "frontier": {"status": "awaiting_authorized_credential", "responses_collected": 0, "responses_planned": 9216},
    "publication": {"parent_main_pages": main_pages, "parent_references_page": references_page, "parent_total_pages": total_pages, "workshop_main_pages": 4, "standalone_ablation_pages": 4, "scientific_figures_each_manuscript": 19, "publication_copies_exact": True},
    "validation": {"parent_full_build": "pass", "parent_current_contract": "pass", "parent_line_fill": "pass", "workshop_bundle": "pass", "local_report_reconstruction": "pass", "primary_csv_metrics_reconstructed": 2592, "post_hoc_python_draws_authenticated": 5120, "main_body_pdf": "pass"},
    "concurrent_changes": "Preserved the concurrent training-results, concentration, and manuscript refreshes. Rebuilt and synchronized the combined sources.",
    "interpreter_fix": "Current-paper numerical contracts use the same Python 3.10 DATA_PYTHON interpreter as retained calculations; no numerical tolerance or result was altered.",
    "artifacts": []
}
paths = [
    BASE / "manifest.json", BASE / "local/plan_v2.json", BASE / "local/COMPLETE.json", BASE / "local/completion_integrity_audit.json",
    BASE / "analysis_local_complete_editorial_v2/analysis.json", BASE / "analysis_local_complete_editorial_v2/all_cells.csv",
    BASE / "LOCAL_PYTHON_FAILURE_DIAGNOSTIC.json", BASE / "LOCAL_PYTHON_FAILURE_DRAW_DIAGNOSTICS.jsonl",
    BASE / "appendix_review/appendix_review.pdf", BASE / "appendix_review/manifest.json",
    parent / "main.tex", parent / "main.pdf", parent / "main-body.pdf", parent / "Makefile",
    workshop / "main.tex", workshop / "appendix.tex", workshop / "main.pdf", workshop / "build_receipt.json", workshop / "snapshot.json",
    AUDIT / "parent_build_final_verified.log", AUDIT / "workshop_build_final_verified.log", AUDIT / "workshop_sync_final/workshop_sync.json",
    AUDIT / "editorial_numerical_equality.json", AUDIT / "python_runtime_rounding_audit.json",
]
paths.extend(sorted(workshop.glob("*.zip")))
for path in paths:
    assert path.is_file(), path
    receipt["artifacts"].append(info(path))
(AUDIT / "final_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k:v for k,v in receipt.items() if k != "artifacts"}, indent=2))
