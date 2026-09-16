"""Apply the audited September 10 result delta to the distinct workshop source."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

ROOT = Path('/n/fs/similarity/maxent-grpo')
PAPER = ROOT / 'paper'
WS = PAPER / 'mathai2026'
AUDIT = PAPER / 'audits/results_refresh_20260910/manuscript_revision'
BEFORE = AUDIT / 'before'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:120])
    return text.replace(old, new, 1)

parent_before = BEFORE / 'paper/main.tex'
parent_after = PAPER / 'main.tex'
workshop_before = {p: sha(WS / p) for p in ('main.tex', 'appendix.tex', 'preamble.tex')}
for p in ('main.tex', 'appendix.tex', 'snapshot.json', 'Makefile'):
    assert (WS / p).read_bytes() == (BEFORE / 'paper/mathai2026' / p).read_bytes(), p
shutil.copy2(WS / 'README.md', BEFORE / 'paper/mathai2026/README.md')

old_lines = parent_before.read_text().splitlines(keepends=True)
new_lines = parent_after.read_text().splitlines(keepends=True)
appendix = (WS / 'appendix.tex').read_text()
applied, workshop_distinct = [], []
for tag, i, j, k, l in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
    if tag == 'equal':
        continue
    old, new = ''.join(old_lines[i:j]), ''.join(new_lines[k:l])
    entry = {'parent_before_line': i + 1, 'old': old, 'new': new}
    if old and appendix.count(old) == 1:
        appendix = once(appendix, old, new)
        applied.append(entry)
    else:
        assert i < 850, ('Unmapped shared appendix delta', i + 1)
        workshop_distinct.append(entry)

appendix = once(appendix,
    'three scales. Eleven MaxRL--ReplayMaxRL blocks are complete at five seeds; the\nfull four-arm intersection contains ten complete blocks and one Falcon',
    'three scales. Twelve MaxRL--ReplayMaxRL blocks are complete at five seeds; the\nfull four-arm intersection contains eleven complete blocks and one Falcon')
appendix = once(appendix,
    'incorporate validated results through the September 9 analysis update.',
    'incorporate validated results through the September 10 analysis update.')
appendix = once(appendix,
    'matched interim checkpoints, including an initial-only Pantry reference\n',
    'matched interim checkpoints, including Pantry seeds at steps 672 and 0\n')
appendix = once(appendix,
    'Python and MathIR are also complete, with positive co-primary replay means\nwhose intervals include zero (Table~\\ref{tab:latest-completed-effects}). Countdown\nand PantryPlan remain incomplete.',
    'Countdown, Python, and MathIR are also complete. Countdown adds $+.017$\nextra modes under ReplayMaxRL, while its correctness interval includes zero.\nPython and MathIR have positive co-primary replay means whose intervals\ninclude zero (Table~\\ref{tab:latest-completed-effects}). PantryPlan remains\nincomplete.')
(WS / 'appendix.tex').write_text(appendix)

main = (WS / 'main.tex').read_text()
main = once(main,
    "Adding replay improves average correctness and verified support under either\nobjective at Qwen2.5-0.5B and Falcon3-1B (Figure~\\ref{fig:maxrl-factorial}).\nQwen2.5-3B also benefits in its completed Dr.GRPO comparison.\nMaxRL comparisons use five paired seeds per domain.\nAt Qwen2.5-0.5B, replay raises MaxRL's \\texttt{pass@8} by $.31$ and\n\\texttt{distinct@8} by $.99$ on average across domains\n(paired estimates and intervals: Appendix~\\ref{app:results-maxrl}).",
    "Adding replay improves average correctness and verified support under both\nobjectives at Qwen2.5-0.5B and Falcon3-1B (Figure~\\ref{fig:maxrl-factorial}).\nAt Qwen2.5-0.5B, its MaxRL gains average $.31$ on \\texttt{pass@8} and\n$.99$ on \\texttt{distinct@8}. The completed five-seed Qwen2.5-3B Graph block\nadds $.163$ extra modes (95\\% interval $[.021,.305]$); its correctness and\nraw-support intervals include zero (Appendix~\\ref{app:results-maxrl}).")
main = once(main,
    'Checkpoints match across levels and methods; Pantry has one seed at step 0\n',
    'Checkpoints match across levels and methods; Pantry has two seeds at steps 672 and 0\n')
(WS / 'main.tex').write_text(main)

readme = (WS / 'README.md').read_text()
readme = once(readme,
    "eleven complete MaxRL--ReplayMaxRL pair blocks cover all five domains at\nQwen2.5-0.5B and Falcon3-1B, plus Qwen2.5-3B Python. The four-arm\nintersections contain ten complete blocks and one four-seed Falcon Countdown",
    "twelve complete MaxRL--ReplayMaxRL pair blocks cover all five domains at\nQwen2.5-0.5B and Falcon3-1B, plus Qwen2.5-3B Graph and Python. The four-arm\nintersections contain eleven complete blocks and one four-seed Falcon Countdown")
start, end = readme.index('## September 9 results refresh'), readme.index('## Build')
readme = readme[:start] + '''## September 10 results refresh

Both manuscripts report 137/150 admitted E118 endpoints, 85/100 E119
endpoints, and 41/45 E120-R1 endpoints. The appendix gives every planned
model/domain comparison with exact paired seed counts: 12 complete E118 pair
blocks, four complete Level-2 factorials, and seven complete E120 blocks.
Figure 6 uses 22 matched domain–seed cells: five terminal seeds each for
Graph, Countdown, Python, and MathIR; PantryPlan seed 43 at step 672 and
seed 45 at step 0, matched across both levels and all four methods.

The newly completed Qwen2.5-3B Graph MaxRL comparison adds .163 extra
correct modes (unadjusted 95% paired interval [.021, .305]); its correctness
and raw-support intervals include zero. Level-2 PantryPlan has its first
ReplayDr.GRPO pair (seed 46), reported descriptively without an interval.
The completed Falcon PantryPlan weighting comparison is inconclusive:
uniform minus frequency replay gives −.146 extra modes, with a paired
bootstrap interval [−.564, +.414]. Partial blocks retain their observed
seed counts without five-seed intervals, and the frozen Qwen-0.5B primary
weighting analysis is preserved. The dated result digest, CSV, JSON, and
forest plot accompany the source bundle. Earlier dated sections below record
historical revisions.

''' + readme[end:]
readme = once(readme,
    'All eleven complete MaxRL replay comparisons retain their registered seed sets.',
    'All twelve complete MaxRL replay comparisons retain their registered seed sets.')
(WS / 'README.md').write_text(readme)
(WS / 'Makefile').write_text((WS / 'Makefile').read_text().replace('20260909', '20260910'))

snapshot = json.loads((WS / 'snapshot.json').read_text())
assets = sorted(set(
    [str(p.relative_to(PAPER)) for p in (PAPER / 'results').glob('*20260910*')]
    + [str(p.relative_to(PAPER)) for p in (PAPER / 'figures').glob('current_campaign_results_20260910.*')]
    + [f'figures/{name}.{ext}' for name in ('e118_all_scale_factorial_progress', 'e118_scale_extensions_appendix', 'modebench_level_admission') for ext in ('pdf', 'png', 'json')]
    + ['results/modebench_level_comparison_snapshot.json']
))
receipts = []
for rel in assets:
    src, dst = PAPER / rel, WS / rel
    before = sha(dst) if dst.exists() else None
    if rel in snapshot['copied_files']:
        assert before == snapshot['copied_files'][rel], rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    after = sha(dst)
    assert after == sha(src), rel
    snapshot['copied_files'][rel] = after
    receipts.append({'path': rel, 'before_sha256': before, 'after_sha256': after})

snapshot['source_sha256'] = sha(parent_after)
snapshot['snapshot_date'] = '2026-09-10'
snapshot['workshop_sources'] = {p: sha(WS / p) for p in workshop_before}
snapshot['correction_history'].append({
    'date': '2026-09-10',
    'reason': 'Integrate the audited September 10 endpoint census, the complete Qwen2.5-3B Graph MaxRL comparison, the first descriptive Level-2 PantryPlan replay pair, and the inconclusive complete Falcon PantryPlan weighting block. Synchronize Figures 5/6 and the 22 matched Figure 6 cells. Preserve the distinct four-page workshop narrative, frozen primary results, and partial-block uncertainty boundaries; correct the workshop expanded-results stale Countdown completion sentence.',
    'audit': '../audits/results_refresh_20260910/manuscript_revision/workshop_sync.json',
    'parent_source_before': sha(parent_before),
    'parent_source_after': sha(parent_after),
    'workshop_sources_before': workshop_before,
    'workshop_sources_after': snapshot['workshop_sources'],
    'assets': receipts,
    'coverage': {
        'e118': {'terminal': 137, 'registered': 150, 'complete_blocks': 12, 'registered_blocks': 15, 'paired_seeds': 66},
        'e119': {'terminal': 85, 'registered': 100, 'complete_blocks': 4, 'registered_blocks': 5},
        'e120': {'terminal': 41, 'registered': 45, 'complete_blocks': 7, 'registered_blocks': 9},
    },
    'figure6_matched_cells': 22,
    'analysis_scope': 'Exact step-3072 paired endpoints only; partial blocks have no five-seed intervals. Figure 6 alone uses matched interim checkpoints (Pantry 43:672 and 45:0). Weighting contrasts are uniform minus fresh-frequency; frozen Qwen-0.5B primary evidence preserved.'
})
(WS / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
(AUDIT / 'workshop_sync.json').write_text(json.dumps({
    'parent_before_sha256': sha(parent_before), 'parent_after_sha256': sha(parent_after),
    'workshop_sources_before': workshop_before, 'workshop_sources_after': snapshot['workshop_sources'],
    'exact_parent_delta_chunks_applied': applied,
    'parent_main_chunks_kept_workshop_distinct': workshop_distinct,
    'assets': receipts,
}, indent=2) + '\n')
print(json.dumps({'exact_chunks': len(applied), 'workshop_distinct_parent_chunks': len(workshop_distinct), 'assets_synced': len(assets), 'workshop_sources': snapshot['workshop_sources']}, indent=2))
