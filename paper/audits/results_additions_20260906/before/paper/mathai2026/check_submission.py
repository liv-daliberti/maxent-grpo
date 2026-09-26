#!/usr/bin/env python3
"""Check the compiled workshop artifact without regenerating research results."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise SystemExit('FAIL: ' + message)

manifest = json.loads((ROOT / 'snapshot.json').read_text())
style_hash = hashlib.sha256((ROOT / 'neurips_2026.sty').read_bytes()).hexdigest()
require(style_hash == manifest['style_sha256'], 'official style has changed')
for name, expected in manifest['copied_files'].items():
    require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
            'frozen source asset changed: ' + name)
aux = (ROOT / 'main.aux').read_text()
labels = dict((name, int(page)) for name, page in
              re.findall(r'\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}', aux))
require(labels.get('main:lastpage') == 4, 'main text must end on page 4')
main_figures = ['fig:story', 'fig:modebench-examples', 'fig:verified-support-story',
                'fig:cross-scale-terminal-effects', 'fig:maxrl-factorial',
                'fig:level2-admission']
for name in main_figures:
    require(1 <= labels.get(name, 999) <= 4, 'main figure outside page limit: ' + name)
require(len([k for k in labels if k.startswith('fig:')]) == 13,
        'expected six main and seven supplementary figures')
log = (ROOT / 'main.log').read_text()
for problem in ['undefined', 'multiply defined', 'Overfull', 'LaTeX Error', 'Fatal error']:
    require(problem not in log, 'compiler diagnostic: ' + problem)
text = subprocess.check_output(['pdftotext', '-layout', str(ROOT / 'main.pdf'), '-'],
                               text=True)
pages = text.split('\f')
require(re.search(r'\bReferences\b', pages[4]) is not None,
        'references should begin on page 5')
require('Anonymous Author(s)' in pages[0], 'submission must use anonymous mode')
for i, page in enumerate(pages[:4], 1):
    require('??' not in page, f'unresolved reference on page {i}')
print('PASS: 4 content pages; all 6 main figures; 7 supplementary figures;')
print('      references start on page 5; anonymous template unchanged;')
print('      frozen assets intact; no unresolved references or overfull boxes.')
