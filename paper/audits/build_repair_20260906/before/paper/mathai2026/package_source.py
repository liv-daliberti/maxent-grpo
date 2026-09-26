#!/usr/bin/env python3
"""Create a standalone LaTeX source bundle."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parent
names = {'Makefile', 'README.md', 'snapshot.json', 'check_submission.py',
         'package_source.py', 'main.bbl'}
for pattern in ('*.tex', '*.sty', '*.bib', 'figures/*.pdf', 'figures/*.json', 'results/*.tex', 'results/*.json'):
    names.update(str(p.relative_to(root)) for p in root.glob(pattern))
output = root / 'mathai2026-source.zip'
with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
    for name in sorted(names):
        archive.write(root / name, name)
print(f'Created {output.name}: {len(names)} files, {output.stat().st_size:,} bytes')
