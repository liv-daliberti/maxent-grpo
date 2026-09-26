#!/usr/bin/env python3
"""Export and independently compile a self-contained Overleaf project for the ICLR paper.

The file list is derived from a real pdfLaTeX run with -recorder, so it always
matches what the current main.tex actually reads rather than a hand-kept list.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / 'paper'

# Needed by Overleaf's BibTeX pass, and provenance sidecars we want to travel
# with the figures; pdfLaTeX never reads these, so -recorder cannot find them.
EXTRA_FILES = ['example_paper.bib', 'iclr2027_conference.bst']

# Outputs of the build, never inputs to it.
DROP_SUFFIXES = {'.aux', '.out', '.log', '.fls', '.fdb_latexmk', '.blg', '.synctex.gz'}

README = '''# ICLR submission — Overleaf project

Measuring and Mitigating Solution Mode Collapse in RLVR

This is the full ICLR manuscript, including the appendix. The MATH-AI workshop
version is packaged separately by
`ops/package_mathai2026_overleaf.py`.

## Upload and compile

1. In Overleaf, choose **New Project -> Upload Project** and upload this whole ZIP.
2. Set the **Main document** to `main.tex` and the **Compiler** to **pdfLaTeX**.
3. Click **Recompile**. The bibliography is built automatically with BibTeX.

Everything needed to compile is included: the manuscript and its full appendix,
every figure PDF, every generated results table fragment, the provider icons,
the bibliography and its style, and the local LaTeX style files. The project
needs only standard TeX Live packages. It requires no repository checkout, no
Python, no data download, no figure regeneration, and no shell escape.

`reference/compiled-main.pdf` is the PDF produced from these exact sources in
the repository; compare against it if your TeX Live version shifts line
wrapping. It was checked with pdfLaTeX (TeX Live 2020) and BibTeX. Select
TeX Live 2020 in Overleaf project settings to match the tested version.

`main.bbl` is included, so the bibliography renders on the very first
compile even before BibTeX has run.

## Where to edit

- `main.tex`: the manuscript through the bibliography.
- `appendix.tex`: the supplement, pulled in by `main.tex`.
- `example_paper.bib`: bibliography entries.
- `figures/`: figure PDFs, each alongside the JSON provenance record it was
  generated with. LaTeX reads only the PDFs.
- `results/`: generated table fragments pulled in by `\\input`. These are
  generated copies in the repository - edit them there and re-export rather
  than by hand here.
- `icons/`: provider logos used in the hosted-model figures and tables.
- `iclr2027_conference.sty`, `iclr2027_conference.bst`: the official style,
  unchanged.
- `fancyhdr.sty`, `natbib.sty`, `wrapfig.sty`: local copies of common packages,
  included so the layout matches the locally tested build exactly.
- `latexmkrc`: pdfLaTeX build defaults; no custom build scripts are needed.
- `PACKAGE_MANIFEST.json`: every packaged file with its SHA-256.

Research-code paths printed in the supplement identify the original
experiments; they are not files required to compile this paper.

Upload instructions: https://www.overleaf.com/learn/latex/Kb/Uploading_a_project
'''

LATEXMKRC = """$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;
@default_files = ('main.tex');
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")


def discover_inputs(workdir):
    """Compile once with -recorder and return paper-relative inputs it read."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # Seed the retained aux/bbl so cross-references and citations resolve in one pass.
    for aux in ('main.aux', 'main.bbl'):
        if (PAPER / aux).exists():
            shutil.copy2(PAPER / aux, workdir / aux)
    subprocess.run(['pdflatex', '-recorder', '-interaction=nonstopmode',
                    f'-output-directory={workdir}', 'main.tex'],
                   cwd=PAPER, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, text=True, errors="replace")
    fls = (workdir / 'main.fls').read_text()
    found = set()
    for line in fls.splitlines():
        if not line.startswith('INPUT '):
            continue
        name = line[len('INPUT '):].strip()
        if name.startswith('/'):
            continue  # TeX Live's own files, and the output directory
        name = name[2:] if name.startswith('./') else name
        if Path(name).suffix in DROP_SUFFIXES:
            continue
        found.add(name)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=PAPER / 'iclr2027_overleaf.zip',
                        help='destination ZIP path')
    parser.add_argument('--skip-verify', action='store_true',
                        help='do not compile the exported project')
    args = parser.parse_args()

    if not (PAPER / 'main.bbl').exists():
        sys.exit('main.bbl is missing; run `make` in paper/ first.')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inputs = discover_inputs(tmp / 'fls')
        print(f'pdfLaTeX read {len(inputs)} files from paper/')

        files = list(inputs) + ['main.bbl'] + EXTRA_FILES
        # Carry each figure's provenance record next to the figure it describes.
        for name in inputs:
            sidecar = Path(name).with_suffix('.json')
            if name.startswith('figures/') and (PAPER / sidecar).exists():
                files.append(str(sidecar))
        files = sorted(set(files))

        missing = [f for f in files if not (PAPER / f).exists()]
        if missing:
            sys.exit('missing sources: ' + ', '.join(missing))

        stage = tmp / 'project'
        for name in files:
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PAPER / name, dest)
        (stage / 'reference').mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / 'main.pdf', stage / 'reference/compiled-main.pdf')
        (stage / 'README.md').write_text(README)
        (stage / 'latexmkrc').write_text(LATEXMKRC)

        manifest = {
            'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'generator': 'ops/package_iclr_overleaf.py',
            'main_document': 'main.tex',
            'compiler': 'pdflatex',
            'source_commit': run(['git', 'rev-parse', 'HEAD'], cwd=REPO).stdout.strip(),
            'files': {f: sha256(PAPER / f) for f in files},
        }
        (stage / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')

        if not args.skip_verify:
            check = tmp / 'verify'
            shutil.copytree(stage, check)
            for step in (['pdflatex', '-interaction=nonstopmode', 'main'],
                         ['bibtex', 'main'],
                         ['pdflatex', '-interaction=nonstopmode', 'main'],
                         ['pdflatex', '-interaction=nonstopmode', 'main']):
                subprocess.run(step, cwd=check, check=(step[0] != 'bibtex'),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace")
            log = (check / 'main.log').read_text(errors='replace')
            bad = [l for l in log.splitlines()
                   if 'not found' in l or l.startswith('! ')]
            if bad:
                sys.exit('exported project did not compile cleanly:\n' + '\n'.join(bad[:20]))
            built = check / 'main.pdf'
            if not built.exists():
                sys.exit('exported project produced no PDF')
            pages = [l for l in log.splitlines() if 'Output written' in l]
            print('standalone compile OK:', pages[0].strip() if pages else 'no page line')

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, 'w', ZIP_DEFLATED) as z:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    z.write(path, path.relative_to(stage).as_posix())

    size = args.output.stat().st_size / 1e6
    print(f'wrote {args.output} ({size:.1f} MB, {len(files) + 4} files)')


if __name__ == '__main__':
    main()
