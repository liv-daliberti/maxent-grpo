#!/usr/bin/env python3
"""Export and independently compile a self-contained Overleaf project."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / 'paper/mathai2026'
DEFAULT_AUDIT = REPO / 'paper/audits/narrative_reorganization_20260911/overleaf'
sys.path.insert(0, str(PAPER))
import check_submission as checks

README = '''# MATH-AI 2026 — Overleaf project

Mode Collapse in RLVR & ModeBench

## Upload and compile

1. In Overleaf, choose **New Project → Upload Project** and upload the whole ZIP.
2. Set the **Main document** to `main.tex` and the **Compiler** to **pdfLaTeX**.
3. Click **Recompile**. The bibliography is built automatically with BibTeX.

All manuscript sources, the full appendix, all figure PDFs, input tables,
bibliography, bibliography style, and local LaTeX style files are included.
The project needs only standard TeX Live packages. It requires no repository
checkout, Python scripts, data downloads, external figure generation, or shell
escape to compile.

The supplied `reference/compiled-main.pdf` is the reviewed PDF for comparison.
The expected layout is four main-content pages, Figure 1 on page 1, references
starting on page 5, and the complete supplement after the references.
Compilation was checked locally with pdfLaTeX (TeX Live 2020) and BibTeX via
latexmk. If a different TeX Live version changes line wrapping, compare against
the reference PDF; select TeX Live 2020 in project settings if available to
match the locally tested version.

## Where to edit

- `main.tex`: title, anonymous author placeholder, abstract, and four-page paper.
- `preamble.tex`: packages, formatting, macros, and workshop footer.
- `appendix.tex`: supplementary material and proofs.
- `example_paper.bib`: bibliography entries.
- `figures/`: figures and their existing JSON provenance.
- `results/`: input table fragments and existing result snapshots.
- `neurips_2026.sty`: the official workshop style, unchanged.
- `latexmkrc`: pdfLaTeX build defaults; no custom build scripts are needed.
- `PACKAGE_MANIFEST.json`: packaged-file hashes and original-source hashes.

This package remains in anonymous submission mode. For an accepted
camera-ready version, insert `\\def\\mathaicameraready{1}` before
`\\input{preamble}` in `main.tex`, replace the anonymous author placeholder,
and update `pdfauthor` in `preamble.tex`.

The document-class declaration is placed directly in `main.tex` so Overleaf
recognizes the entry point. This only relocates the declaration from the
preamble; the manuscript text, figures, results, and formatting are unchanged.
Research-code paths printed in the supplement identify the original experiments;
they are not files required to compile this paper.

Upload instructions: https://www.overleaf.com/learn/latex/Kb/Uploading_a_project
'''

def sha(content):
    return hashlib.sha256(content).hexdigest()

def command(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-directory', type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    AUDIT = args.audit_directory.resolve()
    if (AUDIT / 'validation.json').exists():
        raise ValueError('Completed export audit exists; choose a new --audit-directory')
    AUDIT.mkdir(parents=True,exist_ok=True)
    command([sys.executable,str(PAPER/'check_submission.py')])
    receipt=json.loads((PAPER/'build_receipt.json').read_text())
    snapshot=json.loads((PAPER/'snapshot.json').read_text())
    sources,references=checks.check_sources(snapshot)
    checks.require(sources==receipt['source_sha256'],'source changed before export')
    names=set(snapshot['copied_files']) | set(snapshot['workshop_sources'])
    names.update(p.name for p in PAPER.glob('*.sty'))
    checks.require(references<=names,'export omits a referenced input')
    payload={name:(PAPER/name).read_bytes() for name in sorted(names)}
    for name,content in payload.items():
        checks.require(sha(content)==sources[name],'export source changed: '+name)
    prefix=b'\\documentclass{article}\n\n'
    checks.require(payload['preamble.tex'].startswith(prefix),'unexpected document-class declaration')
    checks.require(payload['main.tex'].startswith(b'\\input{preamble}'),'unexpected main document entry point')
    payload['main.tex']=prefix+payload['main.tex']
    payload['preamble.tex']=payload['preamble.tex'][len(prefix):]
    bst=Path(subprocess.check_output(['kpsewhich','plainnat.bst'],text=True).strip())
    payload['plainnat.bst']=bst.read_bytes()
    payload['latexmkrc']=b"# Standard pdfLaTeX/BibTeX project; main.tex is the entry point.\n$pdf_mode = 1;\n$max_repeat = 5;\n@default_files = ('main.tex');\n"
    payload['README.md']=README.encode()
    payload['reference/compiled-main.pdf']=(PAPER/'main.pdf').read_bytes()
    checks.require(sha(payload['reference/compiled-main.pdf'])==receipt['artifact_sha256']['main.pdf'],'reference PDF changed')
    manifest={
        'schema':'mathai2026-overleaf-package-v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'main_document':'main.tex','compiler':'pdfLaTeX','bibliography':'BibTeX',
        'source_sha256':{name:sources[name] for name in sorted(names)},
        'packaging_adjustment':'Move documentclass from preamble.tex to the beginning of main.tex for main-document detection; no manuscript or figure changes.',
        'file_sha256':{name:sha(content) for name,content in sorted(payload.items())},
    }
    payload['PACKAGE_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    output=PAPER/'mathai2026-overleaf.zip'
    with tempfile.TemporaryDirectory(prefix='.mathai-overleaf-',dir=PAPER) as temporary:
        candidate=Path(temporary)/output.name
        with ZipFile(candidate,'w',ZIP_DEFLATED) as archive:
            for name,content in sorted(payload.items()):
                checks.require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe package path')
                archive.writestr(name,content)
        with tempfile.TemporaryDirectory(prefix='mathai-overleaf-check-',dir='/tmp') as extracted:
            stage=Path(extracted)
            with ZipFile(candidate) as archive:
                checks.require(archive.testzip() is None,'ZIP integrity check failed')
                archive.extractall(stage)
            for name,expected in manifest['file_sha256'].items():
                checks.require(sha((stage/name).read_bytes())==expected,'extracted file mismatch: '+name)
            env=os.environ.copy()
            for key in ('TEXINPUTS','BIBINPUTS','BSTINPUTS','TEXMFOUTPUT'):
                env.pop(key,None)
            build=['latexmk','-norc','-r','latexmkrc','-pdf','-interaction=nonstopmode',
                   '-halt-on-error','-file-line-error','-recorder','-latexoption=-no-shell-escape','main.tex']
            print('Compiling a fresh ZIP extraction outside the repository...',flush=True)
            with (AUDIT/'extracted-build.log').open('w') as log:
                command(build,cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT)
            original_root=checks.ROOT
            checks.ROOT=stage
            try:
                checks.require(checks.referenced_files()==references,'packaged dependency graph changed')
                checks.check_recorded_inputs(references)
                checks.check_output()
            finally:
                checks.ROOT=original_root
            actual=subprocess.check_output(['pdftotext','-layout',str(stage/'main.pdf'),'-'])
            expected=subprocess.check_output(['pdftotext','-layout',str(PAPER/'main.pdf'),'-'])
            checks.require(actual==expected,'standalone PDF text/layout differs from the reviewed PDF')
            outside=[]
            for line in (stage/'main.fls').read_text().splitlines():
                if not line.startswith('INPUT '):continue
                raw=Path(line[6:]);resolved=(stage/raw).resolve() if not raw.is_absolute() else raw.resolve()
                if resolved.is_relative_to(REPO):outside.append(str(resolved))
            checks.require(not outside,'compilation read files from the original repository')
            (AUDIT/'standalone-pages.txt').write_bytes(actual)
            (AUDIT/'standalone-main.pdf').write_bytes((stage/'main.pdf').read_bytes())
            report={
                'status':'passed','archive':str(output.relative_to(REPO)),
                'archive_sha256':sha(candidate.read_bytes()),'archive_bytes':candidate.stat().st_size,
                'file_count':len(payload),'figure_count':sum(n.startswith('figures/') and n.endswith('.pdf') for n in payload),
                'table_count':sum(n.startswith('results/') and n.endswith('.tex') for n in payload),
                'build_command':build,'pdf_text_and_layout_match':True,'original_repository_inputs':outside,
                'main_pages':4,'figure_1_page':1,'references_start_page':5,
                'total_pages':actual.count(b'\x0c'),
                'tex_version':subprocess.check_output(['pdflatex','--version'],text=True).splitlines()[0],
            }
        command([sys.executable,str(PAPER/'check_submission.py')])
        checks.require(json.loads((PAPER/'build_receipt.json').read_text())==receipt,'paper changed during export')
        candidate.replace(output)
    (AUDIT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (AUDIT/'README.md').write_text('Overleaf archive validated from a fresh extraction outside the repository. All manuscript pages match the current reviewed PDF in extracted text and layout. The compile used no original-repository input files and no shell escape. See validation.json and extracted-build.log.\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
