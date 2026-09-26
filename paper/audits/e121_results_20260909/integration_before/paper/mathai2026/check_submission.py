#!/usr/bin/env python3
"""Check source completeness and the compiled workshop without rebuilding results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
BUILD_RECEIPT = ROOT / 'build_receipt.json'
BUILD_ARTIFACTS = ('main.pdf', 'main.aux', 'main.log', 'main.fls', 'main.bbl')


def require(condition, message):
    if not condition:
        raise SystemExit('FAIL: ' + message)


def digest(path):
    require(path.is_file(), 'missing required file: ' + str(path.relative_to(ROOT)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def referenced_files():
    """Follow literal local TeX inputs and require every referenced asset."""
    found = {'main.tex'}
    pending = ['main.tex']
    pattern = re.compile(
        r'\\(input|include|includegraphics|bibliography)\s*'
        r'(?:\[[^\]]*\]\s*)?\{([^{}]+)\}'
    )
    while pending:
        name = pending.pop()
        path = ROOT / name
        require(path.is_file(), 'missing referenced input: ' + name)
        source = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        for command, argument in pattern.findall(source):
            require('\\' not in argument, 'nonliteral source reference: ' + argument)
            names = argument.split(',') if command == 'bibliography' else [argument]
            for raw in names:
                target = Path(raw.strip())
                if not target.suffix:
                    suffix = '.pdf' if command == 'includegraphics' else (
                        '.bib' if command == 'bibliography' else '.tex')
                    target = target.with_suffix(suffix)
                require(not target.is_absolute() and '..' not in target.parts,
                        'source reference leaves standalone bundle: ' + str(target))
                relative = target.as_posix()
                require((ROOT / target).is_file(), 'missing referenced input: ' + relative)
                if relative not in found:
                    found.add(relative)
                    if command in ('input', 'include'):
                        pending.append(relative)
    return found


def check_sources(manifest):
    references = referenced_files()
    require(digest(ROOT / 'neurips_2026.sty') == manifest['style_sha256'],
            'official style has changed')
    workshop = manifest.get('workshop_sources', {})
    require(set(workshop) == {'main.tex', 'appendix.tex', 'preamble.tex'},
            'snapshot must bind main.tex, appendix.tex, and preamble.tex')
    bindings = dict(manifest['copied_files'])
    bindings.update(workshop)
    for name, expected in bindings.items():
        require(digest(ROOT / name) == expected, 'snapshot source changed: ' + name)
    require(references <= set(bindings),
            'referenced files absent from snapshot: ' + ', '.join(sorted(references - set(bindings))))
    names = set(bindings) | {'snapshot.json', 'Makefile', 'check_submission.py',
                              'build_submission.py', 'package_source.py'}
    names.update(path.name for path in ROOT.glob('*.sty'))
    return {name: digest(ROOT / name) for name in sorted(names)}, references


def check_recorded_inputs(references):
    recorder = ROOT / 'main.fls'
    require(recorder.is_file(), 'missing compiler recorder; rebuild with make')
    observed = set()
    for line in recorder.read_text().splitlines():
        if line.startswith('INPUT '):
            path = Path(line[6:])
            if not path.is_absolute():
                path = ROOT / path
            observed.add(path.resolve())
    missing = [name for name in sorted(references)
               if (ROOT / name).resolve() not in observed and not name.endswith('.bib')]
    require(not missing, 'referenced inputs absent from compiled artifact: ' + ', '.join(missing))


def check_output():
    aux = (ROOT / 'main.aux').read_text()
    labels = dict((name, int(page)) for name, page in
                  re.findall(r'\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}', aux))
    require(labels.get('main:lastpage') == 4, 'main text must end on page 4')
    require(labels.get('fig:story') == 1, 'Figure 1 must be on page 1')
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
    require(len(pages) > 4 and re.search(r'\bReferences\b', pages[4]) is not None,
            'references should begin on page 5')
    require('Anonymous Author(s)' in pages[0], 'submission must use anonymous mode')
    first_page = ' '.join(pages[0].split())
    require('Submitted to The 6th Workshop on Mathematical Reasoning and AI '
            '(NeurIPS 2026). Do not distribute.' in first_page,
            'missing MATH-AI workshop submission notice')
    require('Submitted to 40th Conference' not in first_page,
            'generic NeurIPS conference notice remains')
    for i, page in enumerate(pages[:4], 1):
        require('??' not in page, f'unresolved reference on page {i}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-build', action='store_true',
                        help='record input/output hashes after a successful fresh compilation')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'snapshot.json').read_text())
    sources, references = check_sources(manifest)
    artifacts = {name: digest(ROOT / name) for name in BUILD_ARTIFACTS}
    if args.record_build:
        pdf_time = (ROOT / 'main.pdf').stat().st_mtime_ns
        newer = [name for name in sources if (ROOT / name).stat().st_mtime_ns > pdf_time]
        require(not newer, 'PDF predates source changes; rebuild: ' + ', '.join(newer))
    else:
        require(BUILD_RECEIPT.is_file(), 'missing compilation receipt; rebuild with make')
        receipt = json.loads(BUILD_RECEIPT.read_text())
        require(receipt.get('schema') == 'mathai2026-compiled-submission-v1',
                'unsupported compilation receipt; rebuild with make')
        require(receipt.get('source_sha256') == sources,
                'compiled source is stale; rebuild with make')
        require(receipt.get('artifact_sha256') == artifacts,
                'compiled artifacts differ from the validated build; rebuild with make')
    check_recorded_inputs(references)
    check_output()
    if args.record_build:
        BUILD_RECEIPT.write_text(json.dumps({
            'schema': 'mathai2026-compiled-submission-v1',
            'source_sha256': sources,
            'artifact_sha256': artifacts,
        }, indent=2, sort_keys=True) + '\n')
    print('PASS: 4 content pages; all 6 main figures; 7 supplementary figures;')
    print('      Figure 1 is on page 1; references start on page 5;')
    print('      MATH-AI submission footer; anonymous mode and official style intact;')
    print('      referenced inputs and source hashes complete; compiled artifact current;')
    print('      snapshot assets intact; no unresolved references or overfull boxes.')


if __name__ == '__main__':
    main()
