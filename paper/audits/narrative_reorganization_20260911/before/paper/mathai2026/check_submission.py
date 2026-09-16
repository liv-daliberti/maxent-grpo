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


def prompt_ablation_figures(manifest):
    """Authenticate the copied publication scope without opening repository inputs.

    The parent build reconstructs the science before synchronization. This check
    deliberately follows only snapshot hashes and copied figure metadata so the
    resulting source ZIP can build independently of the experiment archive.
    """
    stem = 'modebench_prompt_ablation_20260911'
    report_name, tex_name = 'results/' + stem + '.json', 'results/' + stem + '.tex'
    families = ('frontier', 'local')
    assets = {name for family in families for extension in ('pdf', 'png', 'json')
              for name in [f'figures/modebench_prompt_ablation_{family}.{extension}']}
    if not any((ROOT / name).exists() for name in assets | {report_name, tex_name}):
        return []
    bindings = manifest.get('copied_files', {})

    def copied(name):
        require(name in bindings, 'ablation asset absent from snapshot: ' + name)
        require(digest(ROOT / name) == bindings[name], 'ablation snapshot changed: ' + name)
        return ROOT / name

    report_path, tex_path = copied(report_name), copied(tex_name)
    report = json.loads(report_path.read_text())
    require(report.get('schema') == 'modebench-prompt-ablation-analysis-v1'
            and report.get('status') == 'complete', 'ablation analysis is not complete')
    inventory = report.get('inventory', {})
    finalized = inventory.get('finalized_draws', inventory.get('received_draws'))
    require(inventory.get('status') == 'complete' and inventory.get('runs')
            and all(run.get('complete') is True for run in inventory['runs'])
            and inventory.get('expected_draws') == finalized
            and inventory.get('expected_draws', 0) > 0, 'ablation sampling is incomplete')
    registries = report.get('registries', {})
    require(registries and set(registries) <= {'hosted', 'local'},
            'unknown or missing ablation panel registry')
    included = [family for family, key in [('frontier', 'hosted'), ('local', 'local')]
                if key in registries]
    missing = [family for family in families if family not in included]
    scope = report.get('scope', {})
    require(report.get('experiment_status') == ('partial_panels' if missing else 'complete')
            and scope.get('included_panels') == included
            and scope.get('registered_panels') == list(families)
            and scope.get('omitted_panels') == missing,
            'ablation scope must distinguish included panels from the full experiment')
    require({run.get('family') for run in inventory['runs']} == set(included)
            and {model.get('family') for model in report.get('models', [])} == set(included),
            'ablation model or sampling inventory differs from its scope')
    if missing:
        prose = tex_path.read_text().lower()
        require(('partial' in prose or 'incomplete' in prose)
                and all(family in prose or (family == 'frontier' and 'hosted' in prose)
                        for family in missing), 'partial ablation appendix omits its scope disclosure')
    for family in families:
        figure = f'figures/modebench_prompt_ablation_{family}'
        if family not in included:
            require(not any((ROOT / (figure + '.' + suffix)).exists()
                            for suffix in ('pdf', 'png', 'json')),
                    'omitted ablation panel has a published figure: ' + family)
            continue
        paths = {suffix: copied(figure + '.' + suffix) for suffix in ('pdf', 'png', 'json')}
        metadata = json.loads(paths['json'].read_text())
        require(metadata.get('schema') == 'prompt-ablation-figure-v1'
                and metadata.get('family') == family
                and metadata.get('report_sha256') == digest(report_path),
                'ablation figure does not bind the copied report: ' + family)
        require(set(metadata.get('outputs', {})) == {'pdf', 'png'},
                'ablation figure renderings are incomplete: ' + family)
        for suffix in ('pdf', 'png'):
            require(metadata['outputs'][suffix].get('sha256') == digest(paths[suffix]),
                    'ablation figure rendering changed: ' + family + '.' + suffix)
    return ['fig:prompt-hints-' + family for family in included]



def discovery_figures(manifest):
    """Validate the separate 64-draw publication using only this bundle."""
    stem = 'results/modebench_discovery_curves_20260911'
    figures = {key: name for family in ('frontier', 'local') for key, name in (
        (family, 'figures/modebench_discovery_curves_' + family),
        ('correct_budget_' + family, 'figures/modebench_discovery_correct_budget_' + family))}
    names = {stem + suffix for suffix in ('.json', '.tex')}
    names.update(name + '.' + suffix for name in figures.values() for suffix in ('pdf', 'png', 'json'))
    if not any((ROOT / name).exists() for name in names):
        return []
    bindings = manifest.get('copied_files', {})

    def copied(name):
        require(name in bindings, 'discovery asset absent from snapshot: ' + name)
        require(digest(ROOT / name) == bindings[name], 'discovery snapshot changed: ' + name)
        return ROOT / name

    report_path, tex_path = copied(stem + '.json'), copied(stem + '.tex')
    report = json.loads(report_path.read_text())
    require(report.get('schema') == 'modebench-discovery-curves-analysis-v1'
            and report.get('status') == 'complete', 'discovery analysis is not complete')
    inventory = report.get('inventory', {})
    require(inventory.get('status') == 'complete' and inventory.get('runs')
            and all(run.get('complete') is True for run in inventory['runs'])
            and inventory.get('expected_draws', 0) > 0
            and inventory['expected_draws'] == inventory.get('finalized_draws'),
            'discovery sampling is incomplete')
    registries = report.get('registries', {})
    require(registries and set(registries) <= {'hosted', 'local'}, 'unknown discovery panel registry')
    included = [family for family, key in [('frontier', 'hosted'), ('local', 'local')] if key in registries]
    omitted = [family for family in ('frontier', 'local') if family not in included]
    scope = report.get('scope', {})
    require(report.get('experiment_status') == ('partial_panels' if omitted else 'complete')
            and scope.get('included_panels') == included
            and scope.get('omitted_panels') == omitted
            and scope.get('registered_panels') == ['frontier', 'local'],
            'discovery scope does not distinguish partial panels from the full experiment')
    require({run.get('family') for run in inventory['runs']} == set(included)
            and {model.get('family') for model in report.get('models', [])} == set(included),
            'discovery model or sample inventory differs from scope')
    if omitted:
        prose = tex_path.read_text().lower()
        require(('partial' in prose or 'incomplete' in prose)
                and all(family in prose or (family == 'frontier' and 'hosted' in prose) for family in omitted),
                'partial discovery appendix omits its scope disclosure')
    expected = {key for family in included for key in (family, 'correct_budget_' + family)}
    for key, name in figures.items():
        if key not in expected:
            require(not any((ROOT / (name + '.' + suffix)).exists() for suffix in ('pdf', 'png', 'json')),
                    'omitted discovery panel has a published figure: ' + key)
            continue
        paths = {suffix: copied(name + '.' + suffix) for suffix in ('pdf', 'png', 'json')}
        metadata = json.loads(paths['json'].read_text())
        require(metadata.get('schema') == 'modebench-discovery-figure-v1'
                and metadata.get('family') == key and metadata.get('report_sha256') == digest(report_path),
                'discovery figure does not bind the copied report: ' + key)
        require(set(metadata.get('outputs', {})) == {'pdf', 'png'},
                'discovery figure renderings are incomplete: ' + key)
        for suffix in ('pdf', 'png'):
            require(metadata['outputs'][suffix].get('sha256') == digest(paths[suffix]),
                    'discovery figure rendering changed: ' + key + '.' + suffix)
    return [label for family in included for label in (
        'fig:discovery-curves-' + family, 'fig:discovery-correct-budget-' + family)]


def check_output(manifest=None):
    if manifest is None:
        snapshot_path = ROOT / 'snapshot.json'
        if snapshot_path.is_file():
            manifest = json.loads(snapshot_path.read_text())
        else:
            # The existing Overleaf exporter calls this function after extracting
            # its independently hashed package, which intentionally omits Python.
            package_path = ROOT / 'PACKAGE_MANIFEST.json'
            require(package_path.is_file(), 'missing standalone source bindings')
            package = json.loads(package_path.read_text())
            require(package.get('schema') == 'mathai2026-overleaf-package-v1',
                    'unsupported standalone package bindings')
            manifest = {'copied_files': package['file_sha256']}
    aux = (ROOT / 'main.aux').read_text()
    labels = dict((name, int(page)) for name, page in
                  re.findall(r'\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}', aux))
    require(labels.get('main:lastpage') == 4, 'main material must end on page 4')
    require(1 <= labels.get('sec:conclusion', 999) <= 4,
            'conclusion must remain in the main text')
    require(labels.get('fig:story') == 1, 'Figure 1 must be on page 1')
    require(labels.get('tab:frontier-motivation', 5) > 4,
            'dense hosted motivation table must not appear in the main text')
    main_figures = ['fig:story', 'fig:modebench-examples', 'fig:verified-support-story',
                    'fig:cross-scale-terminal-effects', 'fig:maxrl-factorial',
                    'fig:level2-admission', 'fig:hosted-verified-breadth']
    figure_numbers = {name: int(number) for name, number in
                      re.findall(r'\\newlabel\{(fig:[^}]+)\}\{\{(\d+)\}', aux)}
    for number, name in enumerate(main_figures, 1):
        require(1 <= labels.get(name, 999) <= 4, 'main figure outside page limit: ' + name)
        require(figure_numbers.get(name) == number, 'main figure order changed: ' + name)
    ablation_labels = prompt_ablation_figures(manifest)
    discovery_labels = discovery_figures(manifest)
    supplementary_figures = 11 + len(ablation_labels) + len(discovery_labels)
    require(len([k for k in labels if k.startswith('fig:')]) == 7 + supplementary_figures,
            f'expected seven main and {supplementary_figures} supplementary figures')
    require({name for name in labels if name.startswith('fig:prompt-hints-')} == set(ablation_labels),
            'compiled ablation figures differ from the authenticated panel scope')
    require({name for name in labels if name.startswith('fig:discovery-')} == set(discovery_labels),
            'compiled discovery figures differ from the authenticated panel scope')
    for name in discovery_labels:
        require(labels.get(name, 0) > 4, 'discovery figure must remain in the supplement: ' + name)
    for name in ablation_labels:
        require(labels.get(name, 0) > 4, 'ablation figure must remain in the supplement: ' + name)
    require(labels.get('fig:gpt56-temperature-curve', 0) > 4,
            'GPT temperature curve must remain in the workshop supplement')
    require(labels.get('fig:hosted-graph-comparison', 0) > 4,
            'hosted comparison must remain in the supplement')
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
    return supplementary_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-build', action='store_true',
                        help='record input/output hashes after a successful fresh compilation')
    parser.add_argument('--check-ablation-assets', action='store_true',
                        help='validate copied ablation scope and hashes without repository inputs or a PDF')
    parser.add_argument('--check-discovery-assets', action='store_true',
                        help='validate copied discovery scope and hashes without repository inputs or a PDF')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'snapshot.json').read_text())
    if args.check_ablation_assets:
        count = len(prompt_ablation_figures(manifest))
        print(f'PASS: {count} authenticated ablation figures; standalone copied assets intact.')
        return
    if args.check_discovery_assets:
        count = len(discovery_figures(manifest))
        print(f'PASS: {count} authenticated discovery figures; standalone copied assets intact.')
        return
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
    supplementary_figures = check_output(manifest)
    if args.record_build:
        BUILD_RECEIPT.write_text(json.dumps({
            'schema': 'mathai2026-compiled-submission-v1',
            'source_sha256': sources,
            'artifact_sha256': artifacts,
        }, indent=2, sort_keys=True) + '\n')
    print(f'PASS: 4 content pages; all 7 main figures; {supplementary_figures} supplementary figures;')
    print('      Figure 1 is on page 1; references start on page 5;')
    print('      MATH-AI submission footer; anonymous mode and official style intact;')
    print('      referenced inputs and source hashes complete; compiled artifact current;')
    print('      snapshot assets intact; no unresolved references or overfull boxes.')


if __name__ == '__main__':
    main()
