#!/usr/bin/env python3
"""Add the Levels 4 and 5 configs to the published ModeBench dataset.

Additive by construction: it uploads new parquet files under their own config
paths, appends the new entries to the dataset card's config list, and appends a
section to the card explaining what these levels do and do not carry. No file
belonging to the sealed Levels 1-3 release is rewritten, and no existing config
entry is modified.

The card edit is the part worth care, because the two levels do not share a
status. Level 5 cleared held-out confirmation in all five domains and is
admitted. Level 4 cleared four of five; its MathIR config missed the pass@1 gate
at 1.07 of tolerance and is not difficulty-matched. The card has to say both
where a reader will see them, rather than only in a manifest nobody opens, and it
must not flatten them into one caveat covering all ten configs.

``--plan`` prints exactly what would change and uploads nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'var/artifacts/modebench_levels45_release_20260915/package'
REPO_ID = 'od2961/ModeBench'
TOKEN_FILE = Path.home() / '.maxent_hf_token'
CAVEAT_HEADING = '## Levels 4 and 5: admission status'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def caveat_section(manifest: dict) -> str:
    """The card's admission section, written from the package's corrected labels.

    Levels 4 and 5 do not share a status, so the section cannot state one. Level 5
    is admitted; Level 4 cleared four domains of five. Reading both from the
    manifest keeps the card and the provenance file from drifting apart.
    """
    note = manifest['admission_status']
    five, four = note['level5'], note['level4']
    # The admission records key pantry as 'pantry'; the published config is
    # 'level4_pantry_plan'. Resolve against the manifest so the card can only
    # ever name configs that exist.
    by_domain = {(s['level'], s['domain']): s['config_name'] for s in manifest['splits']}
    def config(domain):
        for key in ((4, domain), (4, domain + '_plan')):
            if key in by_domain:
                return by_domain[key]
        raise SystemExit('no level4 config for domain: ' + domain)
    matched = ', '.join(f'`{config(d)}`' for d in four['difficulty_matched_domains'])
    unmatched = ', '.join(f'`{config(d)}`' for d in four['not_difficulty_matched_domains'])
    return '\n'.join([
        CAVEAT_HEADING, '',
        '**Level 5 is admitted.** All five domains cleared held-out confirmation against '
        '128 fresh rows each, replayed through the original graders. Level 5 carries the '
        'same warrant as Levels 1-3.', '',
        '**Level 4 cleared four of five domains and is not admitted.** '
        f'Difficulty-matched: {matched}. Not difficulty-matched: {unmatched}, which missed '
        'its pass@1 gate at 1.07 of tolerance.', '',
        'A result on Level 5, or on one of the four confirmed Level 4 domains, rests on the '
        'same warrant as one on Levels 1-3. A Level 4 MathIR result does not, and should not '
        'be reported as difficulty-matched.', '',
        four['mathir_note'], '',
        'Every config records its own status; see `provenance/levels45_manifest.json`.', '',
    ])


def commit_message(report: dict) -> str:
    note = report['manifest']['admission_status']
    unmatched = note['level4']['not_difficulty_matched_domains']
    return (f"Add Levels 4 and 5 ({report['new_configs']} configs, {report['rows_added']} rows): "
            'Level 5 admitted, Level 4 confirmed in four of five domains '
            f"(not difficulty-matched: {', '.join(unmatched)})")


def plan(api, token: str) -> dict:
    manifest = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    info = api.dataset_info(REPO_ID, token=token)
    card = info.card_data or {}
    existing = list(card.get('configs') or [])
    existing_names = {c['config_name'] for c in existing}
    additions = []
    for name, items in sorted(manifest['configs'].items()):
        require(name not in existing_names, f'config already published: {name}')
        additions.append({'config_name': name,
                          'data_files': [{'split': i['split'], 'path': i['path']} for i in items]})
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p.suffix == '.parquet')
    require(len(files) == manifest['split_count'],
            f"{len(files)} parquet files for {manifest['split_count']} splits")
    return {'existing_configs': len(existing), 'new_configs': len(additions),
            'resulting_configs': len(existing) + len(additions),
            'parquet_files': len(files), 'rows_added': manifest['row_count'],
            'bytes_added': sum(p.stat().st_size for p in files),
            'card_section': CAVEAT_HEADING, 'manifest': manifest,
            'additions': additions, 'files': files, 'existing': existing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    require(TOKEN_FILE.is_file(), f'missing credential {TOKEN_FILE}')
    token = TOKEN_FILE.read_text().strip()
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    report = plan(api, token)
    if args.plan:
        print(json.dumps({k: report[k] for k in
                          ('existing_configs', 'new_configs', 'resulting_configs',
                           'parquet_files', 'rows_added', 'bytes_added', 'card_section')},
                         indent=2))
        return
    operations = [CommitOperationAdd(path_in_repo=str(p.relative_to(PACKAGE)),
                                     path_or_fileobj=str(p)) for p in report['files']]
    operations.append(CommitOperationAdd(
        path_in_repo='LEVELS_4_5.md',
        path_or_fileobj=str(PACKAGE / 'README_levels45.md')))
    operations.append(CommitOperationAdd(
        path_in_repo='provenance/levels45_manifest.json',
        path_or_fileobj=str(PACKAGE / 'MANIFEST.json')))
    # Rewrite only the card: merge the new configs into its front matter and
    # append the caveat section to its body.
    from huggingface_hub import hf_hub_download
    readme = Path(hf_hub_download(REPO_ID, 'README.md', repo_type='dataset', token=token))
    text = readme.read_text()
    front = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.S)
    require(front is not None, 'dataset card has no YAML front matter')
    import yaml
    meta = yaml.safe_load(front.group(1)) or {}
    meta['configs'] = report['existing'] + report['additions']
    body = front.group(2)
    require(CAVEAT_HEADING not in body, 'caveat section already present')
    body = body.rstrip() + '\n\n' + caveat_section(report['manifest'])
    merged = '---\n' + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).rstrip() + '\n---\n' + body
    operations.append(CommitOperationAdd(path_in_repo='README.md',
                                         path_or_fileobj=merged.encode()))
    commit = api.create_commit(
        repo_id=REPO_ID, repo_type='dataset', operations=operations, token=token,
        commit_message=commit_message(report))
    print(json.dumps({'status': 'published', 'commit': commit.oid,
                      'configs_now': report['resulting_configs'],
                      'rows_added': report['rows_added']}, indent=2))


if __name__ == '__main__':
    main()
