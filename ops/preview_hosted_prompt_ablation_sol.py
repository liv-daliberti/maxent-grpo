#!/usr/bin/env python3
"""Explicitly scoped Sol-only preview using unchanged frozen analysis primitives."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
SOURCE = BASE / 'analysis_code_editorial_v2/ops/analyze_modebench_prompt_ablation.py'
SOURCE_SHA = '2be926e99e98446512f4a14077a2e175d4b5bde0fada2933323f1bf7eaebb891'
OUT = BASE / 'sol_hosted_preview_20260911'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(value):
    return 'undefined' if value is None else f'{value:.4f}'


def interval(record):
    ci = record['ci95']
    return 'undefined' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]'


def main():
    assert sha(SOURCE) == SOURCE_SHA
    registry = json.loads((BASE / 'hosted_analysis_runs.json').read_text())
    entries = [e for e in registry['runs'] if e['model_id'] == 'gpt56sol']
    assert len(entries) == 2 and {e['arm'] for e in entries} == {'original', 'neutral'}
    while not all((Path(e['run_dir']) / 'prompt_ablation_grading_audit.json').exists() for e in entries):
        time.sleep(45)
    assert sha(SOURCE) == SOURCE_SHA
    spec = importlib.util.spec_from_file_location('frozen_sol_preview_analysis', SOURCE)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    design = analyzer.authenticate_design(BASE)
    runs = {e['arm']: analyzer.authenticate_hosted(design, e) for e in entries}
    analyzer.validate_payload_pair(runs['original'], runs['neutral'])
    assert all(len(run['raw']) == 1536 for run in runs.values())
    rng = analyzer.np.random.default_rng(analyzer.SEED)
    indices = {(l, d): rng.integers(0, analyzer.PROMPTS_PER_CELL,
                                  (20000, analyzer.PROMPTS_PER_CELL))
               for l in analyzer.LEVELS for d in analyzer.DOMAINS}
    result = analyzer.analyze_pair(runs['original'], runs['neutral'], indices)
    report = {
        'schema': 'modebench-prompt-ablation-sol-only-preview-v1',
        'status': 'complete_sol_cohort_only',
        'experiment_status': 'other_registered_hosted_models_still_collecting',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'included_model_ids': ['gpt56sol'],
        'excluded_registered_hosted_model_ids': ['gpt54', 'grok43'],
        'responses': 3072, 'prompts_per_domain_level': 32, 'draws_per_prompt_arm': 8,
        'bootstrap_replicates': 20000, 'bootstrap_seed': analyzer.SEED,
        'contrast': analyzer.CONTRAST,
        'resampling': 'Whole paired prompts, separately stratified by domain and level; identical frozen index arrays for arms, metrics and gradings.',
        'interval_scope': 'Pointwise percentile 95% intervals, without multiplicity adjustment. Exploratory; degenerate intervals do not establish equivalence.',
        'collision': 'Total colliding correct pairs divided by total correct pairs within a cell. Correctness-conditioned populations can differ between arms.',
        'scope_limit': 'Complete Sol cohort only. This preview does not claim the full registered hosted panel is complete and does not replace the official full report.',
        'preview_source': analyzer.binding(Path(__file__)),
        'frozen_analyzer_source': analyzer.binding(SOURCE),
        'frozen_audit_source': analyzer.binding(SOURCE.parent / 'audit_hosted_modebench_completion.py'),
        'root_manifest': analyzer.binding(BASE / 'manifest.json'),
        'hosted_registry': analyzer.binding(BASE / 'hosted_analysis_runs.json'),
        'model': result,
    }
    OUT.mkdir(exist_ok=True)
    report_path = OUT / 'sol_only_paired_summary.json'
    assert not report_path.exists()
    analyzer.write_json(report_path, report)
    text = [
        'GPT-5.6 Sol prompt-hint ablation: complete Sol cohort only', '',
        'This provisional summary uses all 3,072 responses from the two completed Sol arms: 32 problems per domain/level and eight draws per prompt/arm. GPT-5.4 and Grok remain outside this preview while collection continues. It does not replace the official full-panel report.', '',
        'Primary strict grades are shown below. P8 is the fraction of prompts with at least one correct response; D8 is the mean number of distinct correct keys; B8 = D8 − P8. Collision is the pooled fraction of correct-response pairs with equal keys. Changes are neutral minus original. Intervals use the unchanged frozen analyzer with 20,000 paired-prompt bootstrap replicates, stratified by cell.', '',
        '| Cell | Metric | Original | Neutral | Change | Paired 95% CI for change |',
        '|---|---|---:|---:|---:|---|',
    ]
    labels = {'pass8': 'P8', 'distinct8': 'D8', 'b8': 'B8', 'correct_pair_collision': 'Conditional collision'}
    cells = result['analyses']['strict']['cells']
    for cell, values in cells.items():
        for metric, label in labels.items():
            effect = values[analyzer.CONTRAST][metric]
            text.append(f"| {cell} | {label} | {fmt(values['original'][metric]['estimate'])} | {fmt(values['neutral'][metric]['estimate'])} | {fmt(effect['estimate'])} | {interval(effect)} |")
    text += ['', 'Every arm estimate and its interval, paired discordance counts, joint collision eligibility, per-prompt statistics, and the unchanged normalization sensitivity are retained in the JSON. Conditional collision compares correctness-conditioned populations, which may differ between prompt arms. Intervals are pointwise and exploratory; an interval containing zero does not establish equivalence.', '',
             f"Source-bound JSON SHA256: `{sha(report_path)}`. Frozen analyzer SHA256: `{SOURCE_SHA}`."]
    md = OUT / 'sol_only_paired_summary.md'
    assert not md.exists()
    md.write_text('\n'.join(text) + '\n')
    analyzer.write_json(OUT / 'artifact_manifest.json', {
        'status': 'complete_sol_cohort_only',
        'files': {p.name: analyzer.binding(p) for p in (report_path, md, Path(__file__), SOURCE)},
    })
    print(json.dumps({'status': 'complete_sol_cohort_only', 'json_path': str(report_path),
                      'json_sha256': sha(report_path), 'md_path': str(md), 'md_sha256': sha(md)}), flush=True)


if __name__ == '__main__':
    main()
