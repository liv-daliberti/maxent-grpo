#!/usr/bin/env python3
"""Reconstruct pass@8 and breadth from all 480 intact, source-bound prompt groups."""
from __future__ import annotations
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_gpt56_temperature_curve_expanded480 as expanded

SOURCE = expanded.OUTPUT.with_suffix('.json')
OUTPUT = SOURCE.with_name('GPT56_PASS8_FRONTIER_EXPANDED480')
require = expanded.require

def authenticate(value, root=ROOT, cache=None):
    return expanded.old_pass8.authenticate(value, root, cache)

def analyze(source):
    require(source.get('schema') == expanded.CURVE_SCHEMA and source.get('status') == 'complete' and
            source.get('model') == expanded.legacy.MODEL and source.get('reasoning_effort') == 'none' and
            source.get('temperatures') == list(expanded.TEMPERATURES), 'Require the complete expanded-480 curve')
    required = ('all_native_receipts_authenticated', 'same_480_prompts_all_eight_slots',
                'only_temperature_varies_within_curve', 'all_returned_controls_match_requests',
                'source_summaries_reconstructed', 'expansion_plan_authenticated',
                'original_4800_measurements_retained_unchanged',
                'same_served_snapshot_and_native_non_temperature_controls', 'cohort_sensitivity_reported')
    require(all(source.get('validation', {}).get(k) is True for k in required),
            'Expanded complete-cohort audit is missing')
    require(source.get('prompts_per_condition') == 480 and source.get('responses_per_condition') == 3840 and
            source.get('total_registered_responses') == 19200 and source.get('draws_per_prompt') == 8 and
            source['validation'].get('authenticated_native_receipt_count') == 19200,
            'Expanded prompt or response inventory is incomplete')
    require(source['bootstrap']['replicates'] == expanded.REPLICATES and
            source['bootstrap']['seed'] == expanded.SEED and source['bootstrap']['individual_draw_pairing'] is False,
            'Require 20,000 paired stratified whole-prompt bootstrap replicates')
    plan = expanded.authenticate_plan(source['approved_expansion_plan']['path'])
    require(expanded.file_sha(plan['path']) == source['approved_expansion_plan']['sha256'],
            'Expanded plan source binding differs')
    result = copy.deepcopy(source)
    original_sets = {}
    for cohort, size in expanded.COHORT_SIZES.items():
        block = source if cohort == 'combined_480' else source['cohort_sensitivity'][cohort]
        rows = plan['rows'] if cohort == 'combined_480' else plan['old_rows'] if cohort == 'original_120' else plan['new_rows']
        require(block['prompts_per_condition'] == size * 15 and block['responses_per_condition'] == size * 15 * 8,
                'Sensitivity cohort size differs from approved selection')
        require(set(block['analyses']) == set(expanded.GRADINGS), 'Missing or extra grading')
        for grading in expanded.GRADINGS:
            analysis = block['analyses'][grading]
            require(set(analysis['temperatures']) == {str(t) for t in expanded.TEMPERATURES},
                    'Incomplete expanded temperature grid')
            records = {float(t): arm['prompt_records'] for t, arm in analysis['temperatures'].items()}
            reconstructed = expanded.analyze_records(records, rows, size)
            require(reconstructed == analysis, 'Expanded empirical metrics or intervals differ from intact prompt groups')
            if cohort == 'combined_480':
                original_sets[grading] = {t: expanded.unique(rs, expanded.row_identity, 'combined prompt')
                                          for t, rs in records.items()}
            else:
                for t, rs in records.items():
                    require(all(r == original_sets[grading][t][expanded.row_identity(r)] for r in rs),
                            'Sensitivity prompt record differs from the combined cohort')
    result['schema'] = expanded.FRONTIER_SCHEMA
    result['validation']['pass8_computed_from_complete_prompt_groups'] = True
    result['pass8_definition'] = 'Mean indicator of at least one verified answer among all eight saved draws; all-zero groups remain in the denominator.'
    return result

def csv_text(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(('cohort', 'grading', 'scope', 'temperature', 'metric', 'estimate', 'ci95_low', 'ci95_high'))
    cohorts = {'combined_480': result['analyses'], **{k: result['cohort_sensitivity'][k]['analyses']
               for k in ('original_120', 'additional_360')}}
    for cohort, analyses in cohorts.items():
        for grading, analysis in analyses.items():
            for t, arm in analysis['temperatures'].items():
                group = arm['groups']['five_domain_macro']
                for scope, metrics in {'overall': group['overall'], **group['levels']}.items():
                    for metric in ('pass8', 'distinct8'):
                        item = metrics[metric]
                        writer.writerow((cohort, grading, scope, t, metric, item['estimate'], *item['ci95']))
    return stream.getvalue()

def build_report(source_path=SOURCE):
    source_path = Path(source_path)
    source = json.loads(source_path.read_text())
    count = authenticate(source)
    result = analyze(source)
    result['pass8_analysis'] = {'source': expanded.binding(source_path),
        'script': expanded.binding(__file__), 'authenticated_source_bindings': count,
        'api_calls': 0, 'regraded_outputs': 0, 'resampling_unit': 'Intact eight-draw prompt groups'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build_report(args.source)
    expanded.native.atomic(args.output.with_suffix('.json'), result)
    args.output.with_suffix('.md').write_text(expanded.markdown(result))
    args.output.with_suffix('.csv').write_text(csv_text(result))
    print(json.dumps({'status': 'complete', 'responses': 19200, 'output': str(args.output.with_suffix('.json'))}))

if __name__ == '__main__':
    main()
