#!/usr/bin/env python3
"""Measure what MathIR's certified support is made of, and how much of it is reachable.

A MathIR mode is a state trajectory, so two action programs that pass through
the same equations are the same mode and a program that detours through extra
equations is a different one. That makes the certified mode count a count of
derivations of one answer rather than of distinct answers, and it is the reason
the domain's breadth barely moves anywhere in the paper. This script separates
the support by route length and then asks, of every verified MathIR draw the
paper retains, which lengths any model has ever actually produced.

Enumeration runs the same interpreter that grades model outputs and takes
several minutes per level, so it is cached in the emitted record; the default
re-renders macros from that record and --recompute re-enumerates.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'paper/results/mathir_support_geometry.json'
MACROS = ROOT / 'paper/results/mathir_support_geometry_macros.tex'

# Frozen evaluation splits, at the levels every reported MathIR cell uses.
SPLITS = {
    'level1': 'var/data/mathir_action_menu_v1/eval',
    'level2': 'var/data/modebench_harder_v2_matched_r5/mathir/eval',
    'level3': 'var/data/modebench_level3_matched_v3/mathir/eval',
}
# Retained draw evidence the manuscript already binds. Each entry is a label and
# a glob of response records carrying a canonical key per graded draw.
DRAWS = {
    'hosted': ('GPT-5.6 Sol, Levels 1-3, eight draws',
               ['artifacts/frontier_modebench_gpt56sol_20260911/samples.jsonl']),
    'local_eight': ('local checkpoints, Levels 2-3, eight draws',
                    ['artifacts/modebench_prompt_ablation_20260911/local/results_v2/'
                     '*mathir*/responses.jsonl']),
    'local_sixtyfour': ('local checkpoints, Levels 2-3, 64 draws',
                        ['artifacts/modebench_discovery_curves_20260911/local/results/'
                         '*mathir*/responses.jsonl']),
}
# add/sub and mul/div undo each other on the same argument; a route carrying
# such a pair reaches its answer by a step it then takes back.
INVERSE = {('add', 'sub'), ('sub', 'add'), ('mul', 'div'), ('div', 'mul')}
DETOUR_SAMPLE = 20


def route_length(canonical_key: str) -> int:
    """States on the trajectory this key records, which is the route's length."""
    return canonical_key.split(':', 2)[2].count('>') + 1


def _has_inverse_pair(programs: list[str]) -> bool:
    parts = []
    for program in programs:
        head, _, rest = re.sub(r'\s+', '', program).partition('(')
        parts.append((head, rest[:-1]))
    return any(
        (parts[i][0], parts[j][0]) in INVERSE and parts[i][1] == parts[j][1]
        for i in range(len(parts)) for j in range(i + 1, len(parts)))


def enumerate_support() -> dict:
    """Route-length structure of every certified mode, per level."""
    sys.path.insert(0, str(ROOT / 'src'))
    from datasets import load_from_disk
    from oat_drgrpo.mathir import enumerate_mathir_action_menu_validations

    levels: dict[str, dict] = {}
    detour_checked = detour_undone = 0
    for level, relative in SPLITS.items():
        rows = load_from_disk(str(ROOT / relative))['multi_answer']
        lengths: collections.Counter = collections.Counter()
        modes: collections.Counter = collections.Counter()
        shortest: collections.Counter = collections.Counter()
        terminals_shared = 0
        per_row = []
        for index, row in enumerate(rows):
            spec = json.loads(row['answer'])
            found = enumerate_mathir_action_menu_validations(spec)
            row_lengths = sorted(len(v.action_ids) for v in found)
            modes[len(found)] += 1
            lengths.update(row_lengths)
            shortest[sum(1 for L in row_lengths if L == min(row_lengths))] += 1
            # Every mode ends at the same equation: the modes are derivations of
            # one answer, which is the whole point of the measurement.
            terminals_shared += len({v.canonical_key.rsplit('>', 1)[-1] for v in found}) == 1
            per_row.append({'row_index': index, 'lengths': row_lengths,
                            'keys': [v.canonical_key for v in found]})
            if level == 'level1' and index < DETOUR_SAMPLE:
                for validation in found:
                    if len(validation.action_ids) < 4:
                        continue
                    detour_checked += 1
                    detour_undone += _has_inverse_pair(
                        [spec['actions'][a] for a in validation.action_ids])
        levels[level] = {
            'split': relative,
            'prompts': len(rows),
            'modes_per_prompt': dict(sorted(modes.items())),
            'length_histogram': dict(sorted(lengths.items())),
            'shortest_routes_per_prompt': dict(sorted(shortest.items())),
            'prompts_with_one_terminal_state': terminals_shared,
            'per_row': per_row,
        }
    return {'levels': levels,
            'detour_audit': {'level': 'level1', 'prompts_sampled': DETOUR_SAMPLE,
                             'four_step_modes': detour_checked,
                             'carrying_an_inverse_pair': detour_undone}}


def read_draws(support: dict) -> dict:
    """Route lengths of every retained verified MathIR draw, and Level-1 coverage."""
    per_source: dict[str, dict] = {}
    overall: collections.Counter = collections.Counter()
    level1_found: dict[int, set] = collections.defaultdict(set)
    hosted_by_level: dict[str, dict[int, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for name, (description, patterns) in DRAWS.items():
        paths = sorted(p for pattern in patterns
                       for p in glob.glob(str(ROOT / pattern)))
        histogram: collections.Counter = collections.Counter()
        for path in paths:
            with open(path) as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get('domain') != 'mathir':
                        continue
                    if str(record.get('verified')) != 'True':
                        continue
                    key = record.get('canonical_key')
                    if not key or not str(key).startswith('mathir:'):
                        continue
                    histogram[route_length(key)] += 1
                    if str(record.get('level')) == '1':
                        level1_found[int(record['row_index'])].add(key)
                    if name == 'hosted':
                        level = f"level{record.get('level')}"
                        hosted_by_level[level][int(record['row_index'])].add(key)
        per_source[name] = {'what': description, 'files': len(paths),
                            'verified_draws': sum(histogram.values()),
                            'length_histogram': dict(sorted(histogram.items()))}
        overall.update(histogram)

    # Coverage is read on Level 1, the only level whose draws span a deployment
    # that answers every prompt and whose support carries two shortest routes.
    rows = {r['row_index']: r for r in support['levels']['level1']['per_row']}
    certified: collections.Counter = collections.Counter()
    covered: collections.Counter = collections.Counter()
    for index, keys in level1_found.items():
        row = rows[index]
        certified.update(row['lengths'])
        for key in keys:
            covered[route_length(key)] += 1
    # Whether a level's zero is forced by the support or chosen by the policy
    # turns on this: how many solved prompts return exactly one key, at a level
    # whose problems mostly admit two shortest routes.
    hosted_levels = {
        level: {'solved_prompts': len(rows_found),
                'prompts_with_one_key': sum(1 for keys in rows_found.values()
                                            if len(keys) == 1)}
        for level, rows_found in sorted(hosted_by_level.items())}
    return {'sources': per_source,
            'hosted_by_level': hosted_levels,
            'verified_draws': sum(overall.values()),
            'length_histogram': dict(sorted(overall.items())),
            'level1_coverage': {
                'solved_prompts': len(level1_found),
                'certified': dict(sorted(certified.items())),
                'produced': {L: covered.get(L, 0) for L in sorted(certified)}}}


def _percent(part: int, whole: int) -> str:
    return '--' if not whole else f'{round(100 * part / whole)}'


def macros(record: dict) -> str:
    """Every number the appendix paragraph quotes, so none is typed by hand."""
    level1 = record['support']['levels']['level1']
    level2 = record['support']['levels']['level2']
    draws = record['draws']
    coverage = draws['level1_coverage']
    hist1 = {int(k): v for k, v in level1['length_histogram'].items()}
    hist2 = {int(k): v for k, v in level2['length_histogram'].items()}
    certified = {int(k): v for k, v in coverage['certified'].items()}
    produced = {int(k): v for k, v in coverage['produced'].items()}
    longest = max(hist1)
    modes = sorted({int(k) for level in record['support']['levels'].values()
                    for k in level['modes_per_prompt']})
    audit = record['support']['detour_audit']
    lines = ['% Generated by ops/build_paper_mathir_support_geometry.py; do not hand edit.',
             fr'\newcommand{{\MIRlevels}}{{{len(record["support"]["levels"])}}}',
             fr'\newcommand{{\MIRlongroute}}{{{longest}}}']
    # The appendix says every problem admits the same number of modes. If a level
    # ever stops being uniform that sentence is wrong, so withhold the macro and
    # let the build fail on it rather than print a count that is not one count.
    if len(modes) == 1:
        lines.append(fr'\newcommand{{\MIRmodes}}{{{modes[0]}}}')
    hist3 = {int(k): v for k, v in
             record['support']['levels']['level3']['length_histogram'].items()}
    for tag, hist in (('One', hist1), ('Two', hist2), ('Three', hist3)):
        total = sum(hist.values())
        lines += [fr'\newcommand{{\MIRl{tag}short}}{{{hist.get(2, 0)}}}',
                  fr'\newcommand{{\MIRl{tag}mid}}{{{hist.get(3, 0)}}}',
                  fr'\newcommand{{\MIRl{tag}long}}{{{hist.get(longest, 0)}}}',
                  fr'\newcommand{{\MIRl{tag}total}}{{{total}}}',
                  fr'\newcommand{{\MIRl{tag}longshare}}{{{_percent(hist.get(longest, 0), total)}}}']
    level3 = record['support']['levels']['level3']
    one = {int(k): v for k, v in level1['shortest_routes_per_prompt'].items()}
    two = {int(k): v for k, v in level2['shortest_routes_per_prompt'].items()}
    three = {int(k): v for k, v in level3['shortest_routes_per_prompt'].items()}
    hosted = draws.get('hosted_by_level', {})
    lines += [
        fr'\newcommand{{\MIRlOnetwoshortest}}{{{one.get(2, 0)}}}',
        fr'\newcommand{{\MIRlOneoneshortest}}{{{one.get(1, 0)}}}',
        fr'\newcommand{{\MIRlTwooneshortest}}{{{two.get(1, 0)}}}',
        fr'\newcommand{{\MIRlTwoprompts}}{{{level2["prompts"]}}}',
        fr'\newcommand{{\MIRlThreetwoshortest}}{{{three.get(2, 0)}}}',
        fr'\newcommand{{\MIRlThreeprompts}}{{{level3["prompts"]}}}',
        fr'\newcommand{{\MIRhostedThreesolved}}'
        fr'{{{hosted.get("level3", {}).get("solved_prompts", 0)}}}',
        fr'\newcommand{{\MIRhostedThreesingle}}'
        fr'{{{hosted.get("level3", {}).get("prompts_with_one_key", 0)}}}',
        fr'\newcommand{{\MIRoneterminal}}{{{level1["prompts_with_one_terminal_state"]}}}',
        fr'\newcommand{{\MIRdetoursampled}}{{{audit["four_step_modes"]}}}',
        fr'\newcommand{{\MIRdetourundone}}{{{audit["carrying_an_inverse_pair"]}}}',
        fr'\newcommand{{\MIRdetourprompts}}{{{audit["prompts_sampled"]}}}',
        fr'\newcommand{{\MIRdraws}}{{{draws["verified_draws"]:,}}}',
        fr'\newcommand{{\MIRdrawslong}}{{{draws["length_histogram"].get(str(longest), 0)}}}',
        fr'\newcommand{{\MIRsolvedprompts}}{{{coverage["solved_prompts"]}}}',
        fr'\newcommand{{\MIRcovershort}}{{{_percent(produced.get(2, 0), certified.get(2, 0))}}}',
        fr'\newcommand{{\MIRcovermid}}{{{_percent(produced.get(3, 0), certified.get(3, 0))}}}',
        fr'\newcommand{{\MIRcoverlong}}{{{produced.get(longest, 0)}}}',
        fr'\newcommand{{\MIRcertifiedlong}}{{{certified.get(longest, 0)}}}',
    ]
    for name, source in draws['sources'].items():
        tag = name.replace('_', '')
        lines.append(fr'\newcommand{{\MIRdraws{tag}}}{{{source["verified_draws"]:,}}}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true',
                        help='re-enumerate the certified support from the frozen splits')
    args = parser.parse_args()
    if args.recompute or not RECORD.is_file():
        support = enumerate_support()
    else:
        support = json.loads(RECORD.read_text())['support']
    # Enumerating the frozen splits takes minutes and only changes when a split
    # does; reading the retained draws takes seconds, so it is never cached.
    draws = read_draws(support)
    record = {'schema': 'paper-mathir-support-geometry-v1',
              'builder': 'ops/build_paper_mathir_support_geometry.py',
              'what': 'MathIR certified modes by route length, and which lengths any '
                      'retained model draw has ever produced.',
              'support': support, 'draws': draws}
    RECORD.write_text(json.dumps(record, indent=1) + '\n')
    MACROS.write_text(macros(record))
    print(json.dumps({'event': 'built', 'record': str(RECORD), 'macros': str(MACROS),
                      'verified_draws': record['draws']['verified_draws'],
                      'longest_route_draws': record['draws']['length_histogram'].get(
                          str(max(int(k) for k in record['support']['levels']
                                  ['level1']['length_histogram'])), 0)}))


if __name__ == '__main__':
    main()
