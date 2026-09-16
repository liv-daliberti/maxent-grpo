#!/usr/bin/env python3
"""Render every registered conditional-concentration contrast into appendix tables."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'paper/results/conditional_concentration_20260911.json'
TARGET = ROOT / 'paper/results/conditional_concentration_20260911_tables.tex'
DOMAINS = {'graph_coloring':'Graph','countdown':'Countdown','python_factors':'Python','mathir':'MathIR','pantry_plan':'Pantry'}
METHODS = {'drgrpo':'Dr.GRPO','grpo':'GRPO','maxrl':'MaxRL','replay_drgrpo':'Re:Dr','replay_maxrl':'Re:Max'}
SCALES = {'qwen05b':'Qwen2.5-0.5B','falcon1b':'Falcon3-1B','qwen3b':'Qwen2.5-3B'}

def number(v):
    return '--' if v is None else f'{v:+.3f}'

def render(record):
    out = ['% Generated from the frozen saved-output concentration analysis.']
    for kind in ('before_after','replay_effect'):
        for level in ('level1','level2'):
            for scale, model in SCALES.items():
                blocks = [b for b in record['blocks'] if (b['kind'],b['level'],b['scale']) == (kind,level,scale)]
                if not blocks:
                    continue
                comparison = 'Initial to final' if kind=='before_after' else 'Replay minus control at the final checkpoint'
                out += [r'\begin{table}[!htbp]',r'\centering',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',
                    r'\begin{tabular}{@{}llcrlrrr@{}}',r'\toprule',
                    r'Domain & Objective & $n$ & $\Delta\widehat C$ & nominal 95\% interval & $|E_s|$ & split 0 ($n$) & split 1 ($n$) \\',r'\midrule']
                for b in blocks:
                    s=b['summaries'];v=s['distinct_streams'];ci=v['ci95']
                    interval='--' if ci is None else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'
                    counts=list(v['eligible_counts'].values())
                    coverage='--' if not counts else (str(min(counts)) if min(counts)==max(counts) else f'{min(counts)}--{max(counts)}')
                    split=[f"{number(s[x]['mean'])} ({s[x]['n']})" for x in ('orientation0','orientation1')]
                    out.append(f"{DOMAINS[b['domain']]} & {METHODS[b['method']]} & {v['n']} & {number(v['mean'])} & {interval} & {coverage} & {split[0]} & {split[1]} " + r'\\')
                out += [r'\bottomrule',r'\end{tabular}',
                    f"\\caption{{\\textbf{{{comparison}: {model}, Level {level[-1]}.}} "
                    r'Positive values indicate greater concentration. Equal prompt weights use $R\ge2$ correct representatives in both conditions, separately for each contrast; $|E_s|$ is the range of eligible prompt counts out of 128 across available seed pairs. '
                    r'$n$ counts seeds with a defined estimate, from five registered seeds. Each disjoint orientation has its own eligibility and displayed $n$. Intervals describe measured seed variability, are unadjusted and untruncated, and are omitted unless all five seeds are defined. Dashes retain unavailable or unidentified comparisons. Every registered comparison remains represented.}',r'\end{table}','']
    return '\n'.join(out)

if __name__=='__main__':
    TARGET.write_text(render(json.loads(SOURCE.read_text()))+'\n')
