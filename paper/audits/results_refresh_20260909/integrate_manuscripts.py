from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[3]
latest=json.loads((ROOT/'paper/results/latest_results_20260909.json').read_text())
assert [latest['campaigns'][c]['admitted_terminal_endpoints'] for c in ('e118','e119','e120')]==[126,83,35]
assert latest['campaigns']['e119']['complete_five_seed_blocks']==4
section=(ROOT/'paper/audits/results_refresh_20260909/current_results_section.tex').read_text()
section=section.replace(
    'Graph provides the clearest completed Level-2 evidence: replay increases',
    'Four Level-2 domains now have complete factorials. Countdown adds a small\nReplayMaxRL breadth effect ($+.017$, interval $[+.001,+.032]$), while its\ncorrectness and raw-support intervals include zero. Graph provides the\nclearest completed Level-2 evidence: replay increases')
start=latest['collection_started_at_utc'][11:19]
end=latest['collection_finished_at_utc'][11:19]

for name in ('paper/main.tex','paper/mathai2026/appendix.tex'):
    path=ROOT/name
    s=path.read_text()
    def replace(old,new):
        global s
        if old not in s: raise RuntimeError(f'missing text in {name}: {old[:100]}')
        s=s.replace(old,new)
    replace('September 8', 'September 9')
    replace('latest_results_20260908', 'latest_results_20260909')
    replace(r'latest\_results\_20260908', r'latest\_results\_20260909')
    replace('15:26:54 to 15:36:39 UTC', f'{start} to {end} UTC')
    replace('3/5 blocks; Graph, Python, MathIR complete & finish Countdown and PantryPlan',
            '4/5 blocks; all except PantryPlan & finish PantryPlan')
    replace('6/9 blocks; 33/45 cells', '6/9 blocks; 35/45 cells')
    replace('''levels and all four methods. Graph, Python, and MathIR use step 3,072 for
all five seeds. For seeds 43--47, Countdown uses steps 3,072, 3,072, 2,592,
2,304, and 2,304. These means describe the matched checkpoints available in
the snapshot. The complete eight-pass Graph, Python, and MathIR factorials''',
            '''levels and all four methods. Graph, Countdown, Python, and MathIR use
step 3,072 for all five seeds. These means describe the matched checkpoints
available in the snapshot. The four complete eight-pass factorials''')
    replace(r'''E118 now contains 59 matched MaxRL/ReplayMaxRL pairs. Its eleven complete
blocks are unchanged; Qwen2.5-3B Graph has three pairs (seeds 70, 72, and 73)
and MathIR has one (seed 74). Countdown and PantryPlan have independently
completed arms but no paired terminal intersection. These prefixes remain
descriptive, and the incomplete Qwen2.5-3B MaxRL grid is not averaged across
domains in Figure~\ref{fig:maxrl-factorial}.''',
            r'''E118 contains 61 matched MaxRL/ReplayMaxRL pairs. Its eleven complete
blocks are unchanged; Qwen2.5-3B Graph has four pairs (seeds 70--73),
MathIR has one (seed 74), and PantryPlan has one (seed 72). Countdown
has no paired terminal intersection. These partial blocks remain descriptive,
and the incomplete Qwen2.5-3B MaxRL grid is not averaged across domains in
Figure~\ref{fig:maxrl-factorial}. Table~\ref{tab:current-e118} reports all
available paired means.''')
    replace(r'''E119 adds complete Python and MathIR factorials to the existing Graph block,
using all five registered seeds in each. Table~\ref{tab:latest-completed-effects}
reports their paired replay effects. Every new \texttt{pass@8} and raw
\texttt{distinct@8} mean is positive, but all corresponding unadjusted 95\%
intervals include zero. Countdown has two complete four-arm seeds (43 and 44);
PantryPlan has no terminal four-arm intersection. Figure~\ref{fig:level2-admission}
uses the refreshed matched-checkpoint snapshot described above.''',
            r'''E119 has complete Graph, Countdown, Python, and MathIR factorials,
using all five registered seeds in each. Table~\ref{tab:latest-completed-effects}
reports the three blocks completed since the September 6 census. Their
\texttt{pass@8} and raw \texttt{distinct@8} replay means are positive, but all
corresponding unadjusted 95\% intervals include zero. Countdown's ReplayMaxRL
extra-mode effect is $+.017$ ($[+.001,+.032]$). PantryPlan has three
individually completed runs and no paired terminal intersection.
Figure~\ref{fig:level2-admission} uses the refreshed matched-checkpoint
snapshot described above; Tables~\ref{tab:current-e119}
and~\ref{tab:level2-factorial-contrasts} give all terminal effects.''')
    replace('Newly completed Level-2 Python and MathIR replay effects.',
            'Completed Level-2 Countdown, Python, and MathIR replay effects.')
    replace(r'''E120 remains at 33 admitted treatment endpoints and six complete blocks;
the Falcon3-1B Graph result in Appendix~\ref{app:e120-falcon-graph-update}
is unchanged. The original 25-cell Qwen2.5-0.5B weighting analysis retains its
September 4 frozen input, and all 25 treatment endpoints were revalidated
without numerical changes. Remaining scale extensions are incomplete.''',
            r'''E120 has 35 admitted treatment endpoints and six complete blocks.
Qwen2.5-3B Graph adds two pairs (seeds 70 and 71); Falcon3-1B PantryPlan
retains three (57--59), and Qwen2.5-3B PantryPlan has none. All observed
pairs pass the persisted weighting-telemetry audit. The Falcon3-1B Graph
result in Appendix~\ref{app:e120-falcon-graph-update} is unchanged.
The original 25-cell Qwen2.5-0.5B weighting analysis retains its September 4
frozen input, and all 25 treatment endpoints were revalidated without
numerical changes. Table~\ref{tab:current-e120} includes the partial extensions.''')
    replace(r'\subsection{Training and evaluation contract}',
            section+'\n\n'+r'\subsection{Training and evaluation contract}')
    s=s.replace('and three domains have complete Level-2 factorials.',
                'and four domains have complete Level-2 factorials.')
    s=s.replace('complete five-seed blocks on Graph, Python, and\nMathIR',
                'complete five-seed blocks on Graph, Countdown, Python, and\nMathIR')
    if name=='paper/main.tex':
        replace(r'''Python and MathIR now also have complete five-seed factorials. Their co-primary
replay means are positive under both objectives, but the unadjusted 95\% intervals
include zero (Appendix Table~\ref{tab:latest-completed-effects}).''',
                r'''Countdown, Python, and MathIR also have complete five-seed factorials.
Their correctness and raw-support intervals include zero. Countdown's
ReplayMaxRL extra-mode gain is small, $+.017$ ($[+.001,+.032]$).
All terminal contrasts and factorial interactions appear in
Appendix~\ref{app:current-campaign-results}.''')
    path.write_text(s)

path=ROOT/'paper/mathai2026/main.tex'
s=path.read_text().replace(
    'Graph, Python, and MathIR have complete five-seed, four-method blocks after eight passes.',
    'Four domains have complete five-seed factorials; PantryPlan is still running.')
s=s.replace(r'''The five-domain comparison remains interim; Python and MathIR co-primary
replay intervals include zero (Appendix Table~\ref{tab:latest-completed-effects}).''',
            r'''Countdown adds $.017$ extra modes under ReplayMaxRL; its correctness
interval includes zero (all terminal results: Appendix~\ref{app:current-campaign-results}).''')
path.write_text(s)

path=ROOT/'ops/check_paper_current_contract.py'
s=path.read_text().replace('latest_results_20260908','latest_results_20260909').replace(
    '"2026-09-08"','"2026-09-09"').replace('September 8 update','September 9 update')
s=s.replace('3/5 blocks; Graph, Python, MathIR complete','4/5 blocks; all except PantryPlan')
s=s.replace('finish Countdown and PantryPlan','finish PantryPlan')
s=s.replace('6/9 blocks; 33/45 cells','6/9 blocks; 35/45 cells')
path.write_text(s)

path=ROOT/'paper/Makefile'
s=path.read_text().replace(
    '$(wildcard results/latest_results_*.json results/latest_results_*.tex',
    '$(wildcard results/current_campaign_results_*.json results/current_campaign_results_*.tex results/level2_factorial_contrasts_*.json results/level2_factorial_contrasts_*.tex results/latest_results_*.json results/latest_results_*.tex')
path.write_text(s)
print('Integrated September 9 terminal results into both manuscripts and the parent contract.')

