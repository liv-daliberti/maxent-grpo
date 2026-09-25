#!/usr/bin/env python3
"""Build E121's registered five-seed fixed-exemplar score analysis.

The independent integrity audit gates admission and binds every input by SHA256.
No identities are selected by their scores. The 10,000-draw bootstrap samples
prompt clusters within each original seed, then five seeds with replacement;
a repeated seed reuses its within-draw prompt sample. Percentile interpolation,
pooling identities equally, and the displayed bootstrap statistics are reporting
choices made after completion; the protocol specifies the hierarchy, count and
RNG seed, but not these implementation details.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'paper/audits/e121_20260909/integrity.json'
PROTOCOL = ROOT / 'paper/preregistration/e121_fixed_bank_survival_telemetry_20260903.md'
LEDGER = ROOT / 'var/artifacts/e121_fixed_bank_survival_telemetry_jobs.json'
OUT = ROOT / 'paper/results/e121_fixed_bank_survival'
FIGURE = ROOT / 'paper/figures/e121_fixed_bank_survival'
SEEDS = (43, 44, 45, 46, 47)
FREEZE, HORIZON = 384, 3072
DRAWS, RNG_SEED = 10_000, 121
STAT_NAMES = ('median', 'p10', 'minimum', 'fraction_below_minus_0_5', 'worst_intermediate_delta')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_audit() -> dict:
    audit = json.loads(AUDIT.read_text())
    if audit.get('schema') != 'e121_independent_integrity_audit_v1' or audit.get('passed') is not True:
        raise ValueError('E121 requires the successful independent coverage audit')
    if audit['ledger_sha256'] != sha256(LEDGER) or audit['preregistration_sha256'] != sha256(PROTOCOL):
        raise ValueError('E121 ledger or registration differs from the coverage audit')
    if sorted(r['seed'] for r in audit['runs']) != list(SEEDS):
        raise ValueError('E121 requires exactly the five registered seeds')
    for run in audit['runs']:
        if (run.get('passed') is not True or run['missing_identity_count'] != 0
                or run['insufficient_visit_count'] != 0
                or run['complete_finite_score_fraction'] != 1
                or run['complete_round_robin_schedule'] is not True):
            raise ValueError(f"seed {run['seed']}: mechanism audit failure")
        if sha256(Path(run['metrics'])) != run['metrics_sha256']:
            raise ValueError(f"seed {run['seed']}: metrics differ from the coverage audit")
        if sha256(Path(run['run_dir']) / 'TRAINING_COMPLETE.json') != run['completion_receipt_sha256']:
            raise ValueError(f"seed {run['seed']}: completion receipt differs from the coverage audit")
    return audit


def read_histories(run: dict) -> list[dict]:
    """Read actual update records already checked against the full frozen bank."""
    history = defaultdict(list)
    prefix = 'train/canonical_replay_'
    updates = []
    with Path(run['metrics']).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record.get('trainer/step', -1))
            # The 3073 completion record carries update3072 scores; the audit
            # verifies the duplicate before this parser excludes that record.
            if not FREEZE <= step <= HORIZON:
                continue
            if record['trainer/policy_sgd_step'] != step or record[prefix + 'bank_membership_frozen'] != 1:
                raise ValueError(f'non-update or non-frozen record at {step}')
            updates.append(step)
            for key in sorted(record):
                if not key.startswith(prefix + 'outcome_fingerprint_row_'):
                    continue
                index = key.rsplit('_', 1)[1]
                identity = (int(record[prefix + 'prompt_fingerprint_row_' + index]), int(record[key]))
                mean = float(record[prefix + 'exemplar_mean_logprob_row_' + index])
                sequence = float(record[prefix + 'exemplar_sequence_logprob_row_' + index])
                if not np.isfinite([mean, sequence]).all():
                    raise ValueError(f'non-finite identity score at {step}')
                history[identity].append([step, mean, sequence])
    if updates != list(range(FREEZE, HORIZON + 1)):
        raise ValueError('missing or repeated post-freeze optimizer update')
    if (len(history) != run['frozen_identities']
            or len({p for p, _ in history}) != run['frozen_prompts']
            or sum(map(len, history.values())) != run['identity_observations']):
        raise ValueError('analysis population differs from independent full-bank audit')
    result = []
    for (prompt, outcome), rows in sorted(history.items()):
        if len(rows) < 2:
            raise ValueError('frozen identity has fewer than two observations')
        scores = np.asarray(rows, dtype=float)[:, 1:]
        delta = scores[-1] - scores[0]
        worst = np.minimum(0, (scores - scores[0]).min(axis=0))
        result.append(dict(seed=run['seed'], prompt_fingerprint=prompt,
                           outcome_fingerprint=outcome, visit_count=len(rows),
                           delta_mean_logprob=float(delta[0]),
                           delta_sequence_logprob=float(delta[1]),
                           worst_mean_intermediate_delta=float(worst[0]),
                           worst_sequence_intermediate_delta=float(worst[1]),
                           observations=rows))
    return result


def array_from_identities(identities: list[dict]) -> np.ndarray:
    return np.asarray([[r['delta_mean_logprob'], r['delta_sequence_logprob'],
                        r['worst_mean_intermediate_delta'], r['worst_sequence_intermediate_delta']]
                       for r in identities], dtype=float)


def statistics(values: np.ndarray) -> np.ndarray:
    """Rows: median, p10, minimum, fraction below -0.5, worst; columns: mean/sequence."""
    if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
        raise ValueError('statistics require finite nonempty identity rows with four columns')
    deltas = values[:, :2]
    q = np.quantile(deltas, [.5, .1, 0], axis=0, method='linear')
    return np.vstack([q, np.mean(deltas < -.5, axis=0), np.min(values[:, 2:], axis=0)])


def named_statistics(values: np.ndarray) -> dict:
    rows = statistics(values)
    return {metric: {name: float(rows[i, j]) for i, name in enumerate(STAT_NAMES)}
            for j, metric in enumerate(('mean_logprob', 'sequence_logprob'))}


def clustered_sample(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Sample prompts with replacement and retain every identity of each prompt."""
    selection = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in selection], axis=0)


def hierarchical_bootstrap(seed_identities: list[list[dict]], draws: int = DRAWS,
                           rng_seed: int = RNG_SEED) -> dict:
    groups = []
    for identities in seed_identities:
        prompts = defaultdict(list)
        for identity in identities:
            prompts[identity['prompt_fingerprint']].append(identity)
        groups.append([array_from_identities(prompts[p]) for p in sorted(prompts)])
    rng = np.random.default_rng(rng_seed)
    samples = np.empty((draws, len(STAT_NAMES), 2))
    for b in range(draws):
        prompt_samples = [clustered_sample(seed, rng) for seed in groups]
        selected_seeds = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([prompt_samples[i] for i in selected_seeds], axis=0)
        samples[b] = statistics(sample)
    interval = np.quantile(samples, [.025, .975], axis=0, method='linear')
    return {metric: {name: interval[:, i, j].tolist() for i, name in enumerate(STAT_NAMES)}
            for j, metric in enumerate(('mean_logprob', 'sequence_logprob'))}


def signed(value: float) -> str:
    return f'{value:+.3f}' if abs(value) >= .0005 else '0.000'


def write_tables(result: dict) -> None:
    mean_rows, sequence_rows = [], []
    for run in [*result['runs'], result['pooled']]:
        label = str(run.get('seed', 'Pooled'))
        mean, seq = run['statistics']['mean_logprob'], run['statistics']['sequence_logprob']
        counts = [label, str(run['prompt_count']), str(run['identity_count']),
                  f"{run['visit_count_min']}--{run['visit_count_max']}"]
        mean_rows.append(' & '.join(counts + [signed(mean[k]) for k in ('median', 'p10', 'minimum')]
                                   + [f"{100 * mean['fraction_below_minus_0_5']:.2f}"]) + r' \\')
        sequence_rows.append(' & '.join([label] + [signed(seq[k]) for k in ('median', 'p10', 'minimum')]
                                       + [f"{100 * seq['fraction_below_minus_0_5']:.2f}",
                                          signed(mean['worst_intermediate_delta']),
                                          signed(seq['worst_intermediate_delta'])]) + r' \\')
    for suffix, rows in [('_table_body.tex', mean_rows), ('_sequence_table_body.tex', sequence_rows)]:
        Path(str(OUT) + suffix).write_text('% Generated by build_paper_e121_survival.py; do not hand edit.\n'
                                         + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n')
    rows = []
    for key, label in [('median', 'Median'), ('p10', '10th percentile'),
                       ('fraction_below_minus_0_5', r'Fraction $\Delta < -0.5$ (\%)')]:
        cols = [label]
        for metric in ('mean_logprob', 'sequence_logprob'):
            value = result['pooled']['statistics'][metric][key]
            low, high = result['bootstrap']['percentile_95'][metric][key]
            if key.startswith('fraction'):
                cell = f'{100 * value:.2f} [{100 * low:.2f}, {100 * high:.2f}]'
            else:
                cell = f'{signed(value)} [{signed(low)}, {signed(high)}]'
            cols.append(cell)
        rows.append(' & '.join(cols) + r' \\')
    Path(str(OUT) + '_bootstrap_table_body.tex').write_text(
        '% Generated by build_paper_e121_survival.py; do not hand edit.\n' + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n')


def plot(result: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    colors = ['#2D658B', '#D58A32', '#598C61', '#AF607D', '#8070A6']
    for ax, key, label in zip(axes, ('delta_mean_logprob', 'delta_sequence_logprob'),
                             ('Change in mean log probability (nat/token)',
                              'Change in sequence log probability (nat)')):
        all_values = []
        for replicate, (run, color) in enumerate(zip(result['runs'], colors), start=1):
            x = np.sort([r[key] for r in run['identities']])
            ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', color=color,
                    linewidth=1.0, alpha=.85, label=f"Seed {replicate}")
            all_values.extend(x)
        x = np.sort(all_values)
        ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', color='#202735',
                linewidth=1.7, label='Pooled')
        ax.axvline(0, color='#7F8895', linewidth=.7, linestyle=':')
        ax.axvline(-.5, color='#B2B7BE', linewidth=.7, linestyle='--')
        ax.set_xlabel(label, fontsize=8)
        ax.grid(axis='y', color='#E4E6E9', linewidth=.6)
        ax.set_ylim(0, 1.01)
        ax.set_xlim(min(x)-.025*np.ptp(x), max(x)+.025*np.ptp(x))
    axes[0].set_ylabel('Fraction of exemplars')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False,
               fontsize=8, handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, .88), w_pad=1.4)
    fig.savefig(FIGURE.with_suffix('.pdf'), bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIGURE.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    audit = checked_audit()
    runs = []
    for source in sorted(audit['runs'], key=lambda r: r['seed']):
        identities = read_histories(source)
        runs.append(dict(seed=source['seed'], job_id=source['job_id'],
                         prompt_count=source['frozen_prompts'], identity_count=len(identities),
                         identity_observations=source['identity_observations'],
                         postfreeze_updates=source['postfreeze_update_count'],
                         visit_count_min=min(r['visit_count'] for r in identities),
                         visit_count_max=max(r['visit_count'] for r in identities),
                         finite_at_every_scheduled_observation_fraction=1.0,
                         statistics=named_statistics(array_from_identities(identities)),
                         identities=identities))
    all_identities = [identity for run in runs for identity in run['identities']]
    pooled = dict(prompt_count=sum(r['prompt_count'] for r in runs), identity_count=len(all_identities),
                  identity_observations=sum(r['identity_observations'] for r in runs),
                  visit_count_min=min(r['visit_count'] for r in all_identities),
                  visit_count_max=max(r['visit_count'] for r in all_identities),
                  finite_at_every_scheduled_observation_fraction=1.0,
                  statistics=named_statistics(array_from_identities(all_identities)))
    intervals = hierarchical_bootstrap([r['identities'] for r in runs])
    result = dict(schema='e121-fixed-bank-survival-paper-v1', analysis_date='2026-09-09',
                  model='Qwen2.5-0.5B-Instruct', method='Re:Dr', domain='Graph coloring',
                  registered_seeds=list(SEEDS), freeze_step=FREEZE, final_optimizer_update=HORIZON,
                  selection_rule='Every frozen seed/prompt/outcome identity; all meet the registered >=2-visit rule; none excluded.',
                  score_timing='Teacher-forced scores at scheduled replay visits; per-identity first/final visits differ across prompts.',
                  bootstrap=dict(draws=DRAWS, rng_seed=RNG_SEED, rng='numpy.random.default_rng (PCG64)',
                                 resampling='Resample prompts within each original seed, preserving every identity in each sampled prompt; then sample five seeds with replacement. Repeated seeds reuse their within-draw prompt sample.',
                                 estimand='Pooled identity-weighted summaries; larger retained prompt banks contribute more identities.',
                                 interpolation='numpy linear quantiles; percentile 95% intervals',
                                 implementation_choices='Pooling, interpolation, and which summaries receive displayed intervals were not specified in the preregistration.',
                                 percentile_95=intervals),
                  limitations=['Fixed-exemplar LM scores are not exact canonical-mode probabilities.',
                               'Finite teacher-forced scores alone do not establish meaningful sampling probability.',
                               'There is no matched frozen-bank no-replay control: score changes cannot be causally attributed to replay.',
                               'One model, one domain, bank discovered during the first pass; no claim for unseen modes.',
                               'Worst intermediate drops concern scheduled visits, not every intervening update.',
                               'Sequence scores are length-dependent; exemplar token lengths range from 4 to 100.',
                               'Resume checkpoints were pruned automatically on completion; full-bank aggregate counts plus complete scheduler cycles establish telemetry coverage.'],
                  provenance=dict(audit=str(AUDIT.relative_to(ROOT)), audit_sha256=sha256(AUDIT),
                                  protocol=str(PROTOCOL.relative_to(ROOT)), protocol_sha256=sha256(PROTOCOL),
                                  ledger=str(LEDGER.relative_to(ROOT)), ledger_sha256=sha256(LEDGER),
                                  source_snapshot_sha256=audit['snapshot_identity_sha256'],
                                  analysis_script_sha256=sha256(Path(__file__)), numpy_version=np.__version__,
                                  source_runs=audit['runs'],
                                  operational_amendment='User-authorized September8 removal of E120 scheduling dependencies and node204 placement; scientific recipe unchanged.'),
                  runs=runs, pooled=pooled)
    OUT.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    write_tables(result)
    plot(result)
    print(json.dumps(dict(pooled=pooled, intervals=intervals), indent=2))


if __name__ == '__main__':
    main()
