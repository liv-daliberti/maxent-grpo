"""E118 keeps valid MaxRL pairs when a historical Dr.GRPO pair is excluded."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e118_integrity_plotter",
    ROOT / "ops/exp_scaling/plot_paper_e118_all_scale_progress.py",
)
assert SPEC is not None and SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def matched_fixture(excluded_seed=59):
    base = {"models": {}}
    trajectory = {"cells": {}}
    record = {"cells": {}}
    for scale, model, seeds in plotter.MODELS[:2]:
        domains = base["models"].setdefault(model, {"domains": {}})["domains"]
        record["cells"][scale] = {}
        for index, domain in enumerate(plotter.DOMAINS):
            sources = domains.setdefault(domain, {"methods": {}})["methods"]
            for arm, offset in (("control", 0.1), ("replay", 0.4)):
                sources[arm] = {"per_seed": {
                    str(seed): {
                        metric: offset + index * 0.01 + position * 0.05
                        for metric in ("pass8", "distinct8")
                    }
                    for position, seed in enumerate(seeds)
                }}
            if scale == "falcon1b" and domain == "countdown":
                del sources["replay"]["per_seed"][str(excluded_seed)]
            trajectory["cells"][f"{scale}/{domain}"] = {"methods": {
                "drgrpo": {"summaries": {"0": {
                    metric: {"per_seed": {
                        str(seed): 0.2 + index * 0.01 + position * 0.05
                        for position, seed in enumerate(seeds)
                    }}
                    for metric in ("pass8", "distinct8")
                }}}
            }}
            cell = {"matched_seeds": list(seeds), "methods": {
                method: {metric: [value] * 5 for metric in ("pass8", "distinct8")}
                for method, value in (("maxrl", 0.5), ("replay_maxrl", 0.8))
            }}
            plotter.attach_reference_methods(
                cell, base=base, trajectory=trajectory,
                scale=scale, model=model, domain=domain,
            )
            record["cells"][scale][domain] = cell
    return base, record


@pytest.mark.parametrize("excluded_seed", [56, 59])
def test_missing_dr_pair_does_not_remove_valid_maxrl_seed(excluded_seed):
    _, record = matched_fixture(excluded_seed)
    cell = record["cells"]["falcon1b"]["countdown"]
    retained = [seed for seed in (55, 56, 57, 58, 59) if seed != excluded_seed]
    assert cell["matched_seeds"] == [55, 56, 57, 58, 59]
    for method in ("maxrl", "replay_maxrl", "before_training"):
        assert cell["method_seeds"][method] == [55, 56, 57, 58, 59]
        assert len(cell["methods"][method]["pass8"]) == 5
    for method in ("drgrpo", "replay_drgrpo", "before_training_drgrpo"):
        assert cell["method_seeds"][method] == retained
        assert len(cell["methods"][method]["pass8"]) == 4


@pytest.mark.parametrize("excluded_seed", [56, 59])
def test_macro_averages_identical_seed_subset_in_every_domain(excluded_seed):
    base, record = matched_fixture(excluded_seed)
    averages = plotter.absolute_cross_domain_averages(record)
    retained = [seed for seed in (55, 56, 57, 58, 59) if seed != excluded_seed]
    source = base["models"]["Falcon3-1B"]["domains"]
    for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
        actual = averages["falcon1b"]["pass8"][method]
        expected = {
            str(seed): statistics.fmean(
                source[domain]["methods"][arm]["per_seed"][str(seed)]["pass8"]
                for domain in plotter.DOMAINS
            )
            for seed in retained
        }
        assert actual["per_seed"] == expected
        assert actual["mean"] == statistics.fmean(expected.values())
        assert actual["seeds"] == retained
        assert actual["n"] == 4
        assert "student_t_95" not in actual
    assert averages["falcon1b"]["pass8"]["before_training_drgrpo"]["seeds"] == retained
    for scale in ("qwen05b", "falcon1b"):
        assert averages[scale]["pass8"]["maxrl"]["n"] == 5
        assert averages[scale]["pass8"]["replay_maxrl"]["n"] == 5
    assert averages["qwen05b"]["pass8"]["drgrpo"]["n"] == 5


def test_main_and_appendix_label_partial_track_and_match_initial_reference():
    _, record = matched_fixture()
    averages = plotter.absolute_cross_domain_averages(record)
    figure, axes = plotter.plt.subplots(1, 2)
    try:
        plotter.draw_cross_domain_panel(
            axes[0], absolute_averages=averages, metric="pass8", title="main",
            xlim=(0, 1),
        )
        assert [text.get_text() for text in axes[0].texts].count("n=4") == 1
        initial_points = [
            float(line.get_xdata()[0]) for line in axes[0].lines
            if line.get_marker() == "D"
        ]
        for method in ("before_training", "before_training_drgrpo"):
            assert averages["falcon1b"]["pass8"][method]["mean"] in initial_points
        assert (
            averages["falcon1b"]["pass8"]["before_training"]["mean"]
            != averages["falcon1b"]["pass8"]["before_training_drgrpo"]["mean"]
        )
        plotter.draw_pair_panel(
            axes[1], record=record, absolute_averages=averages,
            scale="falcon1b", metric="pass8", title="appendix",
            rows=tuple(zip(plotter.DOMAINS, plotter.LABELS)), xlim=(0, 1),
            show_untrained=True,
        )
        assert [text.get_text() for text in axes[1].texts].count("n=4") == 1
        figure.canvas.draw()
    finally:
        plotter.plt.close(figure)


def qwen3b_reference_fixture():
    base, record = matched_fixture()
    seeds = [70, 71, 72, 73, 74]
    domains = base['models'].setdefault('Qwen2.5-3B', {'domains': {}})['domains']
    initial = {'target_step': 3072, 'domains': {}}
    record['cells']['qwen3b'] = {}
    for index, domain in enumerate(plotter.DOMAINS):
        methods = {}
        for arm, offset in (('control', .2), ('replay', .5)):
            methods[arm] = {'per_seed': {
                str(seed): {metric: offset + .01 * index + .02 * position
                            for metric in ('pass8', 'distinct8')}
                for position, seed in enumerate(seeds)
            }}
        domains[domain] = {'methods': methods}
        initial['domains'][domain] = {'per_seed': {
            str(seed): {
                'pass0': {metric: .1 + .01 * index for metric in ('pass8', 'distinct8')},
                'pass8': dict(methods['control']['per_seed'][str(seed)]),
            }
            for seed in seeds
        }}
        record['cells']['qwen3b'][domain] = {
            'matched_seeds': [],
            'method_seeds': {method: [] for method in ('maxrl', 'replay_maxrl')},
            'methods': {method: {metric: [] for metric in ('pass8', 'distinct8')}
                        for method in ('maxrl', 'replay_maxrl')},
        }
    return base, record, {'arms': {'drgrpo': {'scales': {'qwen3b': initial}}}}


def test_complete_qwen3b_reference_ignores_unfinished_maxrl_pairs():
    import copy
    base, record, precheck = qwen3b_reference_fixture()
    original = copy.deepcopy(record)
    plotter.attach_complete_qwen3b_reference_track(record, base=base, precheck=precheck)
    for scale in ('qwen05b', 'falcon1b'):
        assert record['cells'][scale] == original['cells'][scale]
    for domain in plotter.DOMAINS:
        cell = record['cells']['qwen3b'][domain]
        assert cell['matched_seeds'] == []
        for method in ('maxrl', 'replay_maxrl'):
            assert cell['method_seeds'][method] == []
            assert cell['methods'][method] == original['cells']['qwen3b'][domain]['methods'][method]
        for method in ('before_training_drgrpo', 'drgrpo', 'replay_drgrpo'):
            assert cell['method_seeds'][method] == [70, 71, 72, 73, 74]
    averages = plotter.absolute_cross_domain_averages(record)
    for metric in ('pass8', 'distinct8'):
        assert set(averages['qwen3b'][metric]) == {
            'before_training_drgrpo', 'drgrpo', 'replay_drgrpo',
        }
        assert averages['qwen3b'][metric]['drgrpo']['mean'] == pytest.approx(.26)
        assert averages['qwen3b'][metric]['replay_drgrpo']['mean'] == pytest.approx(.56)
    figure, axis = plotter.plt.subplots()
    try:
        plotter.draw_cross_domain_panel(
            axis, absolute_averages=averages, metric='pass8', title='main', xlim=(0, 1),
        )
        assert [text.get_text() for text in axis.get_yticklabels()] == [
            'Qwen2.5-0.5B', 'Falcon3-1B', 'Qwen2.5-3B',
        ]
        assert not any(line.get_marker() == 's' and min(line.get_ydata()) < .5
                       for line in axis.lines)
        assert any(line.get_marker() == 'o' and min(line.get_ydata()) < .5
                   for line in axis.lines)
        figure.canvas.draw()
    finally:
        plotter.plt.close(figure)


def test_qwen3b_reference_fails_closed_if_a_completed_pair_is_missing():
    base, record, precheck = qwen3b_reference_fixture()
    del base['models']['Qwen2.5-3B']['domains']['countdown']['methods']['replay']['per_seed']['74']
    with pytest.raises(RuntimeError, match='requires five paired seeds'):
        plotter.attach_complete_qwen3b_reference_track(record, base=base, precheck=precheck)


def qwen3b_partial_factorial_fixture():
    base, record, precheck = qwen3b_reference_fixture()
    counts = (5, 3, 5, 3, 1)
    for index, (domain, n) in enumerate(zip(plotter.DOMAINS, counts)):
        cell = record['cells']['qwen3b'][domain]
        seeds = ([70, 71, 72, 73, 74] if n == 5 else [70, 71, 74] if n == 3 else [72])
        cell['matched_seeds'] = list(seeds)
        for method, offset in (('maxrl', .2), ('replay_maxrl', .6)):
            cell['method_seeds'][method] = list(seeds)
            cell['methods'][method] = {
                metric: [offset + .01 * index + .02 * (seed - 70) for seed in seeds]
                for metric in ('pass8', 'distinct8')
            }
    plotter.attach_complete_qwen3b_reference_track(record, base=base, precheck=precheck)
    return record, precheck


def test_qwen3b_partial_maxrl_initial_reference_uses_its_own_paired_seeds():
    record, precheck = qwen3b_partial_factorial_fixture()
    for domain in plotter.DOMAINS:
        cell = record['cells']['qwen3b'][domain]
        seeds = cell['matched_seeds']
        assert cell['method_seeds']['before_training'] == seeds
        assert cell['method_seeds']['before_training_drgrpo'] == [70, 71, 72, 73, 74]
        origin = precheck['arms']['drgrpo']['scales']['qwen3b']['domains'][domain]['per_seed']
        for metric in ('pass8', 'distinct8'):
            assert cell['methods']['before_training'][metric] == [
                origin[str(seed)]['pass0'][metric] for seed in seeds
            ]
    averages = plotter.absolute_cross_domain_averages(record)
    for metric in ('pass8', 'distinct8'):
        assert set(averages['qwen3b'][metric]) == {
            'before_training_drgrpo', 'drgrpo', 'replay_drgrpo',
        }


@pytest.mark.parametrize('empty_domain', [None, 'pantry_plan'])
def test_qwen3b_domain_detail_shows_exact_coverage_and_only_paired_maxrl(empty_domain):
    record, _ = qwen3b_partial_factorial_fixture()
    if empty_domain:
        cell = record['cells']['qwen3b'][empty_domain]
        cell['matched_seeds'] = []
        for method in ('maxrl', 'replay_maxrl', 'before_training'):
            cell['method_seeds'][method] = []
            cell['methods'][method] = {'pass8': [], 'distinct8': []}
    figure, axis = plotter.plt.subplots()
    try:
        plotter.draw_pair_panel(
            axis, record=record, absolute_averages={}, scale='qwen3b',
            metric='distinct8', title='domain evidence',
            rows=tuple(zip(plotter.DOMAINS, plotter.LABELS)), xlim=(0, 2.5),
            tracks=plotter.PAIR_TRACKS[:1], counts_in_labels=True,
        )
        assert [text.get_text() for text in axis.get_yticklabels()] == [
            f"{label} (n={len(record['cells']['qwen3b'][domain]['matched_seeds'])})"
            for domain, label in zip(plotter.DOMAINS, plotter.LABELS)
        ]
        markers = [line for line in axis.lines if line.get_marker() == 's']
        assert len(markers) == 2 * (5 - bool(empty_domain))
        assert not any(line.get_marker() in ('o', 'D') for line in axis.lines)
        expected = [
            statistics.fmean(record['cells']['qwen3b'][domain]['methods'][method]['distinct8'])
            for domain in plotter.DOMAINS if domain != empty_domain
            for method in ('maxrl', 'replay_maxrl')
        ]
        assert [float(line.get_xdata()[0]) for line in markers] == expected
        # The domain panel represents available paired means, with no intervals.
        assert not axis.containers
        figure.canvas.draw()
    finally:
        plotter.plt.close(figure)



def test_descriptive_qwen3b_average_balances_domains_without_inventing_common_seeds():
    record, _ = qwen3b_partial_factorial_fixture()
    cells = record['cells']['qwen3b']
    assert not set.intersection(*(set(cell['matched_seeds']) for cell in cells.values()))
    result = plotter.descriptive_available_domain_averages(record)['qwen3b']
    for metric in ('pass8', 'distinct8'):
        for method in ('before_training', 'maxrl', 'replay_maxrl'):
            summary = result[metric][method]
            expected = {
                domain: statistics.fmean(cells[domain]['methods'][method][metric])
                for domain in plotter.DOMAINS
            }
            assert summary['domain_means'] == expected
            assert summary['mean'] == statistics.fmean(expected.values())
            assert summary['domain_seed_counts'] == dict(zip(plotter.DOMAINS, (5, 3, 5, 3, 1)))
            assert summary['domain_weights'] == {domain: .2 for domain in plotter.DOMAINS}
            assert not {'n', 'seeds', 'per_seed', 'student_t_95', 'standard_error'} & set(summary)
            pooled = [value for cell in cells.values() for value in cell['methods'][method][metric]]
            assert summary['mean'] != pytest.approx(statistics.fmean(pooled))
            for domain, cell in cells.items():
                assert summary['per_domain_per_seed'][domain] == dict(zip(
                    map(str, cell['matched_seeds']), cell['methods'][method][metric], strict=True,
                ))
    # The existing valid common-seed aggregates stay unchanged and separate.
    regular = plotter.absolute_cross_domain_averages(record)
    assert set(regular['qwen3b']['pass8']) == {'before_training_drgrpo', 'drgrpo', 'replay_drgrpo'}


def test_descriptive_qwen3b_requires_every_domain_and_its_matched_reference():
    record, _ = qwen3b_partial_factorial_fixture()
    record['cells']['qwen3b']['pantry_plan']['method_seeds']['before_training'] = [70]
    with pytest.raises(RuntimeError, match='unmatched available-domain reference'):
        plotter.descriptive_available_domain_averages(record)
    record['cells']['qwen3b']['pantry_plan']['matched_seeds'] = []
    assert plotter.descriptive_available_domain_averages(record) == {}


def test_main_model_row_shows_qwen3b_maxrl_without_fake_seed_paths_or_intervals():
    record, _ = qwen3b_partial_factorial_fixture()
    regular = plotter.absolute_cross_domain_averages(record)
    descriptive = plotter.descriptive_available_domain_averages(record)
    figure, axis = plotter.plt.subplots()
    try:
        plotter.draw_cross_domain_panel(
            axis, absolute_averages=regular, descriptive_averages=descriptive,
            metric='pass8', title='main', xlim=(0, 1),
        )
        upper_3b_lines = [line for line in axis.lines
                          if all(0 < y < .5 for y in line.get_ydata())]
        squares = [line for line in upper_3b_lines if line.get_marker() == 's']
        assert len(squares) == 2
        assert [float(line.get_xdata()[0]) for line in squares] == [
            descriptive['qwen3b']['pass8'][method]['mean']
            for method in ('maxrl', 'replay_maxrl')
        ]
        assert not any(line.get_alpha() is not None for line in upper_3b_lines)
        assert 'n=1–5/domain' in [text.get_text() for text in axis.texts]
        assert not axis.containers
        figure.canvas.draw()
    finally:
        plotter.plt.close(figure)
