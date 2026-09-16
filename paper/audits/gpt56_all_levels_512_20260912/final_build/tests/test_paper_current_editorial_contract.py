"""Guard narrative relocation without weakening evidence or rendered assets."""
from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'paper_current_editorial_test', ROOT / 'ops/check_paper_current_contract.py')
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture
def manuscript():
    text = (ROOT / 'paper/main.tex').read_text()
    main, appendix = text.split(r'\appendix', 1)
    return text, main, appendix


def test_requested_displays_and_roles_and_preserved_math_pass(manuscript):
    text, main, appendix = manuscript
    checker.check_editorial_structure(main, appendix)
    checker.check_formal_preservation(text, checker.PROOF_REFERENCE.read_text())


def test_evidence_roles_do_not_depend_on_heading_wording(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\section{Results}', r'\section{What does replay change?}')
    main = main.replace(r'\section{Related Work}', r'\section{Connections and interpretation}')
    checker.check_editorial_structure(main, appendix)


def test_hosted_cannot_displace_the_original_method_position(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\label{sec:hosted-concentration}', 'SWAPPED_ROLE')
    main = main.replace(r'\label{sec:method}', r'\label{sec:hosted-concentration}')
    main = main.replace('SWAPPED_ROLE', r'\label{sec:method}')
    with pytest.raises(SystemExit, match='main sections or final boundary are out of order'):
        checker.check_editorial_structure(main, appendix)


def test_new_summary_cannot_replace_an_original_main_panel(manuscript):
    _, main, appendix = manuscript
    main = main.replace('figures/e118_all_scale_factorial_progress.pdf',
                        'figures/replay_factorial_effects.pdf')
    with pytest.raises(SystemExit, match='eight-figure narrative with the hosted summary table'):
        checker.check_editorial_structure(main, appendix)


def test_new_analysis_figure_label_must_be_retained_in_appendix(manuscript):
    _, main, appendix = manuscript
    appendix = appendix.replace(r'\label{fig:replay-factorial-effects}', '')
    with pytest.raises(SystemExit, match='retain its unique appendix label fig:replay-factorial-effects'):
        checker.check_editorial_structure(main, appendix)


def test_hosted_plot_must_retain_its_appendix_label(manuscript):
    _, main, appendix = manuscript
    appendix = appendix.replace(r'\label{fig:hosted-verified-breadth}', '')
    with pytest.raises(SystemExit, match='retain its unique appendix label fig:hosted-verified-breadth'):
        checker.check_editorial_structure(main, appendix)


def test_hosted_summary_must_stay_in_main_hosted_results(manuscript):
    _, main, appendix = manuscript
    include = r'\input{results/hosted_reasoning_off_20260912_main.tex}'
    assert include in main
    main = main.replace(include, '', 1).replace(
        r'\label{sec:method}', r'\label{sec:method}' + include, 1)
    with pytest.raises(SystemExit, match='hosted level-average table must remain in its scientific role'):
        checker.check_editorial_structure(main, appendix)


@pytest.mark.parametrize('copies', (0, 2))
def test_hosted_summary_must_be_compiled_exactly_once(manuscript, copies):
    _, main, appendix = manuscript
    include = r'\input{results/hosted_reasoning_off_20260912_main.tex}'
    main = main.replace(include, include * copies)
    with pytest.raises(SystemExit, match='hosted level-average table must be compiled once in the main'):
        checker.check_editorial_structure(main, appendix)


def test_each_role_needs_its_own_evidence_reference(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\ref{fig:modebench-examples}', 'a missing evidence reference')
    with pytest.raises(SystemExit, match='sec:modebench lacks its evidence reference'):
        checker.check_editorial_structure(main, appendix)


def test_original_opening_figure_cannot_move_after_the_introduction(manuscript):
    _, main, appendix = manuscript
    token = r'\includegraphics[width=\linewidth]{figures/modecollapse_story.pdf}'
    assert token in main
    main = main.replace(token, '', 1).replace(
        r'\label{sec:introduction}', r'\label{sec:introduction}' + token, 1)
    with pytest.raises(SystemExit, match='original opening story figure must remain'):
        checker.check_editorial_structure(main, appendix)


def test_original_figure_cannot_change_sections_while_retaining_figure_order(manuscript):
    _, main, appendix = manuscript
    token = r'\includegraphics[width=\linewidth]{figures/modebench_examples.pdf}'
    assert token in main
    main = main.replace(token, '', 1).replace(
        r'\label{sec:modebench}', token + r'\label{sec:modebench}', 1)
    with pytest.raises(SystemExit, match='original figure modebench_examples must remain in scientific role sec:modebench'):
        checker.check_editorial_structure(main, appendix)


def test_retaining_proof_labels_does_not_hide_changed_mathematics(manuscript):
    text, _, _ = manuscript
    changed = text.replace(r'\end{lemma}', r'\quad 0=1.\end{lemma}', 1)
    with pytest.raises(SystemExit, match='formal statements/proofs changed'):
        checker.check_formal_preservation(changed, checker.PROOF_REFERENCE.read_text())


def test_output_hash_rejects_changed_rendered_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, 'ROOT', tmp_path)
    stem = 'concentration_story'
    directory = tmp_path / 'paper/figures'
    directory.mkdir(parents=True)
    record = {'outputs': {}}
    for extension in ('pdf', 'png'):
        path = directory / f'{stem}.{extension}'
        path.write_bytes(b'bound scientific figure')
        record['outputs'][extension] = {
            'path': str(path.relative_to(tmp_path)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    checker._check_figure_outputs(stem, record)
    (directory / f'{stem}.pdf').write_bytes(b'changed scientific figure')
    with pytest.raises(SystemExit, match='rendered pdf differs'):
        checker._check_figure_outputs(stem, record)


def test_both_rendered_formats_must_be_bound():
    with pytest.raises(SystemExit, match='lacks both rendered-output hash bindings'):
        checker._check_figure_outputs('concentration_story', {'outputs': {'pdf': {}}})


def test_level_construction_cannot_move_outside_the_methods_design_subsection(manuscript):
    _, main, appendix = manuscript
    blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", main, flags=re.DOTALL)
    block = next(block for block in blocks if 'figures/modebench_level_construction.pdf' in block)
    main = main.replace(block, '', 1).replace(
        r'\label{sec:method}', r'\label{sec:method}' + block, 1)
    with pytest.raises(SystemExit, match='original figure modebench_level_construction must remain in scientific role sec:levels-design'):
        checker.check_editorial_structure(main, appendix)


@pytest.mark.parametrize('stem,description', (
    ('hosted_level_averages_20260911.tex', 'original medium-reasoning level-average table'),
    ('hosted_reasoning_off_20260912_appendix.tex', 'matched reasoning-control appendix'),
    ('gpt56_all_levels_sampling_20260912.tex', 'all-level sampling-budget appendix'),
))
@pytest.mark.parametrize('copies', (0, 2))
def test_hosted_original_and_control_appendices_must_be_compiled_once(
        manuscript, stem, description, copies):
    _, main, appendix = manuscript
    include = r'\input{results/' + stem + '}'
    assert include in appendix
    appendix = appendix.replace(include, include * copies)
    with pytest.raises(SystemExit, match=description + ' must be compiled once in the appendix'):
        checker.check_editorial_structure(main, appendix)


def test_temperature_display_is_preserved_in_its_numerical_appendix(manuscript):
    _, main, appendix = manuscript
    assert 'figures/gpt56_temperature_curve.pdf' not in main
    assert appendix.count('figures/gpt56_temperature_curve.pdf') == 1
    assert r'\ref{fig:gpt56-temperature-curve}' in main
    assert 'gpt56_all_levels_sampling_budget' in checker.MAIN_FIGURES
    assert len(checker.MAIN_FIGURES) == 8
    checker.check_editorial_structure(main, appendix)


def test_temperature_appendix_figure_cannot_lose_its_label(manuscript):
    _, main, appendix = manuscript
    appendix = appendix.replace(r'\label{fig:gpt56-temperature-curve}', '')
    with pytest.raises(SystemExit, match='retain its unique appendix label fig:gpt56-temperature-curve'):
        checker.check_editorial_structure(main, appendix)


def test_temperature_figure_must_stay_attached_to_its_numerical_appendix(manuscript):
    _, main, appendix = manuscript
    appendix = appendix.replace('file/after/gpt56_temperature_curve_20260911_appendix.tex',
                                'file/after/unrelated_appendix.tex')
    with pytest.raises(SystemExit, match='temperature figure must remain attached'):
        checker.check_editorial_structure(main, appendix)


def test_hosted_text_must_keep_its_temperature_evidence_reference(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\ref{fig:gpt56-temperature-curve}', 'a missing temperature reference')
    with pytest.raises(SystemExit, match='sec:hosted-concentration lacks its evidence reference fig:gpt56-temperature-curve'):
        checker.check_editorial_structure(main, appendix)


def test_sampling_figure_must_keep_its_numerical_appendix_reference(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\ref{app:gpt56-all-levels-discovery}', 'a missing sampling appendix')
    with pytest.raises(SystemExit, match='hosted sampling figure must retain its numerical-appendix reference'):
        checker.check_editorial_structure(main, appendix)


def test_prior_two_level_sampling_asset_cannot_replace_all_three_levels(manuscript):
    _, main, appendix = manuscript
    main = main.replace('figures/gpt56_all_levels_sampling_budget.pdf',
                        'figures/gpt56_all_domain_sampling_budget.pdf')
    with pytest.raises(SystemExit, match='eight-figure narrative with the hosted summary table'):
        checker.check_editorial_structure(main, appendix)
