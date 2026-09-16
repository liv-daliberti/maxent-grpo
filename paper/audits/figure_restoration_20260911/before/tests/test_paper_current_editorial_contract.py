"""Guard narrative relocation without weakening evidence or rendered assets."""
from __future__ import annotations

import hashlib
import importlib.util
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


def test_current_question_led_roles_and_preserved_math_pass(manuscript):
    text, main, appendix = manuscript
    checker.check_editorial_structure(main, appendix)
    checker.check_formal_preservation(text, checker.PROOF_REFERENCE.read_text())


def test_evidence_roles_do_not_depend_on_heading_wording(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\section{Controlled Replay Comparisons}',
                        r'\section{What does replay change?}')
    main = main.replace(r'\section{Related Work and Discussion}',
                        r'\section{Connections and interpretation}')
    checker.check_editorial_structure(main, appendix)


def test_hosted_cannot_move_after_the_method(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\label{sec:hosted-concentration}', 'SWAPPED_ROLE')
    main = main.replace(r'\label{sec:method}', r'\label{sec:hosted-concentration}')
    main = main.replace('SWAPPED_ROLE', r'\label{sec:method}')
    with pytest.raises(SystemExit, match='main sections or final boundary are out of order'):
        checker.check_editorial_structure(main, appendix)


def test_old_main_figure_cannot_return_as_an_extra_main_panel(manuscript):
    _, main, appendix = manuscript
    main += r'\includegraphics{figures/modecollapse_story.pdf}'
    with pytest.raises(SystemExit, match='five-figure scientific narrative'):
        checker.check_editorial_structure(main, appendix)


def test_relocated_figure_label_must_be_retained(manuscript):
    _, main, appendix = manuscript
    appendix = appendix.replace(r'\label{fig:maxrl-factorial}', '')
    with pytest.raises(SystemExit, match='retain its unique appendix label fig:maxrl-factorial'):
        checker.check_editorial_structure(main, appendix)


def test_each_role_needs_its_own_evidence_reference(manuscript):
    _, main, appendix = manuscript
    main = main.replace(r'\ref{fig:replay-key-weighting}', 'a missing evidence reference')
    with pytest.raises(SystemExit, match='sec:key-weighting lacks its evidence reference'):
        checker.check_editorial_structure(main, appendix)


def test_first_figure_cannot_precede_measurement_definitions(manuscript):
    _, main, appendix = manuscript
    token = r'\includegraphics[width=\linewidth]{figures/concentration_story.pdf}'
    assert token in main
    main = main.replace(token, '', 1).replace(
        r'\label{sec:modebench}', r'\label{sec:modebench}' + token, 1)
    with pytest.raises(SystemExit, match='first main figure must follow benchmark measurement'):
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
