#!/usr/bin/env python3
"""Verify the hypothetical worked numbers in appendix sections A9--A11.

These checks validate arithmetic and retained formal text, not neural training.
Only Python's standard library is required.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from decimal import Decimal, localcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {
    'exemplar_bridge_section.tex': 'app:theory-exemplar-bridge',
    'optimizer_section.tex': 'app:theory-optimizer',
    'discovery_section.tex': 'app:theory-discovery-admission',
}


def close(actual: float, expected: float, *, rel: float = 1e-12) -> None:
    assert math.isclose(actual, expected, rel_tol=rel, abs_tol=1e-15), (actual, expected)


def main() -> dict:
    results: dict = {}
    with localcontext() as ctx:
        ctx.prec = 50
        half = Decimal('0.5')
        sequence = half ** 4
        mean_score = half.ln()
        geometric_mean = mean_score.exp()
        key_floor = sequence + Decimal(1) / 32
        energy_floor = (-(Decimal(4) * Decimal(2).ln())).exp()
        assert sequence == Decimal(1) / 16
        close(float(geometric_mean), 0.5)
        close(float(energy_floor), float(sequence))
        assert key_floor == Decimal(3) / 32
        results['complete_response'] = {
            'scored_tokens_including_termination': 4,
            'conditional_token_probability': 0.5,
            'mean_log_likelihood': float(mean_score),
            'exp_mean_log_likelihood': float(geometric_mean),
            'complete_response_probability': float(sequence),
            'energy_D': float(Decimal(2).ln()),
            'energy_floor_for_alpha_1_L_4': float(energy_floor),
            'two_disjoint_response_key_floor': float(key_floor),
        }
        appearance8 = 1 - (Decimal(15) / 16) ** 8
        close(float(appearance8), 0.403, rel=0.001)
        visibility_budget = math.ceil(16 * math.log(4 / 0.05))
        assert visibility_budget == 71
        assert 4 * math.exp(-visibility_budget / 16) <= 0.05
        assert 4 * math.exp(-(visibility_budget - 1) / 16) > 0.05
        results['finite_visibility'] = {
            'single_key_probability': 1 / 16,
            'appearance_probability_in_8_draws': float(appearance8),
            'number_of_protected_keys': 4,
            'sufficient_95_percent_draw_budget': visibility_budget,
            'exponential_union_bound_at_71': 4 * math.exp(-71 / 16),
            'exact_union_bound_at_71': 4 * (15 / 16) ** 71,
        }
        # F = -log sigmoid(theta); F'' = p(1-p) <= 1/4 and |F'| <= 1.
        L_F, M, G, eta, s, delta = map(Decimal, ['0.25', '1', '1', '0.1', '0.1', '0.05'])
        N = 100
        assert eta * L_F * M <= 1
        d = L_F * eta ** 2 * s ** 2 / 2
        assert d == Decimal(1) / 80000
        B = N * d
        assert B == Decimal('0.00125')
        D0 = Decimal(2).ln()
        basic = (-(D0 + B) / delta).exp()
        V = N * (eta * M * G * s) ** 2
        Q = N * L_F * M ** 2 * eta ** 2 * s ** 2 / 2
        assert V == Decimal('0.01')
        assert Q == B
        sharp_energy = D0 + Q + (2 * V * (1 / delta).ln()).sqrt()
        sharp = (-sharp_energy).exp()
        close(float(basic), 9.30e-7, rel=0.001)
        close(float(sharp), 0.391, rel=0.001)
        assert sharp > basic
        B_infinity = math.pi ** 2 / 480000
        close(B_infinity, float(d) * math.pi ** 2 / 6)
        close(B_infinity, 2.06e-5, rel=0.003)
        results['toy_optimizer'] = {
            'objective': '-log(sigmoid(theta)); one complete Bernoulli response; J=0',
            'initial_theta': 0,
            'initial_energy': float(D0),
            'global_smoothness_bound': float(L_F),
            'gradient_norm_bound': float(G),
            'preconditioner': float(M),
            'noise': 'independent centered Rademacher +/- 0.1',
            'constant_step': float(eta),
            'confidence': 0.95,
            'step_noise_budget': float(d),
            'B_100_constant_steps': float(B),
            'B_infinity_constant_steps': 'diverges',
            'B_infinity_steps_0_1_over_t_plus_1': B_infinity,
            'basic_floor_through_100_updates': float(basic),
            'V_100': float(V),
            'Q_100': float(Q),
            'sharp_energy_ceiling': float(sharp_energy),
            'sharp_floor_through_100_updates': float(sharp),
        }
        hazard = Decimal('0.04') * Decimal('0.5')
        assert hazard == Decimal('0.02')
        admission_budget = math.ceil(math.log(4 / 0.05) / float(hazard))
        assert admission_budget == 220
        failure = Decimal(4) * (-hazard * admission_budget).exp()
        assert failure < Decimal('0.05')
        assert 4 * math.exp(-float(hazard) * (admission_budget - 1)) > 0.05
        assert 2 < 4  # Two non-evicting slots cannot contain four distinct keys.
        results['admission'] = {
            'number_of_possible_correct_keys': 4,
            'per_missing_key_generation_floor': 0.04,
            'conditional_insertion_floor': 0.5,
            'actual_admission_hazard_floor': float(hazard),
            'sufficient_95_percent_opportunity_count': admission_budget,
            'failure_union_bound_at_220': float(failure),
            'two_slot_capacity_covers_four_keys': False,
        }
        rho = Decimal('0.3')
        R_old = Decimal(4).ln()
        surprise_new = Decimal(16).ln()
        R_new = (2 * R_old + surprise_new) / 3
        delta_F = rho / 3 * (surprise_new - R_old)
        close(float(delta_F), 0.1386, rel=0.001)
        close(float(delta_F), float(rho * (R_new - R_old)))
        assert rho / 2 == Decimal('0.15')
        assert rho / 3 == Decimal('0.10')
        assert Decimal(2) / 4 + Decimal(1) / 16 < 1
        results['switching_cost'] = {
            'replay_dose': float(rho),
            'old_key_probabilities': [0.25, 0.25],
            'new_key_probability': 1 / 16,
            'old_mean_surprisal': float(R_old),
            'new_mean_surprisal': float(R_new),
            'energy_jump': float(delta_F),
            'old_coefficient': 0.15,
            'new_coefficient': 0.10,
            'coefficient_floor_at_capacity_3': 0.10,
        }
    source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in FILES}
    manuscript = (HERE.parents[1] / 'main.tex').read_text()
    preserved = {}
    for name, label in FILES.items():
        text = (HERE / name).read_text()
        pos = manuscript.index(r'\label{' + label + '}')
        start = manuscript.rfind(r'\subsection{', 0, pos)
        end = manuscript.index(r'\subsection{', pos)
        baseline = manuscript[start:end]
        formal_pattern = r'\\begin\{(lemma|theorem|corollary|proof)\}([\s\S]*?)\\end\{\1\}'
        citation_pattern = r'\\cite\w*(?:\[[^\]]*\])*\{[^}]+\}'
        assert re.findall(formal_pattern, text) == re.findall(formal_pattern, baseline), name
        assert re.findall(citation_pattern, text) == re.findall(citation_pattern, baseline), name
        preserved[name] = {'formal_blocks': len(re.findall(formal_pattern, text)),
                           'all_existing_citations_preserved': True}
    results['formal_preservation'] = preserved
    output = {'status': 'PASS', 'scope': 'Hypothetical examples only; no empirical measurements or optimizer certification.',
              'arithmetic_precision_decimal_digits': 50, 'results': results, 'staging_sha256': source_hashes}
    (HERE / 'verify_optimizer_examples.json').write_text(json.dumps(output, indent=2) + '\n')
    return output


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
