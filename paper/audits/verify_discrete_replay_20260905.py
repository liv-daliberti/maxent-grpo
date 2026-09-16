"""Standard-library checks of fixed-step deterministic categorical replay GD.

These finite samples supplement a separate proof. They do not establish a
uniform Hessian bound or asymptotic convergence, and say nothing about SGD,
Adam, neural parameter sharing, changing banks/weights, or replay schedules.
Run: python paper/audits/verify_discrete_replay_20260905.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def probabilities(z):
    largest = max(z)
    log_normalizer = largest + math.log(sum(math.exp(v - largest) for v in z))
    logp = [v - log_normalizer for v in z]
    return [math.exp(v) for v in logp], logp


class Objective:
    def __init__(self, method, group_size, target, correct, rho):
        self.method, self.group_size = method, group_size
        self.target, self.correct, self.rho = target, correct, rho
        self.m1 = (group_size - 1) / group_size if method == 'DrGRPO' else group_size - 1
        self.m2 = 0.0 if method == 'DrGRPO' else (group_size - 1) * (group_size - 2) / 2
        self.bound = rho / 2 + self.m1 / 2 + self.m2 / 8

    def potential(self, mass):
        if self.method == 'DrGRPO':
            return self.m1 * mass, self.m1, 0.0
        powers = [(1 - mass) ** j for j in range(self.group_size)]
        psi = sum((1 - powers[k]) / k for k in range(1, self.group_size))
        first = sum(powers[:self.group_size - 1])
        second = -sum(j * powers[j - 1] for j in range(1, self.group_size - 1))
        return psi, first, second

    def evaluate(self, z):
        p, logp = probabilities(z)
        mass = sum(v for v, valid in zip(p, self.correct) if valid)
        psi, first, second = self.potential(mass)
        replay = -dot(self.target, logp)
        mass_gradient = [v * (valid - mass) for v, valid in zip(p, self.correct)]
        gradient = [self.rho * (v - w) - first * a
                    for v, w, a in zip(p, self.target, mass_gradient)]
        return {'p': p, 'logp': logp, 'mass': mass, 'replay': replay,
                'energy': self.rho * replay - psi, 'gradient': gradient,
                'mass_gradient': mass_gradient, 'first': first, 'second': second}

    def hessian_vector(self, z, direction):
        q = self.evaluate(z)
        p = q['p']
        jacobian_v = [v * (d - dot(p, direction)) for v, d in zip(p, direction)]
        a_dot_v = dot(q['mass_gradient'], direction)
        hessian_mass_v = [j * (valid - q['mass']) - v * a_dot_v
                          for j, valid, v in zip(jacobian_v, self.correct, p)]
        return [self.rho * j - q['second'] * a_dot_v * a - q['first'] * h
                for j, a, h in zip(jacobian_v, q['mass_gradient'], hessian_mass_v)]


def derivative_checks(obj, seed):
    rng = random.Random(seed)
    gradient_error = hvp_error = maximum_hvp_norm = 0.0
    eps = 1e-5
    states = [[0.0] * 5,
              [-12.0, -12.0, -12.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, -12.0, -12.0]]
    states += [[rng.uniform(-10, 10) for _ in range(5)] for _ in range(32)]
    for state in states:
        q = obj.evaluate(state)
        for i in range(5):
            plus, minus = state.copy(), state.copy()
            plus[i] += eps
            minus[i] -= eps
            fd = (obj.evaluate(plus)['energy'] - obj.evaluate(minus)['energy']) / (2 * eps)
            gradient_error = max(gradient_error, abs(fd - q['gradient'][i]))
        for _ in range(4):
            direction = [rng.gauss(0, 1) for _ in range(5)]
            direction = [v / norm(direction) for v in direction]
            hvp = obj.hessian_vector(state, direction)
            plus = obj.evaluate([z + eps * v for z, v in zip(state, direction)])['gradient']
            minus = obj.evaluate([z - eps * v for z, v in zip(state, direction)])['gradient']
            fd = [(a - b) / (2 * eps) for a, b in zip(plus, minus)]
            hvp_error = max(hvp_error, norm([a - b for a, b in zip(hvp, fd)]))
            maximum_hvp_norm = max(maximum_hvp_norm, norm(hvp))
            assert norm(hvp) <= obj.bound + 1e-11
    assert gradient_error < 2e-8
    assert hvp_error < 2e-8
    return {'states': len(states), 'unit_directions_per_state': 4,
            'max_gradient_finite_difference_error': gradient_error,
            'max_hessian_vector_finite_difference_error': hvp_error,
            'max_sampled_hessian_vector_norm': maximum_hvp_norm,
            'claimed_global_smoothness_bound': obj.bound,
            'max_sampled_hessian_to_bound_ratio': maximum_hvp_norm / obj.bound}


def trajectory(obj, initial, fraction, steps=20000):
    eta = fraction * 2 / obj.bound
    z = [math.log(v) for v in initial]
    q = obj.evaluate(z)
    bound_constant = q['replay'] + (obj.potential(1.0)[0] - obj.potential(q['mass'])[0]) / obj.rho
    bank = [i for i, w in enumerate(obj.target) if w > 0]
    log_floors = {i: -bound_constant / obj.target[i] for i in bank}
    minimum_log_margin = min(q['logp'][i] - log_floors[i] for i in bank)
    max_descent_excess = max_energy_increase = 0.0
    checkpoints = []
    for step in range(steps + 1):
        if step in {0, 10, 100, 1000, 10000, steps}:
            checkpoints.append({'step': step, 'effective_time': step * eta,
                                'correct_mass': q['mass'], 'probabilities': q['p'],
                                'target_l1_error': sum(abs(v - w) for v, w in zip(q['p'], obj.target)),
                                'gradient_norm': norm(q['gradient']), 'energy': q['energy']})
        if step == steps:
            break
        next_z = [v - eta * g for v, g in zip(z, q['gradient'])]
        next_q = obj.evaluate(next_z)
        promised_drop = eta * (1 - obj.bound * eta / 2) * dot(q['gradient'], q['gradient'])
        excess = next_q['energy'] - q['energy'] + promised_drop
        max_descent_excess = max(max_descent_excess, excess)
        max_energy_increase = max(max_energy_increase, next_q['energy'] - q['energy'])
        assert excess <= 2e-11 * (1 + abs(q['energy']))
        for i in bank:
            minimum_log_margin = min(minimum_log_margin, next_q['logp'][i] - log_floors[i])
        assert minimum_log_margin >= -1e-10
        z, q = next_z, next_q
    return {'step_fraction_of_2_over_L': fraction, 'eta': eta, 'steps': steps,
            'max_descent_inequality_excess': max_descent_excess,
            'max_energy_increase': max_energy_increase,
            'replay_upper_bound_constant': bound_constant,
            'bank_probability_lower_bounds': {str(i): math.exp(log_floors[i]) for i in bank},
            'minimum_observed_log_retention_margin': minimum_log_margin,
            'checkpoints': checkpoints,
            'finite_horizon_target_error_reduced': checkpoints[-1]['target_l1_error'] < checkpoints[0]['target_l1_error'],
            'convergence_scope': 'Finite-horizon values only; exact asymptotic convergence is not tested.'}


def main():
    cases = []
    initial = [.02, .17, .06, .25, .50]
    for method in ['DrGRPO', 'implemented_MaxRL']:
        for group_size in [2, 3, 16]:
            for bank_name, target in [('full_correct_bank', [.5, .3, .2, 0., 0.]),
                                       ('strict_correct_subset', [.7, 0., .3, 0., 0.])]:
                obj = Objective(method, group_size, target, [True, True, True, False, False], rho=.8)
                cases.append({'method': method, 'G': group_size, 'bank': bank_name,
                              'correct_indices': [0, 1, 2], 'target_extended_with_zeros': target,
                              'rho': obj.rho, 'M1': obj.m1, 'M2': obj.m2, 'L': obj.bound,
                              'derivative_checks': derivative_checks(obj, 20260905 + group_size),
                              'trajectories': [trajectory(obj, initial, fraction) for fraction in [.1, .5, .95]]})
    assert all(t['finite_horizon_target_error_reduced'] for c in cases for t in c['trajectories'])
    output = {'schema': 'discrete-weighted-categorical-replay-numerical-audit-v1',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Independent fixed-step deterministic exact-gradient checks in categorical logits only. Finite numerical samples do not prove a global bound or asymptotic convergence. No SGD, Adam, or neural-network claim.',
              'objective': 'F(z)=rho*(-sum_b w_b log softmax(z)_b)-Psi(P(z))',
              'implemented_MaxRL_potential': 'sum_{k=1}^{G-1} [1-(1-P)^k]/k',
              'smoothness_bound': 'L=rho/2+M1/2+M2/8',
              'descent_inequality': 'F(z_next)<=F(z)-eta*(1-L*eta/2)*||grad F(z)||^2',
              'case_count': len(cases), 'trajectory_count': sum(len(c['trajectories']) for c in cases),
              'all_checks_passed': True, 'cases': cases}
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    trajectories = [t for c in cases for t in c['trajectories']]
    print(json.dumps({'all_checks_passed': True, 'cases': len(cases), 'trajectories': len(trajectories),
                      'max_gradient_error': max(c['derivative_checks']['max_gradient_finite_difference_error'] for c in cases),
                      'max_hvp_error': max(c['derivative_checks']['max_hessian_vector_finite_difference_error'] for c in cases),
                      'max_sampled_hessian_bound_ratio': max(c['derivative_checks']['max_sampled_hessian_to_bound_ratio'] for c in cases),
                      'max_descent_excess': max(t['max_descent_inequality_excess'] for t in trajectories),
                      'minimum_log_retention_margin': min(t['minimum_observed_log_retention_margin'] for t in trajectories),
                      'final_target_l1_error_range': [min(t['checkpoints'][-1]['target_l1_error'] for t in trajectories), max(t['checkpoints'][-1]['target_l1_error'] for t in trajectories)]}, indent=2))


if __name__ == '__main__':
    main()
