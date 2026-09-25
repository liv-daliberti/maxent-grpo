"""Independent finite checks of source specializations, not proofs or run evidence."""
import itertools
import json
import math
from pathlib import Path

import numpy as np


def softmax(z):
    v = np.exp(z - np.max(z))
    return v / v.sum()


def binary_kl(a, r):
    return a * math.log(a / r) + (1 - a) * math.log((1 - a) / (1 - r))


def lower_root(a, budget):
    entropy = -a * math.log(a) - (1 - a) * math.log1p(-a)
    lo, hi = -(budget + entropy) / a, math.log(a)
    for _ in range(180):
        mid = (lo + hi) / 2
        divergence = a * (math.log(a) - mid) + (1 - a) * (
            math.log1p(-a) - math.log1p(-math.exp(mid))
        )
        if divergence > budget:
            lo = mid
        else:
            hi = mid
    return math.exp(hi)


def coverage_tails(p, draws):
    # Category zero is failure. Enumerate every response group exactly.
    mass = np.zeros(len(p))
    for group in itertools.product(range(len(p)), repeat=draws):
        distinct = len(set(group) - {0})
        mass[distinct] += math.prod(p[j] for j in group)
    return np.array([sum(mass[r:]) for r in range(1, len(p))])


def main():
    rng = np.random.default_rng(20260921)
    out = {}
    max_gradient_error = 0.0
    for _ in range(60):
        # Shared nonlinear parameters; several response strings per key.
        W = rng.normal(size=(9, 4))
        V = rng.normal(size=(9, 4))
        theta = rng.normal(size=4)
        keys = np.array([0, 0, 0, 1, 1, 2, -1, -1, -1])

        def policy(t):
            return softmax(W @ t + 0.2 * np.sin(V @ t))

        def metric(t):
            p = policy(t)
            q = np.array([p[keys == c].sum() for c in range(3)])
            q /= q.sum()
            return 1 - q @ q

        p = policy(theta)
        jac = W + 0.2 * np.cos(V @ theta)[:, None] * V
        score = jac - p @ jac
        correct = keys >= 0
        q = np.array([p[keys == c].sum() for c in range(3)])
        q /= q.sum()
        coefficients = 2 * p[correct] / p[correct].sum() * (
            q @ q - q[keys[correct]]
        )
        imported = coefficients @ score[correct]
        eps = 1e-5
        numeric = np.array([
            (metric(theta + eps * e) - metric(theta - eps * e)) / (2 * eps)
            for e in np.eye(4)
        ])
        max_gradient_error = max(max_gradient_error, np.max(abs(numeric - imported)))
    assert max_gradient_error < 1e-8
    out['setpo_neural_gradient_60_nonlinear_cases_max_error'] = max_gradient_error

    max_tightness_error = 0.0
    for _ in range(200):
        k = int(rng.integers(2, 8))
        w = rng.dirichlet(np.full(k, 3.0))
        p = rng.dirichlet(np.full(k + 2, 2.0))
        excess = sum(w * np.log(w / p[:k]))
        assert p[:k].sum() >= math.exp(-excess) - 1e-12
        for b in range(k):
            floor = lower_root(w[b], excess)
            assert p[b] >= floor - 1e-12
            assert binary_kl(w[b], p[b]) <= excess + 1e-12
            tight = (1 - floor) * w / (1 - w[b])
            tight[b] = floor
            tight_kl = sum(w * np.log(w / tight))
            max_tightness_error = max(max_tightness_error, abs(tight_kl - excess))
    assert max_tightness_error < 1e-9
    out['sharp_retention_200_laws_max_attainment_error'] = max_tightness_error
    out['illustrative_uniform_16'] = {
        'excess_cross_entropy': 0.01,
        'sharp_floor': lower_root(1 / 16, 0.01),
        'old_floor': math.exp(-16 * (math.log(16) + 0.01)),
        'bank_mass_floor': math.exp(-0.01),
    }

    max_update_error = max_gap_error = 0.0
    for _ in range(100):
        k = int(rng.integers(3, 10))
        z = rng.normal(size=k)
        p = softmax(z)
        mu = rng.dirichlet(np.full(k, 2.0))
        reward = np.arange(k) < k // 2
        alpha, beta = 15 / 16, float(rng.uniform(0.1, 1.0))
        eta = float(rng.uniform(0.05, 1.0)) / beta
        kl = sum(p * np.log(p / mu))
        original = alpha * p * (reward - p @ reward) - beta * p * (np.log(p / mu) - kl)
        shifted = alpha * reward + beta * np.log(mu)
        B = max(1.0, np.ptp(shifted))
        normalized_reward = (shifted - min(shifted)) / B
        residual = normalized_reward - (beta / B) * np.log(p)
        imported = p * (residual - p @ residual)
        max_update_error = max(max_update_error, np.max(abs(eta * original - eta * B * imported)))
        optimum = softmax(shifted / beta)
        objective = lambda x: alpha * (x @ reward) - beta * sum(x * np.log(x / mu))
        direct_gap = objective(optimum) - objective(p)
        kl_gap = beta * sum(p * np.log(p / optimum))
        max_gap_error = max(max_gap_error, abs(direct_gap - kl_gap))
    assert max_update_error < 1e-12 and max_gap_error < 1e-12
    out['mei_reward_reduction_100_cases'] = {
        'max_update_error': max_update_error, 'max_KL_gap_error': max_gap_error,
    }

    count = 0
    for P in (0.2, 0.7, 1.0):
        for q in ([0.8, 0.1, 0.1], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]):
            p = np.r_[1 - P, P * np.array(q)]
            uniform = np.r_[1 - P, np.full(3, P / 3)]
            for K in range(1, 7):
                assert np.all(coverage_tails(uniform, K) >= coverage_tails(p, K) - 1e-12)
                count += 1
    out['coupon_tail_dominance_exhaustive_group_cases'] = count
    out['all_checks_passed'] = True
    print(json.dumps(out, indent=2))
    Path(__file__).with_name('import_checks.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
