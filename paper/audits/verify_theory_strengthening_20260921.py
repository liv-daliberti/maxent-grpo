"""Small deterministic checks of proposed theory; not neural training evidence.

Run: python paper/audits/verify_theory_strengthening_20260921.py
Uses exact enumeration of finite groups, not Monte Carlo.
"""

import itertools
import json
import math

import numpy as np


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def pcmd(p, indices):
    q = p[indices] / p[indices].sum()
    return 1 - q @ q


def enumerate_groups(p, m, group_size, method):
    gradients, weights = [], []
    for group in itertools.product(range(len(p)), repeat=group_size):
        r = np.array([j < m for j in group], dtype=float)
        count = r.sum()
        if method == "dr":
            a = r - count / group_size
        else:
            a = group_size * r / count - 1 if count else np.zeros(group_size)
        g = sum(ai * (np.eye(len(p))[j] - p) for ai, j in zip(a, group)) / group_size
        gradients.append(g)
        weights.append(math.prod(p[j] for j in group))
    return np.array(gradients), np.array(weights)


def main():
    out = {}
    rng = np.random.default_rng(42)
    max_drift_error = 0.0
    for _ in range(100):
        p = rng.dirichlet(np.ones(6))
        z = np.log(p)
        correct = np.arange(4)
        P = p[correct].sum()
        c, rho = 0.8, float(rng.uniform(0, 2))
        gp = p * (np.arange(6) < 4) - P * p
        for bank in (np.arange(4), np.array([0, 2, 3])):
            w = np.zeros(6)
            w[bank] = 1 / len(bank)
            velocity = c * gp + rho * (w - p)
            M = p[bank].sum()
            r = p[bank] / M
            variance = np.sum(r**3) - np.sum(r**2) ** 2
            predicted = 2 * M * (rho - c * (1 - P)) * variance
            eps = 1e-5
            actual = (pcmd(softmax(z + eps * velocity), bank) - pcmd(softmax(z - eps * velocity), bank)) / (2 * eps)
            max_drift_error = max(max_drift_error, abs(actual - predicted))
    assert max_drift_error < 1e-8
    out["pcmd_drift_max_absolute_error_200_cases"] = max_drift_error

    m, G, P = 3, 4, 0.6
    p = np.array([P / m] * m + [1 - P])
    z = np.log(p)
    stochastic = {}
    for method in ("dr", "maxrl"):
        gs, weights = enumerate_groups(p, m, G, method)
        mean = weights @ gs
        c = (G - 1) / G if method == "dr" else (1 - (1 - P) ** (G - 1)) / P
        gp = p * (np.arange(len(p)) < m) - P * p
        assert np.max(np.abs(mean - c * gp)) < 1e-12
        coefficient = 0.0
        for r in range(G + 1):
            pr = math.comb(G, r) * P**r * (1 - P) ** (G - r)
            a1 = 1 - r / G if method == "dr" else (G / r - 1 if r else 0)
            coefficient += pr * r * a1**2 / G**2
        coefficient *= (m - 1) / m**3
        ratios = []
        for eta in (0.01, 0.005, 0.002):
            expected = sum(weight * pcmd(softmax(z + eta * g), np.arange(m)) for weight, g in zip(weights, gs))
            loss = (1 - 1 / m) - expected
            assert loss > 0
            ratios.append(loss / eta**2)
        assert abs(ratios[-1] / coefficient - 1) < 0.002
        stochastic[method] = {"predicted_quadratic_coefficient": coefficient, "enumerated_loss_over_eta_squared": ratios}
    out["uniform_start_stochastic_symmetry_breaking"] = stochastic

    # Covariance decomposition for nonuniform p and a general linear logit map.
    p = np.array([0.13, 0.27, 0.21, 0.39])
    m, G = 2, 3
    P = p[:m].sum()
    J = rng.normal(size=(4, 3))
    scores = (np.eye(4) - p) @ J
    v = sum(p[j] * scores[j] for j in range(m))
    means = [sum(p[j] * scores[j] for j in range(m, 4)) / (1 - P), v / P]
    covs = []
    for r, ix in ((0, range(m, 4)), (1, range(m))):
        mass = P if r else 1 - P
        covs.append(sum(p[j] * np.outer(scores[j] - means[r], scores[j] - means[r]) for j in ix) / mass)
    max_cov_error = 0.0
    for method in ("dr", "maxrl"):
        gz, weights = enumerate_groups(p, m, G, method)
        gs = gz @ J
        mean = weights @ gs
        cov = sum(w * np.outer(g - mean, g - mean) for w, g in zip(weights, gs))
        betas, probs = [], []
        within = np.zeros((3, 3))
        for r in range(G + 1):
            pr = math.comb(G, r) * P**r * (1 - P) ** (G - r)
            if method == "dr":
                a1, a0 = 1 - r / G, -r / G
            else:
                a1, a0 = (G / r - 1, -1) if r else (0, 0)
            betas.append((r * a1 / P - (G - r) * a0 / (1 - P)) / G)
            probs.append(pr)
            within += pr * (r * a1**2 * covs[1] + (G - r) * a0**2 * covs[0]) / G**2
        betas, probs = np.array(betas), np.array(probs)
        predicted = within + (probs @ betas**2 - (probs @ betas)**2) * np.outer(v, v)
        max_cov_error = max(max_cov_error, float(np.max(np.abs(cov - predicted))))
    assert max_cov_error < 1e-12
    out["neural_score_covariance_max_absolute_error"] = max_cov_error

    # Counterexample to retention constant for arbitrary signed fresh objectives.
    rho, kappa, k = 0.2, 1.0, 2
    P = rho / kappa
    p_b = P / k
    claimed_C = -math.log(p_b) - kappa * (1 - P) / rho
    claimed_floor = math.exp(-k * claimed_C)
    assert claimed_floor > 1
    out["signed_advantage_counterexample"] = {"actual_stationary_banked_probability": p_b, "superseded_formula_claimed_floor": claimed_floor}

    # For binary KL recovery, dot p = 2 beta p^2 (1-p)^2 log((1-p)/p).
    # Coefficient tends to 2, invalidating a leading coefficient of 1.
    out["two_mode_KL_velocity_coefficient"] = [(2 * s**2 * (1 - s)**2 * math.log((1 - s) / s)) / (s**2 * math.log(1 / s)) for s in (1e-3, 1e-5, 1e-7)]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
