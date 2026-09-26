"""Independent finite-difference and RK4 checks of the weighted categorical flow.

Uses only Python's standard library. Writes its sibling JSON audit, never
manuscripts or experimental state. Numerical checks supplement the proof;
they do not establish asymptotic convergence or a neural-network guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def softmax(z):
    values = [math.exp(x - max(z)) for x in z]
    return [x / sum(values) for x in values]


def run_case(lengths, initial, objective):
    k = len(lengths)
    inverse_lengths = [1 / length for length in lengths]
    amplitude = sum(inverse_lengths) / k
    weights = [v / sum(inverse_lengths) for v in inverse_lengths]
    rho_lm = 1.0
    rho_w = rho_lm * amplitude
    target = weights + [0.0]
    z = [math.log(p) for p in initial]

    def coefficient(p):
        return 15 / 16 if objective == "drgrpo" else sum((1 - p) ** j for j in range(15))

    def potential(p):
        return 15 * p / 16 if objective == "drgrpo" else sum((1 - (1 - p) ** j) / j for j in range(1, 16))

    def quantities(state):
        p = softmax(state)
        correct = sum(p[:k])
        replay = -sum(w * math.log(v) for w, v in zip(weights, p))
        energy = rho_w * replay - potential(correct)
        grad_correct = [v * ((i < k) - correct) for i, v in enumerate(p)]
        velocity = [coefficient(correct) * g - rho_w * (v - w)
                    for g, v, w in zip(grad_correct, p, target)]
        return p, correct, replay, energy, velocity

    p0, correct0, replay0, energy0, velocity0 = quantities(z)
    length_loss = -sum(math.log(p0[i]) / lengths[i] for i in range(k)) / k
    loss_identity_error = abs(rho_lm * length_loss - rho_w * replay0)
    eps = 1e-5
    fd_errors = []
    for i in range(len(z)):
        plus, minus = z.copy(), z.copy()
        plus[i] += eps
        minus[i] -= eps
        derivative = (quantities(plus)[3] - quantities(minus)[3]) / (2 * eps)
        fd_errors.append(abs(derivative + velocity0[i]))
    constant = replay0 + (potential(1.0) - potential(correct0)) / rho_w
    log_bound = [-constant / w for w in weights]
    dt, steps = 0.1, 30000
    previous_energy = energy0
    max_energy_increase = 0.0
    min_log_bound_margin = math.inf
    min_probabilities = p0[:k].copy()
    for _ in range(steps):
        v1 = quantities(z)[4]
        v2 = quantities([x + dt * v / 2 for x, v in zip(z, v1)])[4]
        v3 = quantities([x + dt * v / 2 for x, v in zip(z, v2)])[4]
        v4 = quantities([x + dt * v for x, v in zip(z, v3)])[4]
        z = [x + dt * (a + 2 * b + 2 * c + d) / 6 for x, a, b, c, d in zip(z, v1, v2, v3, v4)]
        p, correct, replay, energy, _ = quantities(z)
        max_energy_increase = max(max_energy_increase, energy - previous_energy)
        previous_energy = energy
        for i in range(k):
            min_probabilities[i] = min(min_probabilities[i], p[i])
            min_log_bound_margin = min(min_log_bound_margin, math.log(p[i]) - log_bound[i])
    conditional = [v / correct for v in p[:k]]
    result = {
        "objective": objective, "lengths": lengths, "initial_probabilities": initial,
        "inverse_length_target": weights, "amplitude_A": amplitude, "rho_lm": rho_lm,
        "rho_w": rho_w, "loss_identity_absolute_error": loss_identity_error,
        "finite_difference_max_gradient_error": max(fd_errors),
        "integrator": "fixed-step classical RK4", "dt": dt, "steps": steps,
        "final_time": steps * dt, "final_correct_mass": correct,
        "final_conditional_distribution": conditional,
        "max_conditional_target_error": max(abs(a - b) for a, b in zip(conditional, weights)),
        "minimum_observed_banked_probabilities": min_probabilities,
        "theorem_probability_lower_bounds": [math.exp(v) for v in log_bound],
        "minimum_observed_log_bound_margin": min_log_bound_margin,
        "maximum_observed_energy_increase": max_energy_increase,
    }
    assert loss_identity_error < 1e-12
    assert max(fd_errors) < 1e-8
    assert min_log_bound_margin >= -1e-10
    assert max_energy_increase < 1e-10
    assert correct > 0.999
    assert result["max_conditional_target_error"] < 0.001
    return result


def main():
    cases = [run_case([1, 2], [.08, .72, .20], "drgrpo"),
             run_case([1, 2], [.65, .05, .30], "maxrl"),
             run_case([1, 1], [.08, .72, .20], "drgrpo")]
    raw, nu = .1 * 15 / 256, 1 / 384
    frequency = {"raw_coefficient": raw, "fresh_frequency": nu,
                 "equal_replay_frequency": nu, "equal_frequency_effective_coefficient": raw * nu / nu,
                 "double_replay_frequency": 1 / 192, "double_frequency_effective_coefficient": raw * (1 / 192) / nu}
    assert frequency["equal_frequency_effective_coefficient"] == raw
    assert frequency["double_frequency_effective_coefficient"] == 2 * raw
    output = {"schema": "weighted-categorical-replay-independent-numerical-audit-v1",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Finite-difference, algebraic scaling, and finite-horizon ODE checks only; no empirical result changes and no proof by simulation.",
              "cases": cases, "relative_frequency_checks": frequency, "all_checks_passed": True}
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
