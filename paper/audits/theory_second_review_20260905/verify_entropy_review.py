"""Standard-library checks for the entropy review; numerical checks are not proofs."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/oat_drgrpo/semantic_shannon.py"


def softmax(z):
    values = [math.exp(x - max(z)) for x in z]
    return [x / sum(values) for x in values]


def quantities(z, correct_count, coefficient, group_size, kind):
    p = softmax(z)
    P = sum(p[:correct_count])
    if kind == "DrGRPO":
        c = (group_size - 1) / group_size
        potential = c * P
    else:
        c = sum((1 - P) ** k for k in range(group_size - 1))
        potential = sum((1 - (1 - P) ** k) / k for k in range(1, group_size))
    entropy = -sum(x * math.log(x) for x in p)
    gradient = [
        x * (c * (float(i < correct_count) - P)
             + coefficient * (-math.log(x) - entropy))
        for i, x in enumerate(p)
    ]
    return potential + coefficient * entropy, gradient, p


def main():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"OpenSetSemanticSignal", "open_set_success_semantic_signal"}]
    namespace = {"math": math, "dataclass": dataclasses.dataclass}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    signal = namespace["open_set_success_semantic_signal"]

    # Exact current source, one observed verified key and a structural unseen bucket.
    singleton = []
    for count in [1, 2, 16, 128, 10000]:
        actual = .1 * signal(explicit_counts={"a": count}, sampled_key="a").centered_clipped_surprisal
        u, v = (count + 1) / (count + 2), 1 / (count + 2)
        expected = -.1 * v * (min(math.log(count + 2), 5) + math.log(u)) / 5
        assert actual < 0 and abs(actual - expected) < 1e-15
        singleton.append({"history_and_peer_count": count, "source_advantage": actual,
                          "formula_advantage": expected})

    group_checks = []
    for P, history_count, G in itertools.product([.01, .3, .9, .999], [1, 16, 10000], [2, 3, 16]):
        # A single correct category and a single incorrect category.  H(q)=0 identically.
        multiplier = sum(math.comb(G, k) * P**k * (1-P)**(G-k) * k/G * .1 *
                         signal(explicit_counts={"a": history_count+k-1}, sampled_key="a").centered_clipped_surprisal
                         for k in range(1, G+1))
        dP = multiplier * 2 * P * (1-P)**2
        assert multiplier < 0 and dP < 0
        group_checks.append({"P": P, "history_count": history_count, "G": G,
                             "mean_score_multiplier": multiplier, "semantic_dP_dt": dP,
                             "exact_conditional_entropy_gradient": 0.0})

    rng = random.Random(20260905)
    finite_differences = []
    lower_barriers = []
    for kind, G, d, m, coefficient in itertools.product(["DrGRPO", "MaxRL"], [2, 3, 16], [3, 5], [1, 2], [.1, 1]):
        if m >= d:
            continue
        z = [rng.uniform(-3, 3) for _ in range(d)]
        _, grad, _ = quantities(z, m, coefficient, G, kind)
        epsilon = 1e-5
        error = 0
        for i in range(d):
            plus, minus = z.copy(), z.copy()
            plus[i] += epsilon
            minus[i] -= epsilon
            numeric = (quantities(plus,m,coefficient,G,kind)[0]-quantities(minus,m,coefficient,G,kind)[0])/(2*epsilon)
            error = max(error, abs(numeric-grad[i]))
        assert error < 2e-8
        finite_differences.append(error)
        M = (G-1)/G if kind == "DrGRPO" else G-1
        # Put one coordinate strictly below the mean-M/lambda threshold.
        depth = d/(d-1) * (M/coefficient+1)
        boundary = [0.0] * d
        boundary[-1] = -depth
        _, boundary_gradient, p = quantities(boundary,m,coefficient,G,kind)
        mean = sum(boundary)/d
        lower = p[-1] * (-M + coefficient*(mean-boundary[-1]))
        assert lower > 0 and boundary_gradient[-1] >= lower - 1e-13
        lower_barriers.append({"kind":kind,"G":G,"d":d,"lambda":coefficient,
                               "min_logit_velocity":boundary_gradient[-1],"proven_lower_bound":lower})

    # Exact finite-group enumeration permits arbitrary group-dependent bounded scores.
    expected_bound_checks = []
    for rare in [.2, .01, 1e-5]:
        p = [rare, .6*(1-rare), .4*(1-rare)]
        for G in [2, 3, 4]:
            total = 0.0
            for actions in itertools.product(range(3), repeat=G):
                weight = math.prod(p[a] for a in actions)
                advantages = [.1*math.sin(1+i+sum((j+1)*(a+1) for j,a in enumerate(actions))) for i in range(G)]
                coordinate = sum(A*(float(a==0)-p[0]) for a,A in zip(actions,advantages))/G
                total += weight*coordinate
            assert abs(total) <= .2*p[0] + 1e-15
            expected_bound_checks.append({"rare_probability":rare,"G":G,"expected_coordinate":total,"bound":.2*rare})

    aliases = []
    for N in [1, 100, 1000000]:
        q = [N/(N+1), 1/(N+1)]
        aliases.append({"raw_completions_per_key":[N,1],"entropy_optimal_key_mass":q,
                        "distinct_at_8":sum(1-(1-x)**8 for x in q),
                        "raw_entropy_gap_of_single_key_policy":math.log(N+1)-math.log(N)})
    result = {"status":"PASS", "scope":"Finite-dimensional analytic identities and exact source-level predictor checks. No SGD, Adam, neural-policy, or universal-collapse claim.",
              "semantic_source_sha256":hashlib.sha256(source.encode()).hexdigest(),
              "source_singleton_checks":singleton,"finite_group_leakage_checks":group_checks,
              "exact_entropy_gradient_max_finite_difference_error":max(finite_differences),
              "exact_entropy_gradient_cases":len(finite_differences),"min_logit_barrier_checks":lower_barriers,
              "bounded_advantage_expectation_checks":expected_bound_checks,"raw_to_key_entropy_examples":aliases}
    path = pathlib.Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"status":"PASS","output":str(path),"fd_max_error":max(finite_differences),
                      "finite_group_leakage_cases":len(group_checks),"gradient_cases":len(finite_differences)}))


if __name__ == '__main__':
    main()
