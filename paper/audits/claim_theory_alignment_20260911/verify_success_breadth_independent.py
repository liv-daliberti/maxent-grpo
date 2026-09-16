#!/usr/bin/env python3
"""Exact, independent validation of the shared success/breadth lemma.

Run from any directory with Python's standard library. This writes only its
own JSON/Markdown audit reports next to this script; it never edits papers.
The finite checks enumerate output tuples directly, rather than using the
occupancy formula to generate both sides of a check. Infinite-support scope
is justified analytically in the report, not certified by finite testing.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPERS = (ROOT / 'paper/main.tex', ROOT / 'paper/mathai2026/appendix.tex')


def compositions(total, size):
    if size == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, size - 1):
            yield (first,) + rest


def breadth(p_correct, q, k):
    return sum((1 - (1 - p_correct * qi) ** k for qi in q), F(0))


def enumerated_metrics(p_correct, q, k):
    probs = (1 - p_correct,) + tuple(p_correct * qi for qi in q)
    success = F(0)
    distinct = F(0)
    mass = F(0)
    for draws in product(range(len(probs)), repeat=k):
        weight = F(1)
        for category in draws:
            weight *= probs[category]
        if weight == 0:
            continue
        count = len(set(draws) - {0})
        mass += weight
        success += weight * bool(count)
        distinct += weight * count
    assert mass == 1
    return success, distinct


def majorizes(q, other):
    left, right = sorted(q, reverse=True), sorted(other, reverse=True)
    return all(sum(left[:i]) >= sum(right[:i]) for i in range(1, len(q)))


def block(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


def metric_lemma(text):
    mark = text.index(r'\label{lem:success-breadth}')
    start = text.rfind(r'\begin{lemma}', 0, mark)
    end = text.index(r'\end{proof}', mark) + len(r'\end{proof}')
    return text[start:end]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    probability_grid = (F(0), F(1, 4), F(1, 2), F(1))
    budgets = (1, 2, 3, 4)
    count = {'enumerated_distributions': 0, 'strict_bound_checks': 0,
             'majorization_checks': 0, 'strict_majorization_checks': 0,
             'correctness_monotonicity_checks': 0}
    for size in range(1, 5):
        qs = {tuple(F(x, 4) for x in c) for c in compositions(4, size)}
        qs.add(tuple(F(1, size) for _ in range(size)))
        qs = sorted(qs)
        for q in qs:
            for p_correct in probability_grid:
                for k in budgets:
                    success, distinct = enumerated_metrics(p_correct, q, k)
                    analytic_success = 1 - (1 - p_correct) ** k
                    analytic_distinct = breadth(p_correct, q, k)
                    assert success == analytic_success
                    assert distinct == analytic_distinct
                    assert success <= distinct <= k * p_correct
                    lower = analytic_success
                    upper = size * (1 - (1 - p_correct / size) ** k)
                    assert lower <= distinct <= upper
                    if p_correct == 0:
                        assert success == distinct == 0  # q here is arbitrary bookkeeping.
                    if k == 1:
                        assert distinct == p_correct
                    if p_correct > 0 and k >= 2:
                        assert (distinct == lower) == (sum(x > 0 for x in q) == 1)
                        assert (distinct == upper) == all(x == F(1, size) for x in q)
                        count['strict_bound_checks'] += 1
                    count['enumerated_distributions'] += 1
            for p_low, p_high in zip(probability_grid, probability_grid[1:]):
                for k in budgets:
                    assert breadth(p_low, q, k) < breadth(p_high, q, k)
                    count['correctness_monotonicity_checks'] += 1
        for q in qs:
            for other in qs:
                if not majorizes(q, other):
                    continue
                for p_correct in probability_grid:
                    for k in budgets:
                        value, other_value = breadth(p_correct, q, k), breadth(p_correct, other, k)
                        assert value <= other_value
                        count['majorization_checks'] += 1
                        if p_correct > 0 and k >= 2 and sorted(q) != sorted(other):
                            assert value < other_value
                            count['strict_majorization_checks'] += 1

    # A countable geometric example: q_c=2^{-c}. Each omitted occupancy
    # contribution is <= K*P*q_c, so the infinite tail after N is <= K*P*2^{-N}.
    p_correct, k, short_n, long_n = F(3, 5), 8, 10, 30
    short_q = tuple(F(1, 2**c) for c in range(1, short_n + 1))
    long_q = tuple(F(1, 2**c) for c in range(1, long_n + 1))
    short_b, long_b = breadth(p_correct, short_q, k), breadth(p_correct, long_q, k)
    tail_bound = k * p_correct * F(1, 2**short_n)
    assert short_b < long_b <= short_b + tail_bound

    texts = [p.read_text() for p in PAPERS]
    lemmas = [metric_lemma(t) for t in texts]
    theories = [block(t, r'\section{Mode Collapse and Verified Replay}',
                     r'\section{Source Integrity and Reproducibility}') for t in texts]
    assert lemmas[0] == lemmas[1], 'Metric lemmas differ between root/workshop'
    assert theories[0] == theories[1], 'Integrated theory blocks differ between root/workshop'
    result = {
        'status': 'passed', 'arithmetic': 'exact rational Fraction; no numerical tolerance',
        'counts': count,
        'enumeration_scope': {'valid_keys': [1, 2, 3, 4], 'q_grid_denominator': 4,
                              'uniform_q_added_for_every_size': True,
                              'P': [str(p) for p in probability_grid], 'K': list(budgets)},
        'countable_example': {'q_c': '2^-c, c=1,2,...', 'P': str(p_correct), 'K': k,
                              'N': short_n, 'verified_prefix': long_n,
                              'certified_tail_upper_bound': str(tail_bound)},
        'source_sha256': {str(p.relative_to(ROOT)): sha(t) for p, t in zip(PAPERS, texts)},
        'shared_metric_lemma_sha256': sha(lemmas[0]),
        'shared_theory_sha256': sha(theories[0]),
        'shared_theory_lines': len(theories[0].splitlines()),
        'root_workshop_metric_lemma_identical': True,
        'root_workshop_theory_identical': True,
    }
    json_path = HERE / 'success_breadth_independent_validation.json'
    json_path.write_text(json.dumps(result, indent=2) + '\n')
    report = f'''# Independent success/breadth lemma validation

Status: passed. Reproduce with `python {Path(__file__).relative_to(ROOT)}`.
The script uses only Python's standard library and exact rational arithmetic.
It does not modify either manuscript.

## Mathematical review

- For iid draws from a fixed law on a finite or countable valid-key set,
  all failures have probability `(1-P)^K`. Each key's appearance indicator
  has expectation `1-(1-mu_c)^K`. Tonelli's theorem permits summing the
  nonnegative indicators for countably many keys. The series is finite since
  `1-(1-mu_c)^K <= K*mu_c`, hence `B_K <= K*P <= K`.
- A finite valid-key set is required only for the stated finite-dimensional
  majorization result and the uniform upper bound involving its size `m`.
  In an uncountable non-atomic output space, summing singleton masses need
  not recover success probability. Explicitly specifying finite/countable
  keys and iid draws removes that interpretation. Language-model finite
  token-string responses induce at most countably many keys.
- For `P>0` and `K>=2`, `f(v)=1-(1-P*v)^K` is strictly concave on `[0,1]`.
  Its derivative `K*P*(1-P*v)^(K-1)` is strictly decreasing; this remains
  true when `P=1`, despite a zero second derivative at the single endpoint
  `v=1` for `K>2`. Symmetry and strict concavity establish Schur concavity.
  A strict majorization comparison gives a strict breadth inequality unless
  the vectors differ only by a permutation.
- The point mass majorizes every distribution; every distribution majorizes
  the uniform vector. Equality in the lower bound requires one positive
  coordinate, while upper-bound equality requires uniformity. When `m=1`,
  these describe the same unique distribution, so both equalities are valid.
- At `P=0`, both metrics are zero and conditional `q` is undefined; the lemma
  explicitly assumes `P>0` before using `q`. At `K=1`, `B_1=P` independently
  of `q`, so strict majorization/equality conclusions correctly exclude it.
- For fixed `q`, every term is nondecreasing in `P`; the displayed finite
  sum derivative is correct. The derivative at `P=1` can be zero for a point
  mass with `K>1`, which does not invalidate monotonicity. The function is
  in fact strictly increasing as `P` increases over distinct values.
- These equations apply per prompt. Replacing promptwise correctness with
  its average inside the nonlinear success formula is generally invalid.

## Reproducible finite checks

- Independent exhaustive output-tuple enumeration: {count['enumerated_distributions']} distributions.
- Strict lower/upper equality checks: {count['strict_bound_checks']}.
- Majorization comparisons: {count['majorization_checks']}.
- Strict majorization comparisons: {count['strict_majorization_checks']}.
- Correctness monotonicity checks: {count['correctness_monotonicity_checks']}.
- Cases include `m=1`, zero-probability keys, `P=0`, `P=1`, and `K=1`.
- A countably supported geometric example uses the analytic tail certificate
  `sum_(c>N) occupancy_c <= K*P*2^(-N)`, independently checked against a
  longer exact rational prefix. Finite computation is not a proof for all
  countable distributions; the Tonelli argument supplies that proof.

## Shared text verification

The complete lemma/proof and integrated theory section are byte-identical
between `paper/main.tex` and `paper/mathai2026/appendix.tex` at the source
hashes recorded in `{json_path.name}`. The theory comparison spans
`\\section{{Mode Collapse and Verified Replay}}` through, but excluding,
`\\section{{Source Integrity and Reproducibility}}` ({result['shared_theory_lines']} lines).
'''
    (HERE / 'success_breadth_independent_validation.md').write_text(report)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
