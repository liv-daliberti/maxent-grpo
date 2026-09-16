# Independent success/breadth lemma validation

Status: passed. Reproduce with `python paper/audits/claim_theory_alignment_20260911/verify_success_breadth_independent.py`.
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

- Independent exhaustive output-tuple enumeration: 912 distributions.
- Strict lower/upper equality checks: 513.
- Majorization comparisons: 15376.
- Strict majorization comparisons: 4914.
- Correctness monotonicity checks: 684.
- Cases include `m=1`, zero-probability keys, `P=0`, `P=1`, and `K=1`.
- A countably supported geometric example uses the analytic tail certificate
  `sum_(c>N) occupancy_c <= K*P*2^(-N)`, independently checked against a
  longer exact rational prefix. Finite computation is not a proof for all
  countable distributions; the Tonelli argument supplies that proof.

## Shared text verification

The complete lemma/proof and integrated theory section are byte-identical
between `paper/main.tex` and `paper/mathai2026/appendix.tex` at the source
hashes recorded in `success_breadth_independent_validation.json`. The theory comparison spans
`\section{Mode Collapse and Verified Replay}` through, but excluding,
`\section{Source Integrity and Reproducibility}` (703 lines).
