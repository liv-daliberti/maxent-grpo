# Proof-chain snapshot 2026-09-16
Supersedes `proof_chain_20260915b`. Pinned by
`ops/check_paper_current_contract.py` (`PROOF_REFERENCE`,
`PROOF_REFERENCE_SHA256`).

## Why the chain was re-frozen
The supplement's notation was disambiguated: four symbols each carried
more than one meaning across the appendix, and the collisions sat inside
the formal blocks themselves, so no notation fix could leave them
untouched.

| symbol | had meant | now means |
|---|---|---|
| `B` | `distinct@K` in Lemma A.1, extra modes in Apps. E/H/J, an advantage bound in Lemma N.12 | extra modes only; Lemma A.1 uses `\bar D_K` |
| `M` | replay-bank capacity (16) and the fresh-group success count | bank capacity only; success count is `R` |
| `m` | fresh-group success count and the number of correct modes | number of correct modes only; realized success count is `r` |
| `A_i` | fresh task advantage and the sampled category | advantage only; sampled category is `Z_i` |

## What changed in the chain
All 20 formal blocks are present, in the same order, none added or
removed. Eight differ, and every difference is a symbol substitution:
no statement, hypothesis, bound, constant or proof step changed.

Block 1 of 20 — 6 substitution(s):
    B_K:=\texttt{distinct@}K=\sum_c[1-(1-\mu_c)^K].  ->  \bar D_K:=\texttt{distinct@}K=\sum_c[1-(1-\mu_c)^K].
    $B_K$:  ->  $\bar D_K$:
    $B_K(P,q)\le B_K(P,q')$.  ->  $\bar D_K(P,q)\le\bar D_K(P,q')$.
    1-(1-P)^K\le B_K(P,q)\le  ->  1-(1-P)^K\le\bar D_K(P,q)\le
    $B_1=P$  ->  $\bar D_1=P$
    $B_K$.  ->  $\bar D_K$.

Block 2 of 20 — 2 substitution(s):
    $\partial B_K/\partial  ->  $\partial\bar D_K/\partial
    $B_K$.  ->  $\bar D_K$.

Block 3 of 20 — 1 substitution(s):
    \frac{\E\!\big[w(M)M(G-M)\big]}{G^2P(1-P)}>0.  ->  \frac{\E\!\big[w(R)R(G-R)\big]}{G^2P(1-P)}>0.

Block 4 of 20 — 11 substitution(s):
    $M$.  ->  $R$.
    p_{A_i}\mid  ->  p_{Z_i}\mid
    p_{A_i}\mid  ->  p_{Z_i}\mid
    $M$  ->  $R$
    $(G-M)/G$  ->  $(G-R)/G$
    $G-M$  ->  $G-R$
    $-M/G$.  ->  $-R/G$.
    w(M)\frac{M(G-M)}{G}  ->  w(R)\frac{R(G-R)}{G}
    =w(M)\frac{M(G-M)}{GP(1-P)}\nabla_zP.  ->  =w(R)\frac{R(G-R)}{GP(1-P)}\nabla_zP.
    $M\sim\operatorname{Binomial}(G,P)$  ->  $R\sim\operatorname{Binomial}(G,P)$
    $\E[M(G-M)]=G(G-1)P(1-P)$,  ->  $\E[R(G-R)]=G(G-1)P(1-P)$,

Block 5 of 20 — 4 substitution(s):
    $A_i^{\mathrm{MaxRL}}=GR_i/M-1$  ->  $A_i^{\mathrm{MaxRL}}=GR_i/R-1$
    $M>0$  ->  $R>0$
    $M=0$,  ->  $R=0$,
    p_{A_i}$  ->  p_{Z_i}$

Block 6 of 20 — 6 substitution(s):
    $M=m>0$.  ->  $R=r>0$.
    $(G-m)/m$  ->  $(G-r)/r$
    (G-m)\!\left(\nabla_z\log  ->  (G-r)\!\left(\nabla_z\log
    =\frac{G-m}{P(1-P)}\nabla_zP.  ->  =\frac{G-r}{P(1-P)}\nabla_zP.
    $M\sim\operatorname{Binomial}(G,P)$,  ->  $R\sim\operatorname{Binomial}(G,P)$,
    \E[(G-M)\1\{M>0\}] =G(1-P)-G\Pr(M=0)  ->  \E[(G-R)\1\{R>0\}] =G(1-P)-G\Pr(R=0)

Block 19 of 20 — 2 substitution(s):
    $A_1,\ldots,A_G$  ->  $Z_1,\ldots,Z_G$
    e_{A_i}-p).  ->  e_{Z_i}-p).

Block 20 of 20 — 1 substitution(s):
    $\E|\1\{A_i=b\}-p_b|=2p_b(1-p_b)$,  ->  $\E|\1\{Z_i=b\}-p_b|=2p_b(1-p_b)$,

## Verification
Reproduce with the block extractor in
`ops/check_paper_current_contract.py` (`formal_blocks`): comparing this
snapshot against `proof_chain_20260915b` yields exactly the
substitutions listed above.
