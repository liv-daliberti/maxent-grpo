# E49AB — all-observed persistent pairwise hard-MATH route discovery

**Status: PREREGISTERED BEFORE E49AA COMPLETES — 2026-07-26**

## Activation and isolated change

E49AB may run only if final E49AA does not yield ten bidirectionally
executable menus.  E49AA uses at most the sixteen shortest correct natural
responses per problem.  That is a coverage restriction rather than a
semantic safety requirement.  The 74 frozen candidate problems contain 1,550
of E49Y's 1,647 answer-positive responses; all 1,647 are at most 3,706
characters and therefore fit the calibrated judge's frozen 4,000-character
per-item bound.  The other 97 remain outside the already-frozen
four-positive problem gate.

E49AB changes only that coverage restriction.  It reuses the exact frozen
E49Y responses, corrected answer-validator decisions, E47W pairwise-veto
canonicalizer, Qwen72 endpoint, menu audits, and forced-route execution gate.
It does not change the eventual training data format, reward, canonical bank,
novelty coefficient, or normalized E46 Haarnoja controller.

## Frozen online replay

For each of the 74 E49Y problems with at least four validator-positive
responses:

1. retain every validator-positive response;
2. order responses by full response SHA-256 ascending;
3. divide that order into consecutive groups of at most sixteen; and
4. replay the groups sequentially through one fresh persistent instance of
   the exact calibrated E47W pairwise-veto canonicalizer.

Problems may run concurrently, but groups for one problem must remain
sequential.  Every group uses the same prompt-local token identity, so later
responses must compare against the representatives accumulated from earlier
groups.  No bank state crosses problems.  The canonicalizer SHA-256 remains
`91a1a8fdc3b29fa49b1089154f5955ec2d9e759cc85e94b35bc7d101d81bf988`;
the passing E47W analysis remains
`3ad930355d4ef3cc30f153035db3cfdca5bf8290566b00fe3383111c4c7455d6`.

Every admitted response contributes to the support count of its emitted key.
A problem is cluster-eligible only if at least two keys each have at least two
admitted natural members across the full replay.  Retain the two largest
components, breaking ties by key.  Rank eligible problems by minimum retained
component size descending, combined retained size descending, then exact
source index ascending.

## Unchanged menu and execution gate

Only the retained components may be converted into an S1/S2 finite action
menu.  The conversion prompt supplies the first three component members in
full response-SHA order, and each route cites one to three of those observed
members.  Two independent
Qwen72 audits must establish route soundness, sufficiency, exact exemplar
binding, genuine distinctness, and absence of answer leakage.

Each surviving route then receives eight forced Qwen2.5-0.5B-Instruct
attempts.  A success requires the exact answer validator and unanimous
assignment to the requested route by the frozen E49T finite-menu
canonicalizer.  Wrong-route solutions do not count.  A problem is
bidirectionally executable only when both routes succeed at least once.

E49AB passes only with ten bidirectionally executable problems, selecting the
first ten in the frozen rank order.  Failed, ambiguous, singleton, or
wrong-route evidence may not be relabeled or promoted.
