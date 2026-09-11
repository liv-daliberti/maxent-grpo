# E49D toy support-recall result

**Decision: FAILED — NO TRAINING**

E49D toy preprocessing job `30073878` ran on `node302` from
2026-07-24 11:37:07 through 12:07:30 EDT and terminated `FAILED`
(`NonZeroExitCode`, exit code `1:0`). It completed all 84 scheduled
support-recall requests, but only 67 produced locally valid durable recall
records; 17 completed responses failed the local menu/audit contract.

The latest durable-record view contains 100 rows and superficially reports:

- 80 singleton menus;
- 20 two-strategy menus;
- 9 multi-strategy train rows and 11 multi-strategy evaluation rows; and
- four support-recall promotions over the prerecall 16/100 result.

These counts must not be interpreted as a passed support gate. Rows whose
recall response failed local validation retain their earlier passing record in
the append-only latest-record view. More importantly, manual inspection
falsified the soundness of promoted routes accepted by the two E49D judges:

1. `train:0027:train/number_theory/816.json`: the purported CRT route only
   determines a residue modulo 30. Its final action explicitly verifies against
   direct-conversion actions `A1`–`A3`, which are outside that strategy's
   declared action combination. The route therefore imports the answer from
   the comparison strategy and does not independently solve the problem.
2. `eval:0001:36611d0560b627daf41a`: the stars-and-bars route treats
   indistinguishable boxes as distinguishable and then divides by `2!`, while
   the generating-function route uses the distinguishable-box generating
   function. One judge nevertheless certified the false arithmetic
   `C(6,1) = 6 / 5 = 3`.

Additional retained menus exhibited the same classes of defect: hidden
cross-strategy dependencies, invalid arithmetic, and superficially different
wording around the same decisive operation. Because the false-new requirement
is zero on the manually audited set, a single such counterexample is terminal;
E49D is archived as a preprocessing diagnostic and cannot authorize training.

## Frozen evidence

The terminal evidence is copied under
`var/artifacts/e49d_support_recall_failed_20260724`:

| File | SHA-256 |
|---|---|
| `menu_records.jsonl` | `e5cf259aa734423329f9a97e03d8a15b94645abf07b02b142701d6894da25f76` |
| `generation_summary.json` | `6dd1ec119f115ab4b4e0acf85de72eee5cb1d755dcfb687b0ccc9af663f830c5` |
| `materialize-30073878.out` | `e46850f0dc882551b6350e09673bfe2a9079ec7cfc5ead392553648fe891bbd2` |
| `materialization_job.finished-30073878.json` | `d36184ee7a0752085dc19784ca6d033348bbe6a3dd65712955e51fcb42483f12` |

## Successor

E49E consumes the E49D outputs only as untrusted candidate proposals. It
re-certifies each route from scratch with exact action-by-action execution,
rejects references to actions outside the declared combination, checks the
derived answer with the frozen MATH verifier, and applies two new pairwise
equivalence attacks before exposing any strategy bank to training.
