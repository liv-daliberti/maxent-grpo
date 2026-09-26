# E49D guided-schema compatibility amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

The first pending-only preprocessing retry failed every direct menu request
with the endpoint response:

```text
HTTP 400 Bad Request:
Grammar error: Unimplemented keys: ["uniqueItems"]
```

This was a transport/schema-compiler failure: no proposal or audit was
produced by those requests, no dataset was materialized, and no E49D training
was launched.

The unsupported `uniqueItems` annotation is removed from the guided JSON
schema. The deterministic local `parse_strategy_menu` validator already
requires every action ID within a strategy to be unique and rejects the
proposal otherwise. It also rejects duplicate complete action sequences; the
materializer's already frozen deterministic duplicate-strategy collapse
continues to run before that validation.

The HTTP client now records a bounded copy of any HTTP error body so future
transport failures are diagnosable. This change does not alter the proposal
distribution, mathematical audits, maximal-clique certification, singleton
fallback, policy data, runtime execution checks, reward, E46 controller,
cohorts, schedule, or gates. All earlier v4 certifications remain valid.
