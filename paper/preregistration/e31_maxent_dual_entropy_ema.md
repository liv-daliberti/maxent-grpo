# E31: responsive entropy-EMA Haarnoja dual

**Status: FROZEN — prospective method change (2026-07-22, before new compute).**

## Motivation

The free-form controller observes one prompt group per optimizer update. The
historical Haarnoja-style rule sent that raw entropy observation directly to
log-alpha Adam, allowing prompt identity to flip the dual-gradient sign. A
very slow EMA would reduce noise but add undesirable control lag.

## Frozen controller

For raw entropy observation \(\widehat H_t\), define

\[
\bar H_t = 0.7\,\bar H_{t-1} + 0.3\,\widehat H_t.
\]

The first observation initializes the EMA. The dual gradient is

\[
g_t = \alpha_t(\bar H_t-H^\star),
\]

and the existing Adam update, learning rate, target, and alpha projection are
otherwise unchanged. Decay `0.7` has an approximately two-update half-life,
absorbs 90% of a sustained shift in about seven updates, and corresponds to
an effective window of roughly six independent observations. Decay `0`
recovers the historical instantaneous-feedback rule.

Log raw entropy, EMA entropy, decay, EMA target error, alpha loss, alpha
gradient, log-alpha, and optimizer step on every update. Persist the EMA and
decay in the controller checkpoint. The checkpoint rule is versioned; a
legacy instantaneous-feedback controller checkpoint must not resume silently
under this method.

## Provenance boundary

This is the default for newly launched Haarnoja-dual experiments. E16--E29
and their repair continuations retain their frozen historical controller and
must not be relabeled as EMA runs. Any comparison using this controller needs
a new experiment stamp and must keep the dual learning rate unchanged if it
is intended to isolate the EMA intervention.
