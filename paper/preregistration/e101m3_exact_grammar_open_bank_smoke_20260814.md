# E101m3: exact-grammar open-bank actuator smoke

Date frozen: 2026-08-14, after E101m2 completed and before E101m3 submission.

## Motivation

E101m2 observed one singleton-eligible update but zero validator-positive rows
in three untouched-prompt proposal groups (48 responses). Earlier conditioned
proposal searches either copied the anchor or destroyed validity. Blind model
resampling is therefore too sparse to identify the replay/balance composition.

E63's deterministic Countdown sign rewrites are not reused: the runtime
validator accepted unary-negative trees outside the dataset's exact enumerated
binary grammar. E101m3 instead searches a fixed radius-two neighborhood in the
public three-position easy3 action grammar. It maps the model's verified anchor
to its grammar-code aliases without consulting the target, mutates at most two
of three code positions, decodes those codes, and lets the ordinary validator
filter them. It never reads `num_completions`, an answer list, desired entropy,
or evaluation feedback. An offline audit over the frozen 32 training prompts
found zero exact-support mismatches, an expandable neighbor for 113/136 valid
anchor modes, and at least one expandable anchor on 30/32 prompts.

## Frozen cell

- Qwen2.5-0.5B-Instruct; the same E101 tiny Countdown split and seed 102 as
  E101m2; 32 prompts, one pass, 32 optimizer updates.
- Task-only Dr.GRPO plus replay mass 0.10 and known-bank balance 0.10. Every
  semantic/PPO-advantage coefficient is zero.
- Singleton-only, separate objective support, legacy sign transforms off,
  exact-grammar radius-two transforms on, at most one admitted outcome.
- One untouched-prompt proposal attempt remains only as a registered fallback.
  A clean mechanism success requires admission from transform telemetry and
  zero sampled proposal groups, so fallback sampling cannot receive credit.
- Proposal/transform rows enter replay support only: zero PPO rows, zero
  neutral objective-support change, and zero gold/support-size/eval feedback.
- Evaluation at steps 0 and 32 with eight prompts, K=4, one draw is an integrity
  readout only.
- Placement: `all` partition, `mltheory` account, one generic GPU on the frozen
  healthy >=24 GB whitelist, 22 GB host memory, hard 30-minute limit.

## Gate

Success requires singleton eligibility, exact-grammar transform activation,
at least one validator-positive novel transformed outcome, admission, zero
sampled proposal groups, zero leakage, and replay actuation at or after the
first admission. This is a mechanism result, not a performance estimate. A
pass authorizes a separately named matched comparison; failure returns focus to
the explorer and does not authorize semantic advantage mixing.
