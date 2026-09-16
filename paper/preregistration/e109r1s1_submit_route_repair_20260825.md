# E109-R1-S1 submit-route repair

Frozen: 2026-08-25 after continuation job 30873997 was submitted user-held,
failed the pre-release audit at zero runtime, and was canceled. It created no
training attempt and changed no E109 run directory. No evaluation endpoint was
read.

The site submit router rewrote requested partition `all` to `cs` while retaining
the frozen A6000 node pool and byte-identical scientific export. This is the
same submit-side normalization previously observed in this campaign.

For the replacement transaction, accept `cs` only as the audited initial held
state. While each new job remains user-held at zero runtime, explicitly update
partition to `all` and account to `allcs`, then repeat the complete resource and
environment audit. Release the two continuations only after both held records
show `all`. On failure, cancel every replacement. All E109-R1 scientific and
checkpoint requirements remain unchanged.

