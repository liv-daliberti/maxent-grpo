# Pending A5000 placement expansion — September 9, 2026

The user requested additional concurrent work to finish E118/E119/E120-R1 as
soon as possible. After the five-cell A5000 allocation amendment, expand only
still-pending jobs31158503–31158507 from nodes202/203 to nodes105/202/203/204.
All four nodes provide the same qualified A5000 hardware and accept the jobs'
existing mltheory account in lowprio. Preserve job IDs, account, partition,
automatic QoS, one typed A5000, 16 CPUs, 116/128 GiB RAM, 72-hour walltime,
Nice, automatic requeue, exclusions, frozen launchers/source, all exported
runtime values, run directories, checkpoints and scientific identities.

The sole scheduler mutation is candidate NodeList. Node203 entered DRAIN for
GPU temperature at10:26; retain it as an existing candidate while Slurm prevents
allocation until its health is restored. Do not change any node-health state.
Add healthy qualified node105 and node204 so completions there can admit these
jobs earlier. E119 Pantry remains on its separate larger-GPU routes.

Revalidate canonical mappings, incomplete scientific status, checkpoint identity,
sole scheduler writer and pending state before each owned hold. Skip started
jobs and preexisting holds. Audit unchanged resources and complete submitted
exports before releasing an amended job. Preserve the completed original
transaction and its hashed protocol byte-for-byte; write no campaign ledger.
Store before/after records and restartable state in
`var/artifacts/campaign_a5000_widening_20260909/`. Actual starts require separate
scheduler and optimizer-progress observations; dry-run estimates are not starts.
