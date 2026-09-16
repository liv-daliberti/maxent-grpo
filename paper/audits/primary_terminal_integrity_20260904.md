# Primary terminal integrity audit

Audited 445 unique run directories and 484 source logs. Exactly 1 run has conflicting terminal records; 0 runs have identical duplicates only.

The scan covers all recorded rows at step 3,072 for the four fixed sampled-evaluation draws, across every debug-job directory named by the six primary source ledgers. Source SHA-256s, byte counts, exact duplicate row numbers, payload hashes, and endpoint vectors are retained in the JSON. Concurrently active logs were read once; hashes identify the bytes actually read. No source data or figures were changed.

- Affected run: `var/data/xdr_falcon3_1b_instruct_verified_first_replay_rehearsal_only_e79_falcon_aligned_countdown_replay_s59`.
  - Draw 0: lines 172, 338; 2 different endpoint vectors.
  - Draw 1: lines 173, 339; 2 different endpoint vectors.
  - Draw 2: lines 174, 340; 2 different endpoint vectors.
  - Draw 3: lines 175, 341; 2 different endpoint vectors.

The affected Falcon Countdown ReplayDr.GRPO seed 59 is already excluded by `paper/preregistration/e112r1_two_scale_49pair_terminal_integrity_amendment_20260828.md`. Its first terminal sequence averages pass@8/distinct@8 = .64453125/.83984375; the last sequence = .650390625/.8515625. The old core reader selected the latter by overwriting keys. No later source file or registered repair was found. Exclusion leaves 74 admissible core pairs, 14 complete five-seed blocks, and a four-seed Falcon Countdown prefix. MaxRL/ReplayMaxRL pair coverage remains unchanged.
