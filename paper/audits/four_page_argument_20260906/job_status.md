Live read: 2026-09-06T17:54:56.352765+00:00 to 2026-09-06T17:54:56.968600+00:00

Counts are scientific cells, with completed / running / queued / other. Running cells with queued continuation jobs are counted once.

E118

| Size | Domain | Completed | Running | Queued | Other | Matched completed seeds |
|---|---|---:|---:|---:|---:|---|
| falcon1b | Countdown | 10/10 | 0 | 0 | 0 | 55,56,57,58,59 |
| falcon1b | Graph | 10/10 | 0 | 0 | 0 | 55,56,57,58,59 |
| falcon1b | MathIR | 10/10 | 0 | 0 | 0 | 55,56,57,58,59 |
| falcon1b | Pantry | 10/10 | 0 | 0 | 0 | 55,56,57,58,59 |
| falcon1b | Python | 10/10 | 0 | 0 | 0 | 55,56,57,58,59 |
| qwen05b | Countdown | 10/10 | 0 | 0 | 0 | 43,44,45,46,47 |
| qwen05b | Graph | 10/10 | 0 | 0 | 0 | 43,44,45,46,47 |
| qwen05b | MathIR | 10/10 | 0 | 0 | 0 | 43,44,45,46,47 |
| qwen05b | Pantry | 10/10 | 0 | 0 | 0 | 43,44,45,46,47 |
| qwen05b | Python | 10/10 | 0 | 0 | 0 | 43,44,45,46,47 |
| qwen3b | Countdown | 1/10 | 0 | 9 | 0 | none |
| qwen3b | Graph | 3/10 | 3 | 4 | 0 | 73 |
| qwen3b | MathIR | 2/10 | 0 | 8 | 0 | 74 |
| qwen3b | Pantry | 0/10 | 3 | 7 | 0 | none |
| qwen3b | Python | 10/10 | 0 | 0 | 0 | 70,71,72,73,74 |

{'completed': 116, 'queued': 28, 'running': 6}

E119

| Size | Domain | Completed | Running | Queued | Other | Matched completed seeds |
|---|---|---:|---:|---:|---:|---|
| qwen05b | Countdown | 3/20 | 7 | 10 | 0 | none |
| qwen05b | Graph | 20/20 | 0 | 0 | 0 | 43,44,45,46,47 |
| qwen05b | MathIR | 15/20 | 5 | 0 | 0 | 43,44 |
| qwen05b | Pantry | 0/20 | 0 | 20 | 0 | none |
| qwen05b | Python | 14/20 | 6 | 0 | 0 | 45,47 |

{'running': 18, 'completed': 52, 'queued': 30}

E118 matched seeds require both MaxRL and ReplayMaxRL; E119 matched seeds require all four methods. A complete block has all five registered seeds. Completion receipts retain the latest earlier efficacy-audit counts but this operational read does not rescan evaluation outcomes.
