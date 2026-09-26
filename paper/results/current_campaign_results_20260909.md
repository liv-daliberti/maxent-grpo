# Current campaign results — 2026-09-09

All available audited, exactly step-3072 paired endpoints from E118, E119 and E120-R1. Completed blocks are usable paper results; unfinished blocks retain their observed n/5 and do not stand in for five seeds.

Snapshot collection: 2026-09-09T17:34:45.613650+00:00 to 2026-09-09T17:43:25.010961+00:00.

## Findings

- E118 has 11 complete five-seed model/domain comparisons. ReplayMaxRL has positive point estimates in 11/11 for pass@8, 10/11 for distinct@8, and 10/11 for B8. These are directional counts, not a pooled effect or a significance test.
- E118 Qwen-0.5B Graph (5/5 pairs): ΔP8 +0.408 [+0.200, +0.617]; ΔB8 +1.221 [+0.926, +1.517].
- E118 Qwen-0.5B Countdown (5/5 pairs): ΔP8 +0.084 [-0.074, +0.242]; ΔB8 +0.990 [+0.859, +1.121].
- E118 Qwen-3B Python (5/5 pairs): ΔP8 +0.277 [-0.037, +0.590]; ΔB8 +0.115 [+0.080, +0.149].
- E118 Qwen-3B Graph (4/5 pairs): ΔP8 +0.042; ΔB8 +0.189.
- E118 Qwen-3B PantryPlan (1/5 pairs): ΔP8 +0.039; ΔB8 +0.574.
- E118 Falcon-1B Graph (5/5 pairs): ΔP8 +0.014 [-0.016, +0.044]; ΔB8 -0.107 [-0.220, +0.005].
- E119 Level-2 Graph, ReplayDr.GRPO minus Dr.GRPO (5/5 pairs): ΔP8 +0.561 [+0.445, +0.677]; ΔB8 +0.458 [+0.394, +0.522].
- E119 Level-2 Graph, ReplayMaxRL minus MaxRL (5/5 pairs): ΔP8 +0.305 [+0.250, +0.360]; ΔB8 +0.363 [+0.280, +0.446].
- E119 Level-2 Countdown, ReplayDr.GRPO minus Dr.GRPO (5/5 pairs): ΔP8 +0.025 [-0.053, +0.102]; ΔB8 +0.020 [-0.014, +0.055].
- E119 Level-2 Countdown, ReplayMaxRL minus MaxRL (5/5 pairs): ΔP8 +0.040 [-0.034, +0.115]; ΔB8 +0.017 [+0.001, +0.032].
- E119 Level-2 Python, ReplayDr.GRPO minus Dr.GRPO (5/5 pairs): ΔP8 +0.075 [-0.053, +0.203]; ΔB8 +0.325 [-0.228, +0.878].
- E119 Level-2 Python, ReplayMaxRL minus MaxRL (5/5 pairs): ΔP8 +0.037 [-0.067, +0.142]; ΔB8 +0.175 [-0.295, +0.645].
- E119 Level-2 MathIR, ReplayDr.GRPO minus Dr.GRPO (5/5 pairs): ΔP8 +0.493 [-0.002, +0.987]; ΔB8 +0.000 [+0.000, +0.000].
- E119 Level-2 MathIR, ReplayMaxRL minus MaxRL (5/5 pairs): ΔP8 +0.173 [-0.277, +0.623]; ΔB8 +0.002 [-0.003, +0.006].
- E120 uniform minus frequency, Qwen-0.5B Graph (5/5 pairs; mechanism eligibility 5/5 available pairs): ΔP8 +0.066 [+0.061, +0.072]; ΔB8 +0.596 [+0.513, +0.687].
- E120 uniform minus frequency, Qwen-0.5B Countdown (5/5 pairs; mechanism eligibility 5/5 available pairs): ΔP8 +0.022 [+0.004, +0.040]; ΔB8 +0.639 [+0.544, +0.733].
- E120 uniform minus frequency, Qwen-0.5B PantryPlan (5/5 pairs; mechanism eligibility 5/5 available pairs): ΔP8 +0.043 [+0.006, +0.091]; ΔB8 +0.298 [+0.174, +0.400].
- E120 uniform minus frequency, Falcon-1B Graph (5/5 pairs; mechanism eligibility 5/5 available pairs): ΔP8 +0.045 [-0.000, +0.091]; ΔB8 +0.233 [+0.182, +0.320].
- E120 uniform minus frequency, Falcon-1B PantryPlan (3/5 pairs; mechanism eligibility 3/3 available pairs): ΔP8 -0.058; ΔB8 -0.540.
- E120 uniform minus frequency, Qwen-3B Graph (2/5 pairs; mechanism eligibility 2/2 available pairs): ΔP8 +0.004; ΔB8 +0.138.

## Coverage

| Campaign | Training receipts | Admitted endpoints | Complete factorial blocks | Matched seed contrasts |
|---|---:|---:|---:|---:|
| E118 | 126/150 | 126/150 | 11/15 | 61/75 |
| E119 | 83/100 | 83/100 | 4/5 | 40/50 |
| E120 | 35/45 | 35/45 | 6/9 | 35/45 |

E118 blocks require both MaxRL arms; E119 blocks require all four arms. E119 matched contrasts sum the two pairwise comparisons, each using its own available seed intersection. E120 receipts/endpoints count the 45 frequency treatment runs; uniform comparator endpoints are reused from the corresponding baseline campaign. Its complete blocks require both arms.

## Reading the tables

P8 is pass@8, D8 is distinct@8, and B8 = D8 − P8 counts correct modes beyond the first. Pass@8 is a probability (multiply its differences by 100 for percentage points); D8 and B8 are counts of modes per prompt. Absolute means and differences use the same paired seeds. Effects are replay minus its base optimizer for E118/E119, and uniform minus frequency replay for E120.

Brackets are stored descriptive paired 95% intervals: unadjusted Student-t for complete E118/E119 blocks and exhaustive 5^5 paired percentile bootstrap for complete E120 blocks. Partial blocks have no intervals. Intervals are not adjusted for the many comparisons; broad intervals and intervals spanning zero preclude strong directional claims. No missing run is imputed and no cross-domain/model estimate is pooled.

## E118: MaxRL factorial

| Model | Domain | Difference | n/5 | ΔP8 [95%] | ΔD8 [95%] | ΔB8 [95%] |
|---|---|---|---:|---:|---:|---:|
| Qwen-0.5B | Graph | ReplayMaxRL − MaxRL | 5/5 | +0.408 [+0.200, +0.617] | +1.630 [+1.155, +2.104] | +1.221 [+0.926, +1.517] |
| Qwen-0.5B | Countdown | ReplayMaxRL − MaxRL | 5/5 | +0.084 [-0.074, +0.242] | +1.074 [+0.822, +1.325] | +0.990 [+0.859, +1.121] |
| Qwen-0.5B | Python | ReplayMaxRL − MaxRL | 5/5 | +0.509 [-0.033, +1.052] | +0.955 [-0.082, +1.993] | +0.446 [-0.156, +1.048] |
| Qwen-0.5B | MathIR | ReplayMaxRL − MaxRL | 5/5 | +0.343 [+0.262, +0.424] | +0.356 [+0.274, +0.438] | +0.013 [+0.008, +0.017] |
| Qwen-0.5B | PantryPlan | ReplayMaxRL − MaxRL | 5/5 | +0.189 [+0.108, +0.271] | +0.941 [+0.636, +1.245] | +0.751 [+0.520, +0.982] |
| Falcon-1B | Graph | ReplayMaxRL − MaxRL | 5/5 | +0.014 [-0.016, +0.044] | -0.093 [-0.229, +0.043] | -0.107 [-0.220, +0.005] |
| Falcon-1B | Countdown | ReplayMaxRL − MaxRL | 5/5 | +0.025 [+0.010, +0.041] | +0.133 [+0.101, +0.166] | +0.108 [+0.082, +0.134] |
| Falcon-1B | Python | ReplayMaxRL − MaxRL | 5/5 | +0.127 [-0.372, +0.625] | +0.280 [-0.209, +0.769] | +0.153 [-0.272, +0.578] |
| Falcon-1B | MathIR | ReplayMaxRL − MaxRL | 5/5 | +0.424 [+0.310, +0.539] | +0.443 [+0.317, +0.569] | +0.019 [+0.006, +0.031] |
| Falcon-1B | PantryPlan | ReplayMaxRL − MaxRL | 5/5 | +0.073 [+0.028, +0.119] | +0.375 [+0.065, +0.685] | +0.302 [+0.034, +0.569] |
| Qwen-3B | Graph | ReplayMaxRL − MaxRL | 4/5 | +0.042 | +0.231 | +0.189 |
| Qwen-3B | Countdown | ReplayMaxRL − MaxRL | 0/5 | — | — | — |
| Qwen-3B | Python | ReplayMaxRL − MaxRL | 5/5 | +0.277 [-0.037, +0.590] | +0.391 [+0.079, +0.704] | +0.115 [+0.080, +0.149] |
| Qwen-3B | MathIR | ReplayMaxRL − MaxRL | 1/5 | +0.234 | +0.242 | +0.008 |
| Qwen-3B | PantryPlan | ReplayMaxRL − MaxRL | 1/5 | +0.039 | +0.613 | +0.574 |

Absolute paired means, shown **base → replay** (E120: **frequency → uniform**):

| Model | Domain | Left arm | n/5 | P8 | D8 | B8 |
|---|---|---|---:|---:|---:|---:|
| Qwen-0.5B | Graph | ReplayMaxRL | 5/5 | 0.537 → 0.945 | 0.679 → 2.308 | 0.142 → 1.363 |
| Qwen-0.5B | Countdown | ReplayMaxRL | 5/5 | 0.582 → 0.666 | 0.621 → 1.695 | 0.039 → 1.029 |
| Qwen-0.5B | Python | ReplayMaxRL | 5/5 | 0.172 → 0.681 | 0.172 → 1.127 | 0.000 → 0.446 |
| Qwen-0.5B | MathIR | ReplayMaxRL | 5/5 | 0.445 → 0.789 | 0.446 → 0.802 | 0.001 → 0.013 |
| Qwen-0.5B | PantryPlan | ReplayMaxRL | 5/5 | 0.531 → 0.721 | 0.531 → 1.472 | 0.000 → 0.751 |
| Falcon-1B | Graph | ReplayMaxRL | 5/5 | 0.922 → 0.936 | 2.095 → 2.002 | 1.173 → 1.066 |
| Falcon-1B | Countdown | ReplayMaxRL | 5/5 | 0.654 → 0.679 | 0.654 → 0.787 | 0.000 → 0.108 |
| Falcon-1B | Python | ReplayMaxRL | 5/5 | 0.834 → 0.961 | 0.834 → 1.114 | 0.000 → 0.153 |
| Falcon-1B | MathIR | ReplayMaxRL | 5/5 | 0.455 → 0.879 | 0.456 → 0.899 | 0.001 → 0.020 |
| Falcon-1B | PantryPlan | ReplayMaxRL | 5/5 | 0.633 → 0.706 | 0.633 → 1.008 | 0.000 → 0.302 |
| Qwen-3B | Graph | ReplayMaxRL | 4/5 | 0.810 → 0.852 | 1.480 → 1.711 | 0.670 → 0.859 |
| Qwen-3B | Countdown | ReplayMaxRL | 0/5 | — | — | — |
| Qwen-3B | Python | ReplayMaxRL | 5/5 | 0.356 → 0.633 | 0.356 → 0.748 | 0.000 → 0.115 |
| Qwen-3B | MathIR | ReplayMaxRL | 1/5 | 0.260 → 0.494 | 0.260 → 0.502 | 0.000 → 0.008 |
| Qwen-3B | PantryPlan | ReplayMaxRL | 1/5 | 0.709 → 0.748 | 1.080 → 1.693 | 0.371 → 0.945 |

## E119: Level-2 factorial

| Model | Domain | Difference | n/5 | ΔP8 [95%] | ΔD8 [95%] | ΔB8 [95%] |
|---|---|---|---:|---:|---:|---:|
| Qwen-0.5B | Graph | ReplayDr.GRPO − Dr.GRPO | 5/5 | +0.561 [+0.445, +0.677] | +1.019 [+0.882, +1.155] | +0.458 [+0.394, +0.522] |
| Qwen-0.5B | Graph | ReplayMaxRL − MaxRL | 5/5 | +0.305 [+0.250, +0.360] | +0.668 [+0.533, +0.803] | +0.363 [+0.280, +0.446] |
| Qwen-0.5B | Countdown | ReplayDr.GRPO − Dr.GRPO | 5/5 | +0.025 [-0.053, +0.102] | +0.045 [-0.047, +0.137] | +0.020 [-0.014, +0.055] |
| Qwen-0.5B | Countdown | ReplayMaxRL − MaxRL | 5/5 | +0.040 [-0.034, +0.115] | +0.057 [-0.022, +0.136] | +0.017 [+0.001, +0.032] |
| Qwen-0.5B | Python | ReplayDr.GRPO − Dr.GRPO | 5/5 | +0.075 [-0.053, +0.203] | +0.400 [-0.077, +0.877] | +0.325 [-0.228, +0.878] |
| Qwen-0.5B | Python | ReplayMaxRL − MaxRL | 5/5 | +0.037 [-0.067, +0.142] | +0.212 [-0.242, +0.667] | +0.175 [-0.295, +0.645] |
| Qwen-0.5B | MathIR | ReplayDr.GRPO − Dr.GRPO | 5/5 | +0.493 [-0.002, +0.987] | +0.493 [-0.002, +0.987] | +0.000 [+0.000, +0.000] |
| Qwen-0.5B | MathIR | ReplayMaxRL − MaxRL | 5/5 | +0.173 [-0.277, +0.623] | +0.175 [-0.275, +0.624] | +0.002 [-0.003, +0.006] |
| Qwen-0.5B | PantryPlan | ReplayDr.GRPO − Dr.GRPO | 0/5 | — | — | — |
| Qwen-0.5B | PantryPlan | ReplayMaxRL − MaxRL | 0/5 | — | — | — |

Absolute paired means, shown **base → replay** (E120: **frequency → uniform**):

| Model | Domain | Left arm | n/5 | P8 | D8 | B8 |
|---|---|---|---:|---:|---:|---:|
| Qwen-0.5B | Graph | ReplayDr.GRPO | 5/5 | 0.202 → 0.762 | 0.224 → 1.243 | 0.023 → 0.480 |
| Qwen-0.5B | Graph | ReplayMaxRL | 5/5 | 0.434 → 0.739 | 0.546 → 1.214 | 0.112 → 0.475 |
| Qwen-0.5B | Countdown | ReplayDr.GRPO | 5/5 | 0.131 → 0.156 | 0.131 → 0.176 | 0.000 → 0.020 |
| Qwen-0.5B | Countdown | ReplayMaxRL | 5/5 | 0.118 → 0.158 | 0.118 → 0.175 | 0.000 → 0.017 |
| Qwen-0.5B | Python | ReplayDr.GRPO | 5/5 | 0.812 → 0.887 | 0.812 → 1.212 | 0.000 → 0.325 |
| Qwen-0.5B | Python | ReplayMaxRL | 5/5 | 0.812 → 0.850 | 0.812 → 1.025 | 0.000 → 0.175 |
| Qwen-0.5B | MathIR | ReplayDr.GRPO | 5/5 | 0.502 → 0.994 | 0.502 → 0.994 | 0.000 → 0.000 |
| Qwen-0.5B | MathIR | ReplayMaxRL | 5/5 | 0.785 → 0.958 | 0.785 → 0.959 | 0.000 → 0.002 |
| Qwen-0.5B | PantryPlan | ReplayDr.GRPO | 0/5 | — | — | — |
| Qwen-0.5B | PantryPlan | ReplayMaxRL | 0/5 | — | — | — |

Common four-arm seed coverage (for strict factorial comparisons): Graph: 5/5 (43, 44, 45, 46, 47); Countdown: 5/5 (43, 44, 45, 46, 47); Python: 5/5 (43, 44, 45, 46, 47); MathIR: 5/5 (43, 44, 45, 46, 47); PantryPlan: 0/5 (none). The JSON also preserves both contrasts restricted to those common seeds.

## E120-R1: fresh-frequency replay ablation

| Model | Domain | Difference | n/5 | ΔP8 [95%] | ΔD8 [95%] | ΔB8 [95%] |
|---|---|---|---:|---:|---:|---:|
| Qwen-0.5B | Graph | Uniform − Frequency | 5/5 | +0.066 [+0.061, +0.072] | +0.662 [+0.575, +0.760] | +0.596 [+0.513, +0.687] |
| Qwen-0.5B | Countdown | Uniform − Frequency | 5/5 | +0.022 [+0.004, +0.040] | +0.661 [+0.550, +0.771] | +0.639 [+0.544, +0.733] |
| Qwen-0.5B | Python | Uniform − Frequency | 5/5 | -0.077 [-0.706, +0.552] | -0.040 [-0.632, +0.552] | +0.037 [-0.012, +0.123] |
| Qwen-0.5B | MathIR | Uniform − Frequency | 5/5 | -0.005 [-0.084, +0.075] | +0.014 [-0.076, +0.105] | +0.019 [-0.001, +0.038] |
| Qwen-0.5B | PantryPlan | Uniform − Frequency | 5/5 | +0.043 [+0.006, +0.091] | +0.341 [+0.198, +0.477] | +0.298 [+0.174, +0.400] |
| Falcon-1B | Graph | Uniform − Frequency | 5/5 | +0.045 [-0.000, +0.091] | +0.278 [+0.184, +0.404] | +0.233 [+0.182, +0.320] |
| Falcon-1B | PantryPlan | Uniform − Frequency | 3/5 | -0.058 | -0.598 | -0.540 |
| Qwen-3B | Graph | Uniform − Frequency | 2/5 | +0.004 | +0.142 | +0.138 |
| Qwen-3B | PantryPlan | Uniform − Frequency | 0/5 | — | — | — |

Absolute paired means, shown **base → replay** (E120: **frequency → uniform**):

| Model | Domain | Left arm | n/5 | P8 | D8 | B8 |
|---|---|---|---:|---:|---:|---:|
| Qwen-0.5B | Graph | Uniform | 5/5 | 0.903 → 0.969 | 1.775 → 2.438 | 0.872 → 1.468 |
| Qwen-0.5B | Countdown | Uniform | 5/5 | 0.650 → 0.672 | 0.976 → 1.637 | 0.326 → 0.965 |
| Qwen-0.5B | Python | Uniform | 5/5 | 0.605 → 0.528 | 0.611 → 0.571 | 0.006 → 0.043 |
| Qwen-0.5B | MathIR | Uniform | 5/5 | 0.801 → 0.796 | 0.814 → 0.829 | 0.013 → 0.032 |
| Qwen-0.5B | PantryPlan | Uniform | 5/5 | 0.686 → 0.729 | 1.199 → 1.539 | 0.513 → 0.811 |
| Falcon-1B | Graph | Uniform | 5/5 | 0.763 → 0.809 | 1.439 → 1.717 | 0.675 → 0.908 |
| Falcon-1B | PantryPlan | Uniform | 3/5 | 0.794 → 0.736 | 2.159 → 1.561 | 1.365 → 0.826 |
| Qwen-3B | Graph | Uniform | 2/5 | 0.826 → 0.830 | 1.513 → 1.654 | 0.687 → 0.824 |
| Qwen-3B | PantryPlan | Uniform | 0/5 | — | — | — |

Mechanism eligibility is separate from endpoint availability. A seed is eligible only when the stored telemetry audit passes and both arms have completion receipts. Full-block mechanism interpretation requires all five eligible paired seeds.

| Model | Domain | Endpoint pairs | Eligible pairs | Full mechanism block |
|---|---|---:|---:|---|
| Qwen-0.5B | Graph | 5/5 | 5/5 | Yes |
| Qwen-0.5B | Countdown | 5/5 | 5/5 | Yes |
| Qwen-0.5B | Python | 5/5 | 5/5 | Yes |
| Qwen-0.5B | MathIR | 5/5 | 5/5 | Yes |
| Qwen-0.5B | PantryPlan | 5/5 | 5/5 | Yes |
| Falcon-1B | Graph | 5/5 | 5/5 | Yes |
| Falcon-1B | PantryPlan | 3/5 | 3/5 | No |
| Qwen-3B | Graph | 2/5 | 2/5 | No |
| Qwen-3B | PantryPlan | 0/5 | 0/5 | No |

A partial effect remains descriptive even when every currently available pair passes telemetry. Opposite-signed domain effects must remain visible; this ablation does not establish a universal advantage for uniform replay. The historical frozen E120 primary analysis is unchanged.

## Reproduction

Source: `paper/results/latest_results_20260909.json` (SHA-256 `048c9adde4e56ead61e6236e2c81f1e4052ee25ec244d3b59c90fc612aad42ec`).

```bash
python ops/exp_scaling/build_paper_current_campaign_results.py --snapshot paper/results/latest_results_20260909.json --figures
```

The CSV and JSON retain full numerical precision, exact paired seed sets, and every stored metric interval. Three campaign TeX table bodies show all planned model/domain rows (including 0/5), effect estimates, and B8 intervals; the coverage table body reports endpoint and pairing progress.
