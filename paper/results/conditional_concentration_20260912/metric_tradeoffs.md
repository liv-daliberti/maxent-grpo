# Collision and correctness changes on the same eligible prompts

Each row uses the primary nominal-stream collision population, then reports original intact-K8 metrics on those prompts. These are simultaneous changes; correctness has not been held fixed. Collision, mean@8 and pass@8 changes are percentage points; distinct@8 and extra-mode changes are expected counts. All blocks, including undefined and partial blocks, are retained.

## Level 1 · Qwen2.5-0.5B

| Domain | Comparison | n | Coverage | ΔC, pp | Δmean@8, pp | Δpass@8, pp | Δdistinct@8 | Δextra modes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | 5 | 13–31% | +24.2 | +53.9 | +0.8 | -0.317 | -0.326 |
| Graph | Final minus initial: GRPO | 5 | 12–22% | +22.1 | +52.2 | +2.0 | -0.279 | -0.299 |
| Graph | Final minus initial: MaxRL | 5 | 14–31% | +14.5 | +21.1 | +0.4 | -0.085 | -0.090 |
| Graph | Final minus initial: ReplayDr.GRPO | 5 | 35–38% | -50.7 | -0.7 | +1.5 | +1.426 | +1.411 |
| Graph | Final minus initial: ReplayMaxRL | 5 | 31–40% | -39.9 | -0.4 | +1.1 | +1.184 | +1.173 |
| Graph | Replay minus no replay: Dr.GRPO | 5 | 29–31% | -75.6 | -56.2 | +0.0 | +1.721 | +1.721 |
| Graph | Replay minus no replay: MaxRL | 5 | 32–57% | -58.7 | -20.9 | +0.1 | +1.433 | +1.431 |
| Countdown | Final minus initial: Dr.GRPO | 5 | 1–2% | +100.0 | +75.0 | +0.0 | -1.000 | -1.000 |
| Countdown | Final minus initial: GRPO | 5 | 1–2% | +96.4 | +75.0 | +0.0 | -0.850 | -0.850 |
| Countdown | Final minus initial: MaxRL | 4 | 0–2% | +97.7 | +75.8 | +0.0 | -0.812 | -0.812 |
| Countdown | Final minus initial: ReplayDr.GRPO | 5 | 1–2% | +46.7 | +39.1 | +1.7 | -0.117 | -0.133 |
| Countdown | Final minus initial: ReplayMaxRL | 5 | 1–2% | +42.0 | +53.4 | +0.0 | +0.375 | +0.375 |
| Countdown | Replay minus no replay: Dr.GRPO | 5 | 33–61% | -52.9 | -5.8 | +0.1 | +1.473 | +1.472 |
| Countdown | Replay minus no replay: MaxRL | 5 | 35–66% | -54.7 | -6.9 | +0.0 | +1.514 | +1.514 |
| Python | Final minus initial: Dr.GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: MaxRL | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: ReplayDr.GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: ReplayMaxRL | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Replay minus no replay: Dr.GRPO | 5 | 17% | -9.0 | -0.9 | +0.0 | +0.193 | +0.193 |
| Python | Replay minus no replay: MaxRL | 5 | 17% | -26.6 | -0.1 | +0.0 | +0.595 | +0.595 |
| MathIR | Final minus initial: Dr.GRPO | 5 | 4–6% | +0.0 | +75.4 | +3.7 | +0.037 | +0.000 |
| MathIR | Final minus initial: GRPO | 5 | 3–8% | +4.0 | +70.6 | +5.2 | +0.032 | -0.020 |
| MathIR | Final minus initial: MaxRL | 5 | 4–7% | +2.2 | +73.9 | +6.1 | +0.050 | -0.011 |
| MathIR | Final minus initial: ReplayDr.GRPO | 5 | 6–9% | +5.9 | +68.7 | +3.7 | +0.007 | -0.029 |
| MathIR | Final minus initial: ReplayMaxRL | 5 | 7–9% | +6.3 | +71.3 | +4.9 | +0.018 | -0.031 |
| MathIR | Replay minus no replay: Dr.GRPO | 5 | 34–53% | -0.8 | +1.6 | -0.0 | +0.018 | +0.018 |
| MathIR | Replay minus no replay: MaxRL | 5 | 25–44% | -0.5 | +1.7 | +0.2 | +0.014 | +0.012 |
| PantryPlan | Final minus initial: Dr.GRPO | 5 | 42–50% | +83.3 | +55.3 | +0.6 | -2.028 | -2.033 |
| PantryPlan | Final minus initial: GRPO | 5 | 50–51% | +84.9 | +56.9 | +1.4 | -1.962 | -1.976 |
| PantryPlan | Final minus initial: MaxRL | 5 | 42–51% | +85.9 | +56.8 | +1.6 | -1.953 | -1.969 |
| PantryPlan | Final minus initial: ReplayDr.GRPO | 5 | 61–70% | +41.3 | +38.2 | +1.4 | -0.638 | -0.652 |
| PantryPlan | Final minus initial: ReplayMaxRL | 5 | 62–69% | +44.6 | +38.4 | +1.6 | -0.750 | -0.766 |
| PantryPlan | Replay minus no replay: Dr.GRPO | 5 | 42–52% | -43.3 | -15.4 | +0.0 | +1.365 | +1.365 |
| PantryPlan | Replay minus no replay: MaxRL | 5 | 41–55% | -41.7 | -15.2 | +0.0 | +1.247 | +1.247 |

## Level 1 · Falcon3-1B

| Domain | Comparison | n | Coverage | ΔC, pp | Δmean@8, pp | Δpass@8, pp | Δdistinct@8 | Δextra modes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | 5 | 32–42% | +23.3 | +30.3 | -0.1 | +0.026 | +0.027 |
| Graph | Final minus initial: GRPO | 5 | 38–44% | +9.8 | +20.1 | -0.9 | +0.288 | +0.297 |
| Graph | Final minus initial: MaxRL | 3 | 38–41% | -11.3 | +11.1 | -1.3 | +0.628 | +0.642 |
| Graph | Final minus initial: ReplayDr.GRPO | 5 | 29–35% | -0.7 | +9.9 | -3.4 | +0.384 | +0.418 |
| Graph | Final minus initial: ReplayMaxRL | 2 | 40–41% | -6.2 | +8.8 | +0.5 | +0.541 | +0.536 |
| Graph | Replay minus no replay: Dr.GRPO | 5 | 24–50% | -30.9 | -16.5 | -2.4 | +0.414 | +0.438 |
| Graph | Replay minus no replay: MaxRL | 5 | 74–78% | -2.1 | -4.7 | +0.5 | -0.112 | -0.117 |
| Countdown | Final minus initial: Dr.GRPO | 5 | 5–6% | +2.2 | +61.2 | +0.0 | +0.240 | +0.240 |
| Countdown | Final minus initial: GRPO | 5 | 5–9% | +18.0 | +66.5 | +2.0 | -0.014 | -0.034 |
| Countdown | Final minus initial: MaxRL | 2 | 9% | +18.2 | +72.3 | +2.3 | -0.114 | -0.136 |
| Countdown | Final minus initial: ReplayDr.GRPO | 4 | 8–9% | +3.5 | +24.4 | +2.3 | +0.211 | +0.188 |
| Countdown | Final minus initial: ReplayMaxRL | 1 | 9% | +13.5 | +53.7 | +2.3 | +0.068 | +0.045 |
| Countdown | Replay minus no replay: Dr.GRPO | 4 | 26–33% | -16.8 | -23.5 | -0.4 | +0.298 | +0.302 |
| Countdown | Replay minus no replay: MaxRL | 5 | 61–63% | -6.3 | -8.4 | +0.1 | +0.172 | +0.171 |
| Python | Final minus initial: Dr.GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: MaxRL | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: ReplayDr.GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: ReplayMaxRL | 0 | — | undefined | undefined | undefined | undefined | undefined |
| Python | Replay minus no replay: Dr.GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Replay minus no replay: MaxRL | 5 | 17–100% | -9.3 | -6.2 | +0.0 | +0.190 | +0.190 |
| MathIR | Final minus initial: Dr.GRPO | 5 | 5% | +10.5 | +66.8 | +8.3 | -0.008 | -0.092 |
| MathIR | Final minus initial: GRPO | 5 | 2–5% | +5.4 | +19.5 | +1.0 | -0.042 | -0.051 |
| MathIR | Final minus initial: MaxRL | 1 | 5% | +27.8 | +61.5 | +8.3 | -0.083 | -0.167 |
| MathIR | Final minus initial: ReplayDr.GRPO | 5 | 5–5% | +8.6 | +13.3 | +6.3 | -0.026 | -0.089 |
| MathIR | Final minus initial: ReplayMaxRL | 2 | 4–5% | +16.7 | +53.9 | +4.2 | +0.079 | +0.037 |
| MathIR | Replay minus no replay: Dr.GRPO | 5 | 7–11% | +0.4 | -49.1 | -1.8 | -0.040 | -0.022 |
| MathIR | Replay minus no replay: MaxRL | 5 | 31–56% | -0.1 | +3.8 | +0.0 | +0.002 | +0.002 |
| PantryPlan | Final minus initial: Dr.GRPO | 5 | 59% | +88.6 | +62.0 | +0.3 | -1.609 | -1.612 |
| PantryPlan | Final minus initial: GRPO | 5 | 59% | +86.3 | +61.6 | +0.3 | -1.544 | -1.547 |
| PantryPlan | Final minus initial: MaxRL | 1 | 59% | +88.6 | +62.0 | +0.3 | -1.609 | -1.612 |
| PantryPlan | Final minus initial: ReplayDr.GRPO | 4 | 59–65% | +43.7 | +38.9 | +0.3 | -0.303 | -0.306 |
| PantryPlan | Final minus initial: ReplayMaxRL | 2 | 59–61% | +74.6 | +56.5 | +0.3 | -1.126 | -1.129 |
| PantryPlan | Replay minus no replay: Dr.GRPO | 5 | 58–63% | -40.4 | -18.6 | +0.0 | +1.217 | +1.217 |
| PantryPlan | Replay minus no replay: MaxRL | 5 | 63% | -12.9 | -5.2 | +0.0 | +0.474 | +0.474 |

## Level 1 · Qwen2.5-3B

| Domain | Comparison | n | Coverage | ΔC, pp | Δmean@8, pp | Δpass@8, pp | Δdistinct@8 | Δextra modes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | 5 | 34–37% | +27.4 | +22.1 | -0.1 | -0.271 | -0.270 |
| Graph | Final minus initial: GRPO | 5 | 36–38% | +27.4 | +22.5 | +0.4 | -0.293 | -0.298 |
| Graph | Final minus initial: MaxRL | 5 | 35–43% | +7.2 | +8.7 | -0.2 | +0.122 | +0.124 |
| Graph | Final minus initial: ReplayDr.GRPO | 5 | 38–41% | -3.8 | +9.9 | +0.1 | +0.447 | +0.446 |
| Graph | Final minus initial: ReplayMaxRL | 5 | 36–41% | -14.5 | +5.9 | -1.0 | +0.457 | +0.467 |
| Graph | Replay minus no replay: Dr.GRPO | 5 | 42–55% | -27.3 | -13.5 | +0.0 | +0.583 | +0.582 |
| Graph | Replay minus no replay: MaxRL | 5 | 53–65% | -16.2 | -3.8 | -0.2 | +0.240 | +0.242 |
| Countdown | Final minus initial: Dr.GRPO | 5 | 9–9% | +16.0 | +67.1 | +4.5 | -0.008 | -0.053 |
| Countdown | Final minus initial: GRPO | 5 | 9% | +18.2 | +69.8 | +4.5 | -0.045 | -0.091 |
| Countdown | Final minus initial: MaxRL | 5 | 7–9% | +5.6 | +60.5 | +2.4 | +0.001 | -0.023 |
| Countdown | Final minus initial: ReplayDr.GRPO | 5 | 9–12% | +2.0 | +42.5 | +1.9 | +0.222 | +0.202 |
| Countdown | Final minus initial: ReplayMaxRL | 5 | 8–9% | -10.0 | +41.2 | +0.4 | +0.241 | +0.237 |
| Countdown | Replay minus no replay: Dr.GRPO | 5 | 55–59% | -13.7 | -18.2 | -0.1 | +0.317 | +0.318 |
| Countdown | Replay minus no replay: MaxRL | 5 | 44–59% | -10.5 | -15.2 | -0.1 | +0.236 | +0.237 |
| Python | Final minus initial: Dr.GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: MaxRL | 5 | 2–7% | +0.0 | +78.0 | +0.0 | +0.000 | +0.000 |
| Python | Final minus initial: ReplayDr.GRPO | 0 | 0% | undefined | undefined | undefined | undefined | undefined |
| Python | Final minus initial: ReplayMaxRL | 5 | 7% | -2.3 | +73.1 | +0.0 | +0.067 | +0.067 |
| Python | Replay minus no replay: Dr.GRPO | 2 | 0–17% | -19.0 | +0.0 | +0.0 | +0.449 | +0.449 |
| Python | Replay minus no replay: MaxRL | 5 | 17–63% | -15.3 | -4.7 | +0.0 | +0.451 | +0.451 |
| MathIR | Final minus initial: Dr.GRPO | 5 | 9–11% | +0.0 | +35.0 | +5.4 | +0.054 | +0.000 |
| MathIR | Final minus initial: GRPO | 5 | 7–9% | +0.0 | +36.3 | -0.5 | -0.005 | +0.000 |
| MathIR | Final minus initial: MaxRL | 5 | 5–11% | +0.0 | +29.1 | +7.9 | +0.079 | +0.000 |
| MathIR | Final minus initial: ReplayDr.GRPO | 4 | 11% | -1.2 | +46.3 | +5.4 | +0.058 | +0.004 |
| MathIR | Final minus initial: ReplayMaxRL | 5 | 9–12% | +0.0 | +47.8 | +7.2 | +0.072 | +0.000 |
| MathIR | Replay minus no replay: Dr.GRPO | 5 | 18–27% | -1.0 | +23.0 | +1.1 | +0.024 | +0.013 |
| MathIR | Replay minus no replay: MaxRL | 5 | 16–21% | +0.0 | +21.7 | +0.4 | +0.004 | +0.000 |
| PantryPlan | Final minus initial: Dr.GRPO | 5 | 38–52% | +44.7 | +53.3 | +0.9 | -0.622 | -0.631 |
| PantryPlan | Final minus initial: GRPO | 5 | 38–48% | +46.9 | +59.0 | +0.9 | -0.664 | -0.673 |
| PantryPlan | Final minus initial: MaxRL | 5 | 46–54% | +43.1 | +51.7 | +0.8 | -0.519 | -0.527 |
| PantryPlan | Final minus initial: ReplayDr.GRPO | 5 | 50–55% | +18.4 | +42.0 | -0.0 | +0.183 | +0.183 |
| PantryPlan | Final minus initial: ReplayMaxRL | 5 | 53–59% | +14.2 | +41.7 | -0.3 | +0.279 | +0.282 |
| PantryPlan | Replay minus no replay: Dr.GRPO | 5 | 55–65% | -30.0 | -8.9 | -0.7 | +0.875 | +0.881 |
| PantryPlan | Replay minus no replay: MaxRL | 5 | 63–70% | -29.2 | -5.8 | -0.1 | +0.860 | +0.860 |

## Level 2 · Qwen2.5-0.5B

| Domain | Comparison | n | Coverage | ΔC, pp | Δmean@8, pp | Δpass@8, pp | Δdistinct@8 | Δextra modes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | 5 | 2–8% | +62.5 | +58.0 | +0.0 | -0.532 | -0.532 |
| Graph | Final minus initial: MaxRL | 5 | 5–8% | +45.2 | +29.2 | +0.0 | -0.219 | -0.219 |
| Graph | Final minus initial: ReplayDr.GRPO | 5 | 7–9% | -20.4 | +2.9 | -2.0 | +0.611 | +0.630 |
| Graph | Final minus initial: ReplayMaxRL | 5 | 7–9% | -12.8 | +4.0 | +0.0 | +0.403 | +0.403 |
| Graph | Replay minus no replay: Dr.GRPO | 5 | 7–20% | -57.7 | -52.1 | -1.3 | +0.752 | +0.765 |
| Graph | Replay minus no replay: MaxRL | 5 | 22–33% | -40.5 | -17.9 | +0.3 | +0.580 | +0.578 |
| Countdown | Final minus initial: Dr.GRPO | 1 | 0–1% | +0.0 | +75.0 | +0.0 | +0.000 | +0.000 |
| Countdown | Final minus initial: MaxRL | 1 | 0–1% | +0.0 | +75.0 | +0.0 | +0.000 | +0.000 |
| Countdown | Final minus initial: ReplayDr.GRPO | 5 | 1% | +0.0 | +27.5 | +0.0 | +0.000 | +0.000 |
| Countdown | Final minus initial: ReplayMaxRL | 3 | 0–1% | +0.0 | +47.9 | +0.0 | +0.000 | +0.000 |
| Countdown | Replay minus no replay: Dr.GRPO | 4 | 0–7% | -23.0 | -29.3 | +0.0 | +0.234 | +0.234 |
| Countdown | Replay minus no replay: MaxRL | 4 | 0–8% | -7.3 | -10.9 | +0.0 | +0.146 | +0.146 |
| Python | Final minus initial: Dr.GRPO | 5 | 25% | +30.2 | +78.9 | +5.5 | -0.109 | -0.164 |
| Python | Final minus initial: MaxRL | 5 | 25% | +30.2 | +78.9 | +5.5 | -0.109 | -0.164 |
| Python | Final minus initial: ReplayDr.GRPO | 5 | 25% | +9.4 | +78.9 | +5.5 | +0.291 | +0.236 |
| Python | Final minus initial: ReplayMaxRL | 5 | 25% | +19.9 | +78.5 | +5.5 | +0.094 | +0.039 |
| Python | Replay minus no replay: Dr.GRPO | 5 | 81% | -20.9 | +0.0 | +0.0 | +0.400 | +0.400 |
| Python | Replay minus no replay: MaxRL | 5 | 81% | -10.6 | -0.6 | +0.0 | +0.215 | +0.215 |
| MathIR | Final minus initial: Dr.GRPO | 3 | 0–4% | +0.0 | +65.6 | +0.0 | +0.000 | +0.000 |
| MathIR | Final minus initial: MaxRL | 4 | 0–5% | +0.0 | +64.7 | +0.0 | +0.000 | +0.000 |
| MathIR | Final minus initial: ReplayDr.GRPO | 5 | 5% | +0.0 | +68.4 | +0.0 | +0.000 | +0.000 |
| MathIR | Final minus initial: ReplayMaxRL | 4 | 5% | +0.0 | +66.8 | +0.0 | +0.000 | +0.000 |
| MathIR | Replay minus no replay: Dr.GRPO | 5 | 9–95% | +0.0 | +3.2 | +0.1 | +0.001 | +0.000 |
| MathIR | Replay minus no replay: MaxRL | 5 | 23–98% | -0.1 | +2.7 | +0.1 | +0.003 | +0.002 |
| PantryPlan | Final minus initial: Dr.GRPO | 3 | 0–2% | -57.8 | -14.2 | -8.3 | +0.056 | +0.139 |
| PantryPlan | Final minus initial: MaxRL | 3 | 1–4% | -23.8 | -5.0 | -1.7 | +0.208 | +0.225 |
| PantryPlan | Final minus initial: ReplayDr.GRPO | 2 | 12–12% | -4.9 | +38.1 | -0.8 | +0.345 | +0.353 |
| PantryPlan | Final minus initial: ReplayMaxRL | 3 | 14–16% | +2.0 | +29.1 | -0.8 | +0.174 | +0.183 |
| PantryPlan | Replay minus no replay: Dr.GRPO | 3 | 0–12% | +49.5 | +38.7 | +0.0 | +0.094 | +0.094 |
| PantryPlan | Replay minus no replay: MaxRL | 4 | 0–15% | +43.9 | +19.5 | +0.7 | -0.227 | -0.234 |

Full-128-prompt metric values, eligible counts, seed-level changes, and uncertainty are preserved in the CSVs and the source JSON. Correctness on this collision-eligible subset is not the complete held-out benchmark average.
