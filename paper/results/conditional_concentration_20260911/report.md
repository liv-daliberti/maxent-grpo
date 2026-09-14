# Conditional concentration from saved verified outputs

This report includes all 135 analyzed blocks: 121 have a defined primary mean and 14 are undefined. No block is selected by effect direction or significance.

## Findings across the complete reported groups

- **Training concentration:** 12/12 Level-1 Graph/PantryPlan Dr.GRPO and GRPO initial-to-final blocks increase collision; 12/12 have nominal intervals entirely above zero. Both disjoint orientations are positive in 11/12. The full tables identify the orientation exceptions.
- **Dr.GRPO replay effect:** 6/6 Graph/PantryPlan blocks decrease collision; 6/6 have nominal intervals entirely below zero, and 6/6 have two negative disjoint orientations.
- **Qwen2.5-3B MaxRL replay across every domain:** Graph -16.2 [-26.1, -6.2] pp, n=5 (split directions differ); Countdown -10.5 [-12.6, -8.4] pp, n=5 (both splits negative); Python -15.3 [-26.5, -4.1] pp, n=5 (both splits negative); MathIR +0.0 [+0.0, +0.0] pp, n=5 (both splits zero); PantryPlan -29.2 [-31.9, -26.5] pp, n=5 (both splits negative).
- **Mitigation does not imply a return to the initial collision profile:** 6/6 PantryPlan replay-arm initial-to-final primary estimates remain positive (+14.2 to +74.6 pp); 2 are partial. These before/after and replay contrasts use their respective eligible populations and must not be subtracted as if they shared one population.
- **Simultaneous correctness tradeoff:** eligible-prompt mean@8 decreases in 12/12 Graph/PantryPlan replay blocks (-56.2 to -3.8 pp). Lower collision here is not evidence that correctness was held fixed. The companion table also reports pass@8, distinct@8 and extra modes.
- **Harder-level boundary:** PantryPlan replay estimates are sparse: Dr.GRPO +49.0 pp, n=1, coverage 0–12%, 2/2 defined splits; MaxRL +100.0 pp, n=1, coverage 1%, 0/2 defined splits. These retained results do not support a uniform replay-improvement claim.

This artifact includes a documented cohort extension performed after the initial concentration results were inspected. Its amended source census determines all additions and withdrawals; the extension is not presented as an analysis frozen before those first results. Original and extension provenance remain bound in the source JSON.

The primary estimate averages per-prompt correct-key collision differences on prompts with at least two correct nominal-stream representatives in both conditions, then averages the measured seed estimates. ΔC is B minus A. For initial-to-final comparisons B is the final checkpoint; for replay comparisons B is the replay arm.

## Reading the figures and tables

- `conditional_concentration_overview.pdf`: the three original objectives before/after training and both paired replay effects, for every Level-1 domain and scale.
- `all_before_after_level1.pdf`: all five methods, including the replay arms, before/after training.
- `conditional_concentration_level2.pdf`: all registered main contrast types on the harder level.
- `block_summaries.csv`: every summary, metric, interval, defined seed count and available coverage field.
- `seed_metrics.csv`: seed-level metrics, selected and full-population values, eligible counts and pair-weighted diagnostics.
- `all_numeric_scalars.csv`: every numeric, Boolean and null scalar under every analyzed block, including sensitivities and fixed-across-seed populations. Prompt identity lists remain in the source JSON.
- `metric_tradeoffs.md`: collision and all four original K8 metric changes side by side for every block, on the same collision-eligible prompt population.

These comparisons do not hold correctness fixed. Read collision changes alongside the original mean@8, pass@8, distinct@8 and extra-mode changes in `metric_tradeoffs.md`; lower collision can occur together with lower correctness on the selected prompts. The K8 metrics retain their original groups and are not recomputed as eleven-sample metrics.

Numbers below are percentage-point changes in collision. Brackets are nominal 95% seed intervals when all five seed estimates are defined. `n` is the number of defined seed estimates; coverage is the range of eligible-prompt fractions across available seeds. Separate disjoint orientations have their own eligibility and counts.

## Level 1 · Qwen2.5-0.5B

| Domain | Comparison | Primary ΔC [95%] | n / admitted | Coverage | A low / B high: ΔC; n; coverage | A high / B low: ΔC; n; coverage |
|---|---|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | +24.2 [+11.0, +37.3] | 5 / 5 | 13–31% | +25.7; 5; 5–19% | +28.4; 5; 11–31% |
| Graph | Final minus initial: GRPO | +22.1 [+9.7, +34.4] | 5 / 5 | 12–22% | +26.8; 5; 5–12% | +28.7; 5; 9–16% |
| Graph | Final minus initial: MaxRL | +14.5 [-3.3, +32.4] | 5 / 5 | 14–31% | +11.5; 5; 8–20% | +17.4; 5; 9–23% |
| Graph | Final minus initial: ReplayDr.GRPO | -50.7 [-55.8, -45.6] | 5 / 5 | 35–38% | -51.9; 5; 12–16% | -46.5; 5; 16–27% |
| Graph | Final minus initial: ReplayMaxRL | -39.9 [-45.8, -33.9] | 5 / 5 | 31–40% | -43.6; 5; 16–21% | -34.8; 5; 16–31% |
| Graph | Replay minus no replay: Dr.GRPO | -75.6 [-83.1, -68.0] | 5 / 5 | 29–31% | -85.3; 5; 23–27% | -80.7; 5; 15–23% |
| Graph | Replay minus no replay: MaxRL | -58.7 [-75.9, -41.4] | 5 / 5 | 32–57% | -62.5; 5; 25–34% | -60.5; 5; 24–42% |
| Countdown | Final minus initial: Dr.GRPO | +100.0 [+100.0, +100.0] | 5 / 5 | 1–2% | undefined; 0; 0% | +100.0; 5; 1–2% |
| Countdown | Final minus initial: GRPO | +96.4 [+86.3, +106.5] | 5 / 5 | 1–2% | undefined; 0; 0% | +92.0; 5; 1–2% |
| Countdown | Final minus initial: MaxRL | +97.7 | 4 / 5 | 0–2% | undefined; 0; 0% | +100.0; 4; 0–1% |
| Countdown | Final minus initial: ReplayDr.GRPO | +46.7 [+11.7, +81.6] | 5 / 5 | 1–2% | undefined; 0; 0% | +71.2; 4; 0–2% |
| Countdown | Final minus initial: ReplayMaxRL | +42.0 [+35.9, +48.0] | 5 / 5 | 1–2% | undefined; 0; 0% | +49.3; 5; 1–2% |
| Countdown | Replay minus no replay: Dr.GRPO | -52.9 [-60.5, -45.2] | 5 / 5 | 33–61% | -52.9; 5; 30–59% | -53.5; 5; 30–59% |
| Countdown | Replay minus no replay: MaxRL | -54.7 [-63.5, -45.9] | 5 / 5 | 35–66% | -57.8; 5; 33–64% | -53.0; 5; 34–65% |
| Python | Final minus initial: Dr.GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: MaxRL | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: ReplayDr.GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: ReplayMaxRL | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Replay minus no replay: Dr.GRPO | -9.0 [-34.1, +16.0] | 5 / 5 | 17% | -11.5; 5; 16–17% | -6.2; 5; 17% |
| Python | Replay minus no replay: MaxRL | -26.6 [-57.7, +4.4] | 5 / 5 | 17% | -29.8; 5; 17% | -27.2; 5; 17% |
| MathIR | Final minus initial: Dr.GRPO | +0.0 [+0.0, +0.0] | 5 / 5 | 4–6% | +0.0; 5; 1–2% | +0.0; 1; 0–2% |
| MathIR | Final minus initial: GRPO | +4.0 [-7.1, +15.1] | 5 / 5 | 3–8% | +0.0; 5; 2% | +25.0; 2; 0–3% |
| MathIR | Final minus initial: MaxRL | +2.2 [-3.9, +8.4] | 5 / 5 | 4–7% | +0.0; 5; 1–2% | +6.7; 5; 1–2% |
| MathIR | Final minus initial: ReplayDr.GRPO | +5.9 [-4.6, +16.3] | 5 / 5 | 6–9% | +0.0; 5; 2–2% | +36.7; 2; 0–4% |
| MathIR | Final minus initial: ReplayMaxRL | +6.3 [-0.9, +13.4] | 5 / 5 | 7–9% | +0.0; 5; 2% | +25.0; 5; 1–3% |
| MathIR | Replay minus no replay: Dr.GRPO | -0.8 [-1.7, +0.1] | 5 / 5 | 34–53% | -0.6; 5; 30–52% | -0.8; 5; 30–53% |
| MathIR | Replay minus no replay: MaxRL | -0.5 [-1.3, +0.3] | 5 / 5 | 25–44% | -0.7; 5; 25–44% | -0.2; 5; 23–42% |
| PantryPlan | Final minus initial: Dr.GRPO | +83.3 [+81.9, +84.7] | 5 / 5 | 42–50% | +100.0; 5; 30–42% | +100.0; 5; 28–38% |
| PantryPlan | Final minus initial: GRPO | +84.9 [+84.5, +85.3] | 5 / 5 | 50–51% | +100.0; 5; 34–42% | +100.0; 5; 37–38% |
| PantryPlan | Final minus initial: MaxRL | +85.9 [+83.6, +88.1] | 5 / 5 | 42–51% | +100.0; 5; 33–42% | +100.0; 5; 27–39% |
| PantryPlan | Final minus initial: ReplayDr.GRPO | +41.3 [+34.5, +48.1] | 5 / 5 | 61–70% | +54.3; 5; 41–45% | +53.8; 5; 43–46% |
| PantryPlan | Final minus initial: ReplayMaxRL | +44.6 [+32.4, +56.7] | 5 / 5 | 62–69% | +59.3; 5; 43–46% | +55.0; 5; 44–46% |
| PantryPlan | Replay minus no replay: Dr.GRPO | -43.3 [-51.8, -34.7] | 5 / 5 | 42–52% | -44.2; 5; 40–49% | -49.4; 5; 38–51% |
| PantryPlan | Replay minus no replay: MaxRL | -41.7 [-54.7, -28.8] | 5 / 5 | 41–55% | -36.3; 5; 39–55% | -50.8; 5; 38–52% |

## Level 1 · Falcon3-1B

| Domain | Comparison | Primary ΔC [95%] | n / admitted | Coverage | A low / B high: ΔC; n; coverage | A high / B low: ΔC; n; coverage |
|---|---|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | +23.3 [+12.4, +34.3] | 5 / 5 | 32–42% | +5.9; 5; 13–30% | +42.8; 5; 16–20% |
| Graph | Final minus initial: GRPO | +9.8 [+2.6, +17.0] | 5 / 5 | 38–44% | -18.6; 5; 20–29% | +27.3; 5; 19–21% |
| Graph | Final minus initial: MaxRL | -11.3 | 3 / 5 | 38–41% | -8.3; 3; 16–18% | +3.9; 3; 17–20% |
| Graph | Final minus initial: ReplayDr.GRPO | -0.7 [-6.5, +5.0] | 5 / 5 | 29–35% | +4.8; 5; 10–12% | +4.5; 5; 12–14% |
| Graph | Final minus initial: ReplayMaxRL | -6.2 | 2 / 5 | 40–41% | -30.0; 2; 19–21% | -1.5; 2; 17–18% |
| Graph | Replay minus no replay: Dr.GRPO | -30.9 [-52.3, -9.4] | 5 / 5 | 24–50% | -12.9; 5; 12–21% | -45.2; 5; 12–29% |
| Graph | Replay minus no replay: MaxRL | -2.1 [-4.0, -0.2] | 5 / 5 | 74–78% | +10.0; 5; 39–47% | -1.3; 5; 42–48% |
| Countdown | Final minus initial: Dr.GRPO | +2.2 [-3.8, +8.2] | 5 / 5 | 5–6% | +0.0; 5; 2–4% | undefined; 0; 0% |
| Countdown | Final minus initial: GRPO | +18.0 [+12.8, +23.1] | 5 / 5 | 5–9% | +12.6; 5; 2–5% | undefined; 0; 0% |
| Countdown | Final minus initial: MaxRL | +18.2 | 2 / 5 | 9% | +14.3; 2; 5% | undefined; 0; 0% |
| Countdown | Final minus initial: ReplayDr.GRPO | +3.5 | 4 / 5 | 8–9% | +15.8; 4; 3–5% | undefined; 0; 0% |
| Countdown | Final minus initial: ReplayMaxRL | +13.5 | 1 / 5 | 9% | +14.3; 1; 5% | undefined; 0; 0% |
| Countdown | Replay minus no replay: Dr.GRPO | -16.8 | 4 / 4 | 26–33% | -13.9; 4; 18–27% | -23.3; 4; 22–27% |
| Countdown | Replay minus no replay: MaxRL | -6.3 [-7.9, -4.7] | 5 / 5 | 61–63% | -6.2; 5; 59–61% | -6.3; 5; 58–61% |
| Python | Final minus initial: Dr.GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: MaxRL | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: ReplayDr.GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: ReplayMaxRL | undefined | 0 / 5 | — | undefined; 0; — | undefined; 0; — |
| Python | Replay minus no replay: Dr.GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Replay minus no replay: MaxRL | -9.3 [-35.0, +16.5] | 5 / 5 | 17–100% | -3.9; 5; 17–100% | -9.0; 5; 17–100% |
| MathIR | Final minus initial: Dr.GRPO | +10.5 [+8.8, +12.2] | 5 / 5 | 5% | undefined; 0; 0% | +9.8; 5; 5% |
| MathIR | Final minus initial: GRPO | +5.4 [-9.9, +20.6] | 5 / 5 | 2–5% | undefined; 0; 0% | +23.3; 5; 1–5% |
| MathIR | Final minus initial: MaxRL | +27.8 | 1 / 5 | 5% | +100.0; 1; 1% | +13.3; 1; 4% |
| MathIR | Final minus initial: ReplayDr.GRPO | +8.6 [+2.6, +14.6] | 5 / 5 | 5–5% | undefined; 0; 0% | +11.1; 5; 2–3% |
| MathIR | Final minus initial: ReplayMaxRL | +16.7 | 2 / 5 | 4–5% | +100.0; 2; 1% | +0.0; 2; 2–3% |
| MathIR | Replay minus no replay: Dr.GRPO | +0.4 [-0.7, +1.5] | 5 / 5 | 7–11% | +0.0; 5; 4–6% | +0.0; 5; 5–5% |
| MathIR | Replay minus no replay: MaxRL | -0.1 [-1.0, +0.9] | 5 / 5 | 31–56% | +0.0; 5; 25–48% | +0.0; 5; 27–49% |
| PantryPlan | Final minus initial: Dr.GRPO | +88.6 [+88.6, +88.6] | 5 / 5 | 59% | +100.0; 5; 26% | +100.0; 5; 49% |
| PantryPlan | Final minus initial: GRPO | +86.3 [+79.8, +92.8] | 5 / 5 | 59% | +94.5; 5; 26% | +100.0; 5; 49% |
| PantryPlan | Final minus initial: MaxRL | +88.6 | 1 / 5 | 59% | +100.0; 1; 26% | +100.0; 1; 49% |
| PantryPlan | Final minus initial: ReplayDr.GRPO | +43.7 | 4 / 5 | 59–65% | +45.1; 4; 26–29% | +61.8; 4; 37–49% |
| PantryPlan | Final minus initial: ReplayMaxRL | +74.6 | 2 / 5 | 59–61% | +74.0; 2; 26% | +97.8; 2; 49% |
| PantryPlan | Replay minus no replay: Dr.GRPO | -40.4 [-60.0, -20.9] | 5 / 5 | 58–63% | -43.5; 5; 56–63% | -35.2; 5; 45–63% |
| PantryPlan | Replay minus no replay: MaxRL | -12.9 [-22.3, -3.5] | 5 / 5 | 63% | -20.3; 5; 63% | -4.3; 5; 63% |

## Level 1 · Qwen2.5-3B

| Domain | Comparison | Primary ΔC [95%] | n / admitted | Coverage | A low / B high: ΔC; n; coverage | A high / B low: ΔC; n; coverage |
|---|---|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | +27.4 [+23.2, +31.6] | 5 / 5 | 34–37% | +3.0; 5; 16–20% | +33.9; 5; 19–20% |
| Graph | Final minus initial: GRPO | +27.4 [+20.7, +34.1] | 5 / 5 | 36–38% | +4.6; 5; 19–22% | +36.4; 5; 20–23% |
| Graph | Final minus initial: MaxRL | +7.2 [-0.2, +14.5] | 5 / 5 | 35–43% | -20.8; 5; 14–21% | +21.5; 5; 17–21% |
| Graph | Final minus initial: ReplayDr.GRPO | -3.8 [-6.0, -1.6] | 5 / 5 | 38–41% | -22.7; 5; 16–19% | +12.1; 5; 18–19% |
| Graph | Final minus initial: ReplayMaxRL | -14.5 [-19.8, -9.3] | 5 / 5 | 36–41% | -22.0; 5; 16–18% | +16.5; 5; 14–20% |
| Graph | Replay minus no replay: Dr.GRPO | -27.3 [-29.9, -24.7] | 5 / 5 | 42–55% | -36.3; 5; 23–29% | -15.1; 5; 17–20% |
| Graph | Replay minus no replay: MaxRL | -16.2 [-26.1, -6.2] | 5 / 5 | 53–65% | -16.9; 5; 30–34% | +5.9; 5; 19–27% |
| Countdown | Final minus initial: Dr.GRPO | +16.0 [+10.9, +21.1] | 5 / 5 | 9–9% | +20.0; 5; 4% | -1.6; 5; 4–5% |
| Countdown | Final minus initial: GRPO | +18.2 [+18.2, +18.2] | 5 / 5 | 9% | +20.0; 5; 4% | +0.0; 5; 4% |
| Countdown | Final minus initial: MaxRL | +5.6 [-4.6, +15.9] | 5 / 5 | 7–9% | +5.0; 5; 2–3% | +0.0; 5; 3–5% |
| Countdown | Final minus initial: ReplayDr.GRPO | +2.0 [-6.0, +9.9] | 5 / 5 | 9–12% | +16.9; 5; 4–5% | -7.1; 5; 4–5% |
| Countdown | Final minus initial: ReplayMaxRL | -10.0 [-14.5, -5.5] | 5 / 5 | 8–9% | +0.0; 5; 2–3% | -5.0; 5; 5–6% |
| Countdown | Replay minus no replay: Dr.GRPO | -13.7 [-18.3, -9.0] | 5 / 5 | 55–59% | -15.3; 5; 50–55% | -13.9; 5; 51–55% |
| Countdown | Replay minus no replay: MaxRL | -10.5 [-12.6, -8.4] | 5 / 5 | 44–59% | -11.3; 5; 39–56% | -9.5; 5; 38–55% |
| Python | Final minus initial: Dr.GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: MaxRL | +0.0 [+0.0, +0.0] | 5 / 5 | 2–7% | +0.0; 2; 0–2% | +0.0; 5; 1–3% |
| Python | Final minus initial: ReplayDr.GRPO | undefined | 0 / 5 | 0% | undefined; 0; 0% | undefined; 0; 0% |
| Python | Final minus initial: ReplayMaxRL | -2.3 [-5.0, +0.3] | 5 / 5 | 7% | +0.0; 5; 2% | -4.0; 5; 3% |
| Python | Replay minus no replay: Dr.GRPO | -19.0 | 2 / 5 | 0–17% | -22.1; 2; 0–17% | -14.1; 2; 0–17% |
| Python | Replay minus no replay: MaxRL | -15.3 [-26.5, -4.1] | 5 / 5 | 17–63% | -20.3; 5; 17–63% | -8.3; 5; 17–63% |
| MathIR | Final minus initial: Dr.GRPO | +0.0 [+0.0, +0.0] | 5 / 5 | 9–11% | +0.0; 5; 3–5% | +0.0; 5; 7–9% |
| MathIR | Final minus initial: GRPO | +0.0 [+0.0, +0.0] | 5 / 5 | 7–9% | +0.0; 5; 4–5% | +0.0; 5; 6–7% |
| MathIR | Final minus initial: MaxRL | +0.0 [+0.0, +0.0] | 5 / 5 | 5–11% | +0.0; 5; 3–5% | +0.0; 5; 4–9% |
| MathIR | Final minus initial: ReplayDr.GRPO | -1.2 | 4 / 5 | 11% | +0.0; 4; 5–8% | +0.0; 4; 9–9% |
| MathIR | Final minus initial: ReplayMaxRL | +0.0 [+0.0, +0.0] | 5 / 5 | 9–12% | +0.0; 5; 5–6% | +0.0; 5; 7–10% |
| MathIR | Replay minus no replay: Dr.GRPO | -1.0 [-2.2, +0.3] | 5 / 5 | 18–27% | -1.7; 5; 15–23% | +0.3; 5; 16–22% |
| MathIR | Replay minus no replay: MaxRL | +0.0 [+0.0, +0.0] | 5 / 5 | 16–21% | +0.0; 5; 13–19% | +0.0; 5; 16–19% |
| PantryPlan | Final minus initial: Dr.GRPO | +44.7 [+26.6, +62.7] | 5 / 5 | 38–52% | +47.8; 5; 21–27% | +24.5; 5; 19–24% |
| PantryPlan | Final minus initial: GRPO | +46.9 [+41.1, +52.6] | 5 / 5 | 38–48% | +51.4; 5; 20–29% | +27.1; 5; 19–20% |
| PantryPlan | Final minus initial: MaxRL | +43.1 [+29.7, +56.5] | 5 / 5 | 46–54% | +39.4; 5; 27–30% | +24.6; 5; 20–22% |
| PantryPlan | Final minus initial: ReplayDr.GRPO | +18.4 [+8.1, +28.8] | 5 / 5 | 50–55% | +8.7; 5; 28–30% | +3.7; 5; 20–22% |
| PantryPlan | Final minus initial: ReplayMaxRL | +14.2 [+1.8, +26.6] | 5 / 5 | 53–59% | +4.6; 5; 30–34% | -1.5; 5; 21–25% |
| PantryPlan | Replay minus no replay: Dr.GRPO | -30.0 [-42.0, -17.9] | 5 / 5 | 55–65% | -33.7; 5; 55–63% | -30.5; 5; 53–64% |
| PantryPlan | Replay minus no replay: MaxRL | -29.2 [-31.9, -26.5] | 5 / 5 | 63–70% | -35.5; 5; 62–69% | -23.7; 5; 61–68% |

## Level 2 · Qwen2.5-0.5B

| Domain | Comparison | Primary ΔC [95%] | n / admitted | Coverage | A low / B high: ΔC; n; coverage | A high / B low: ΔC; n; coverage |
|---|---|---:|---:|---:|---:|---:|
| Graph | Final minus initial: Dr.GRPO | +62.5 [+29.5, +95.5] | 5 / 5 | 2–8% | +63.3; 2; 0–1% | +86.0; 5; 2–3% |
| Graph | Final minus initial: MaxRL | +45.2 [+38.7, +51.8] | 5 / 5 | 5–8% | +76.7; 4; 0–1% | +48.4; 5; 1–4% |
| Graph | Final minus initial: ReplayDr.GRPO | -20.4 [-28.2, -12.6] | 5 / 5 | 7–9% | +3.3; 5; 1% | -37.2; 5; 1–3% |
| Graph | Final minus initial: ReplayMaxRL | -12.8 [-26.2, +0.6] | 5 / 5 | 7–9% | +33.3; 5; 1% | -26.7; 5; 1–2% |
| Graph | Replay minus no replay: Dr.GRPO | -57.7 [-73.6, -41.8] | 5 / 5 | 7–20% | -59.9; 5; 3–12% | -54.4; 5; 2–10% |
| Graph | Replay minus no replay: MaxRL | -40.5 [-47.2, -33.9] | 5 / 5 | 22–33% | -26.2; 5; 9–20% | -41.6; 5; 9–20% |
| Countdown | Final minus initial: Dr.GRPO | +0.0 | 1 / 5 | 0–1% | undefined; 0; 0% | undefined; 0; 0% |
| Countdown | Final minus initial: MaxRL | +0.0 | 1 / 5 | 0–1% | undefined; 0; 0% | undefined; 0; 0% |
| Countdown | Final minus initial: ReplayDr.GRPO | +0.0 [+0.0, +0.0] | 5 / 5 | 1% | undefined; 0; 0% | undefined; 0; 0% |
| Countdown | Final minus initial: ReplayMaxRL | +0.0 | 3 / 5 | 0–1% | undefined; 0; 0% | undefined; 0; 0% |
| Countdown | Replay minus no replay: Dr.GRPO | -23.0 | 4 / 5 | 0–7% | -28.1; 4; 0–5% | -29.2; 4; 0–7% |
| Countdown | Replay minus no replay: MaxRL | -7.3 | 4 / 5 | 0–8% | +0.0; 3; 0–8% | -11.8; 4; 0–8% |
| Python | Final minus initial: Dr.GRPO | +30.2 [+30.2, +30.2] | 5 / 5 | 25% | +0.0; 5; 7% | +37.8; 5; 12% |
| Python | Final minus initial: MaxRL | +30.2 [+30.2, +30.2] | 5 / 5 | 25% | +0.0; 5; 7% | +37.8; 5; 12% |
| Python | Final minus initial: ReplayDr.GRPO | +9.4 [-25.9, +44.8] | 5 / 5 | 25% | -22.4; 5; 7% | +14.6; 5; 12% |
| Python | Final minus initial: ReplayMaxRL | +19.9 [-8.8, +48.5] | 5 / 5 | 25% | -7.0; 5; 7% | +25.8; 5; 12% |
| Python | Replay minus no replay: Dr.GRPO | -20.9 [-56.3, +14.6] | 5 / 5 | 81% | -21.9; 5; 81% | -23.4; 5; 81% |
| Python | Replay minus no replay: MaxRL | -10.6 [-39.7, +18.4] | 5 / 5 | 81% | -7.6; 5; 81% | -12.1; 5; 81% |
| MathIR | Final minus initial: Dr.GRPO | +0.0 | 3 / 5 | 0–4% | undefined; 0; 0% | +0.0; 3; 0–4% |
| MathIR | Final minus initial: MaxRL | +0.0 | 4 / 5 | 0–5% | undefined; 0; 0% | +0.0; 4; 0–4% |
| MathIR | Final minus initial: ReplayDr.GRPO | +0.0 [+0.0, +0.0] | 5 / 5 | 5% | undefined; 0; 0% | +0.0; 5; 4% |
| MathIR | Final minus initial: ReplayMaxRL | +0.0 | 4 / 5 | 5% | undefined; 0; 0% | +0.0; 4; 4% |
| MathIR | Replay minus no replay: Dr.GRPO | +0.0 [+0.0, +0.0] | 5 / 5 | 9–95% | +0.0; 5; 9–95% | +0.0; 5; 9–95% |
| MathIR | Replay minus no replay: MaxRL | -0.1 [-0.4, +0.2] | 5 / 5 | 23–98% | +0.0; 5; 22–95% | -0.1; 5; 23–98% |
| PantryPlan | Final minus initial: Dr.GRPO | -86.7 | 2 / 5 | 1–2% | -50.0; 1; 0–2% | -100.0; 1; 0–1% |
| PantryPlan | Final minus initial: MaxRL | +0.0 | 1 / 5 | 1% | undefined; 0; 0% | undefined; 0; 0% |
| PantryPlan | Final minus initial: ReplayDr.GRPO | -4.9 | 2 / 5 | 12–12% | -7.5; 2; 8% | -5.0; 2; 2% |
| PantryPlan | Final minus initial: ReplayMaxRL | +9.2 | 1 / 5 | 15% | +9.7; 1; 9% | -22.2; 1; 2% |
| PantryPlan | Replay minus no replay: Dr.GRPO | +49.0 | 1 / 2 | 0–12% | +65.6; 1; 0–5% | +56.7; 1; 0–4% |
| PantryPlan | Replay minus no replay: MaxRL | +100.0 | 1 / 1 | 1% | undefined; 0; 0% | undefined; 0; 0% |

## Eligibility, missing estimates and source issues

- `level1/qwen05b/countdown/before_after/maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['47']; issues `[]`.
- `level1/qwen05b/python_factors/before_after/drgrpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '44', '45', '46', '47']; issues `[]`.
- `level1/qwen05b/python_factors/before_after/grpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '44', '45', '46', '47']; issues `[]`.
- `level1/qwen05b/python_factors/before_after/maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '44', '45', '46', '47']; issues `[]`.
- `level1/qwen05b/python_factors/before_after/replay_drgrpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '44', '45', '46', '47']; issues `[]`.
- `level1/qwen05b/python_factors/before_after/replay_maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '44', '45', '46', '47']; issues `[]`.
- `level1/falcon1b/graph_coloring/before_after/maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['55', '58']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 55}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}]`.
- `level1/falcon1b/graph_coloring/before_after/replay_maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['55', '58', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 55}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/countdown/before_after/maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['56', '58', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 56}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/countdown/before_after/replay_drgrpo`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": false, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/countdown/before_after/replay_maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['56', '57', '58', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 56}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 57}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/python_factors/before_after/drgrpo`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints []; zero eligible prompts in seeds ['55', '56', '57', '58', '59']; issues `[]`.
- `level1/falcon1b/python_factors/before_after/grpo`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints []; zero eligible prompts in seeds ['55', '56', '57', '58', '59']; issues `[]`.
- `level1/falcon1b/python_factors/before_after/maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['55', '57', '58', '59']; zero eligible prompts in seeds ['56']; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 55}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 57}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/python_factors/before_after/replay_drgrpo`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints []; zero eligible prompts in seeds ['55', '56', '57', '58', '59']; issues `[]`.
- `level1/falcon1b/python_factors/before_after/replay_maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['55', '56', '57', '58', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 55}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 56}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 57}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/python_factors/replay_effect/drgrpo`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints []; zero eligible prompts in seeds ['55', '56', '57', '58', '59']; issues `[]`.
- `level1/falcon1b/mathir/before_after/maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['55', '56', '58', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 55}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 56}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/mathir/before_after/replay_maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['55', '58', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 55}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/pantry_plan/before_after/maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['56', '57', '58', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 56}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 57}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 58}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/falcon1b/pantry_plan/before_after/replay_drgrpo`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['55']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 55}]`.
- `level1/falcon1b/pantry_plan/before_after/replay_maxrl`: admitted seeds [55, 56, 57, 58, 59]; missing paired endpoints ['56', '57', '59']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 56}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 57}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 59}]`.
- `level1/qwen3b/python_factors/before_after/drgrpo`: admitted seeds [70, 71, 72, 73, 74]; missing paired endpoints []; zero eligible prompts in seeds ['70', '71', '72', '73', '74']; issues `[]`.
- `level1/qwen3b/python_factors/before_after/grpo`: admitted seeds [70, 71, 72, 73, 74]; missing paired endpoints []; zero eligible prompts in seeds ['70', '71', '72', '73', '74']; issues `[]`.
- `level1/qwen3b/python_factors/before_after/replay_drgrpo`: admitted seeds [70, 71, 72, 73, 74]; missing paired endpoints []; zero eligible prompts in seeds ['70', '71', '72', '73', '74']; issues `[]`.
- `level1/qwen3b/python_factors/replay_effect/drgrpo`: admitted seeds [70, 71, 72, 73, 74]; missing paired endpoints []; zero eligible prompts in seeds ['70', '71', '73']; issues `[]`.
- `level1/qwen3b/mathir/before_after/replay_drgrpo`: admitted seeds [70, 71, 72, 73, 74]; missing paired endpoints ['71']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 71}]`.
- `level2/qwen05b/countdown/before_after/drgrpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '44', '45', '46']; issues `[]`.
- `level2/qwen05b/countdown/before_after/maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '44', '46', '47']; issues `[]`.
- `level2/qwen05b/countdown/before_after/replay_maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '47']; issues `[]`.
- `level2/qwen05b/countdown/replay_effect/drgrpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['46']; issues `[]`.
- `level2/qwen05b/countdown/replay_effect/maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['44']; issues `[]`.
- `level2/qwen05b/mathir/before_after/drgrpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['43', '46']; issues `[]`.
- `level2/qwen05b/mathir/before_after/maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints []; zero eligible prompts in seeds ['46']; issues `[]`.
- `level2/qwen05b/mathir/before_after/replay_maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints ['47']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 47}]`.
- `level2/qwen05b/pantry_plan/before_after/drgrpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints ['44', '46', '47']; zero eligible prompts in seeds []; issues `[{"a_available": true, "b_available": false, "kind": "endpoint_unavailable", "seed": 44}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 46}, {"a_available": true, "b_available": false, "kind": "endpoint_unavailable", "seed": 47}]`.
- `level2/qwen05b/pantry_plan/before_after/maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints ['44', '45', '46', '47']; zero eligible prompts in seeds []; issues `[{"a_available": true, "b_available": false, "kind": "endpoint_unavailable", "seed": 44}, {"a_available": true, "b_available": false, "kind": "endpoint_unavailable", "seed": 45}, {"a_available": false, "b_available": false, "kind": "endpoint_unavailable", "seed": 46}, {"a_available": false, "b_available": false, "kind": "endpoint_unavailable", "seed": 47}]`.
- `level2/qwen05b/pantry_plan/before_after/replay_drgrpo`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints ['43', '44', '45']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 43}, {"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 44}, {"a_available": false, "b_available": false, "kind": "endpoint_unavailable", "seed": 45}]`.
- `level2/qwen05b/pantry_plan/before_after/replay_maxrl`: admitted seeds [43, 44, 45, 46, 47]; missing paired endpoints ['44', '45', '46', '47']; zero eligible prompts in seeds []; issues `[{"a_available": false, "b_available": true, "kind": "endpoint_unavailable", "seed": 44}, {"a_available": true, "b_available": false, "kind": "endpoint_unavailable", "seed": 45}, {"a_available": false, "b_available": false, "kind": "endpoint_unavailable", "seed": 46}, {"a_available": true, "b_available": false, "kind": "endpoint_unavailable", "seed": 47}]`.
- `level2/qwen05b/pantry_plan/replay_effect/drgrpo`: admitted seeds [43, 46]; missing paired endpoints []; zero eligible prompts in seeds ['43']; issues `[]`.

The source artifact retains sample-integrity issues for 53 cells. All source checkpoint statuses and issue messages are exported in `source_availability.csv`.

## Interpretation limits

- C is correct-key pair collision on the stated observable prompt population. Positive delta C means greater concentration; it does not establish extinction of unobserved modes.
- The 32 saved positions map to 11 nominal child-seed streams under the audited vLLM V0 n=8 rule (parent seed plus output index). The earliest draw/output representative is selected without inspecting values.
- Archived actor source is available for 150 of 400 primary cells and 0 of 75 GRPO cells. Historical vLLM dependency identity is not cryptographically established for every run. The nominal child-stream interpretation therefore remains conditional where historical sampling identity is not established.
- The per-prompt collision U-statistic has its usual conditional-distribution identity under iid correct labels. Shared nominal streams across compared conditions make jointly eligible primary means descriptive.
- The two disjoint orientations allocate lower/upper nominal streams to opposite conditions (five/six when eleven streams are shared). They reduce same-prompt cross-condition coupling, have different eligibility populations, and are shown separately. Reused seed IDs across prompts still limit an iid population interpretation.
- Training estimates weight jointly eligible prompts equally. Hosted collision pools correct-pair counts and therefore gives greater weight to prompts with more correct pairs; the two aggregates are intentionally different.
- Collision uses the selected nominal streams. Original mean@8, pass@8, distinct@8 and extra-mode metrics retain the original intact K8 groups, even when reported on the collision-eligible prompt population.
- Intervals describe variability among five measured training-seed estimates. They are nominal and unadjusted, not a simultaneous guarantee or proof of sampler independence. Partial and undefined blocks remain visible.

## Provenance

- Source: `paper/results/conditional_concentration_20260911.json`.
- Source SHA-256: `5c9f69d6961ba7f6938a9a535bf7414c352fe645f668a241bbcdae6637e262d4`.
- Analysis code SHA-256: `550ebf40fd701bf101176c6b980a70cb70e0dd66102b32eec5c6b24e712223f0`.
- The source JSON binds the protocol, stream amendment, any cohort extension, sample cache and source manifests. The renderer does not change any measurements or regrade outputs.
