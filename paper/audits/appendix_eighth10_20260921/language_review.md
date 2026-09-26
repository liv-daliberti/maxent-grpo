# Independent editorial review: incoming PDF pages 89–98

Reviewed the incoming page extracts (`page-89.txt` through `page-98.txt`), the incoming `before.tex`, the rewritten `paper/main.tex`, the two O.6 fragments, and the compiled `after-page-089.txt` through `after-page-098.txt`. This was a read-only review of manuscript sources; this file is the only artifact written by this reviewer. Mathematical validity and external-source theorem attribution are outside this language review.

## Finding

The revised pages have no material remaining editorial/process-history wording, old “reasoning modes” terminology, broken `??` references, or caption/population mismatch. The only captions in this range are Tables 56 and 57; there are no figure captions to recast. Both tables were already revised in the preceding pass and follow the main-body caption structure: bold result, sampling setup, metric/population definitions, and uncertainty status. The subsequent material is mathematical prose and theorem/proof environments.

## Specific incoming issues and revised resolution

| Incoming location | Issue | Required interpretation or wording | Revised source status |
| --- | --- | --- | --- |
| p. 91, lines 4865–4873, P introduction | Extensive narration of how the analysis applies established tools and what the authors derive obscures the concrete result. | Lead with the mean-gradient identity, categorical concentration, and replay-retention conditions; retain the limit concerning AdamW/PPO trajectories. | `main.tex:3253` now does this directly. The scientific scope limitation remains. |
| p. 91, lines 4877–4879, P.1 | Narrative about what “results below state separately” can become a direct model distinction. | State that categorical logits, sampled updates, and neural policies have distinct update assumptions. | `main.tex:3267` gives this distinction without workflow narration. |
| p. 91, line 4881, and p. 93, line 4981 | “Correct execution modes” differs from the requested “solution modes” terminology. | Use “correct solution modes,” while preserving the verifier/canonical-category definition. | Both occurrences are removed in the revised rendered range. The assumption uses “solution modes” at `main.tex:3276`. |
| p. 91, lines 4897–4898, P.1 | “AdamW and PPO violate (A3)” compresses two different implementation differences into one phrase; reference KL is omitted from the explanation of (A4). | Separate optimizer preconditioning/momentum from finite repeated PPO updates, and include both KL and entropy under (A4). | `main.tex:3297`–3302 now makes these distinctions. |
| p. 92, lines 4944–4945, before Lemma P.3 | “The next lemma specializes… We invoke that result directly and retain the normalization…” is author-process narration. | “The practical MaxRL estimator has the same mean-gradient direction… Its all-failure convention determines the finite-group coefficient.” | This direct result/convention wording appears at `main.tex:3383`. |
| p. 93, lines 4974–4976, P.2 opening | “The result is a specialization… not a new general theorem” is novelty/positioning narration. | Identify the replicator system and cite it; retain the assumptions, without commenting on manuscript novelty. | `main.tex:3440`–3448 does this. |
| p. 94, lines 5024–5027, P.3 opening | “Invites the question…” and “The reason is one line…” describe exposition rather than the result. | State the shared conditional trajectory and why reward-only advantages do not distinguish identities of correct modes. | Direct wording at `main.tex:3533`–3538. |
| p. 94, lines 5051–5062 | “Checkable by inspection” and “the collapse theorem was never about the three estimators it was stated for” are rhetorical/process language. | State the Bernstein bounds and sufficient nonnegative-gap criterion, with the estimator examples. | Rewritten at `main.tex:3581`–3594. The rhetoric is absent. |
| p. 95, Corollary P.8, line 5093 | “What a stronger fresh objective does buy” is conversational and imprecise as a result title. | “MaxRL amplification of the correctness gradient.” | Revised title at `main.tex:3660`; the coefficient comparison remains explicit. |
| p. 95, lines 5109–5112, P.4 opening | “The extension here gives… then states what survives… This subsection replaces…” narrates the presentation. | State the neural mean/covariance result and the actual limitations from finite steps, shared prompts, and response weighting. | `main.tex:3687`–3696 now does this. |
| p. 97, lines 5234–5237, P.5 opening | “We now connect… extend… quantify…” is roadmap narration. | State that categorical diversity decreases under the specified mean and finite expected updates, while sampled updates can break symmetry and finite budgets limit observation. | Direct result-oriented wording at `main.tex:3944`–3951. |

The line references above identify the source reviewed; later layout-only adjustments may shift them.

## O.6 and caption checks

- **Table 56, p. 89:** the bold lead is supported by the different point estimates. The caption specifies 32 prompts per domain/level, eight responses, normalized grading, equal-domain `pass@8` and `distinct@8`, and the distinct jointly eligible population for promptwise PCMD. It says that eligibility does not match overall accuracy and that entries are point estimates. No claim of a uniform conditional-diversity decline remains.
- **Table 57, p. 90:** the caption distinguishes all 160 prompts at each level and all 480 overall from 65–159 and 245–458 jointly eligible prompts. It correctly warns that PCMD weights differ from the raw success/mode columns. It reports the absence of intervals rather than implying statistical significance.
- **O.6 method text:** retaining “collected at different times” is necessary because collection time and provider defaults can confound the requested-control comparison. This is an experimental limitation, not manuscript/process history to remove. The same applies to the missing DeepSeek response, differing strict grading for its descriptive estimate, Opus 5’s thinking-block control violation, and unknown shared randomness.
- **O.6 interpretation:** raw distinct-mode counts and conditional PCMD remain separate. The text explicitly notes increases for Grok/Kimi, small positive Opus 4.8 differences, the GPT-5.4 sign reversal by level, and the lack of uncertainty intervals. It does not infer equal compute, absence of internal computation, or a causal effect of training.

## Terminology, references, and contents

A contextual scan of both incoming and rewritten rendered pages covered “post hoc,” “frozen,” “registered,” admission/audit/reproduction terms, manuscript/submission/novelty phrases, old reasoning-mode terminology, “execution modes,” `??`, and nonexistent “App. Q.” The rewritten range contains none of the inappropriate terms or unresolved references. The collection-time limitation noted above was the only retained temporal-method phrase matched by the scan.

O.6 and P.1–P.5 all have matching manual contents entries (`main.tex:762` and `765`–`769`) with the same visible titles and label-based page references. The categorical model, finite steps, stochastic updates, and neural-policy limits remain distinguishable. Proof labels and mathematical terms such as “mean flow,” “detached advantage,” “conditional score,” and “fixed-bank replay” are substantive technical language, not process language.

The local citation placement now assigns the Dr.GRPO abstraction to `liu2025understanding` and reward-standardized GRPO to `shao2024deepseekmath` (`main.tex:3319` onward). This checks visible attribution consistency, not the content of the cited papers.

## Minor presentation suggestion

At `main.tex:3540`, P.3 has `\paragraph{Proof of Theorem~\ref{thm:objective-inertness}.}` immediately before `\begin{proof}`, producing two consecutive proof labels. Optional simplification:

```tex
\begin{proof}[Proof of Theorem~\ref{thm:objective-inertness}]
```

This removes the duplicate presentation label without changing mathematics. It was reported to the root editor; this reviewer did not edit the source.

## Outside-range consistency note

The main-body sentence at `main.tex:355`, “Disabling reasoning reduces both correctness and diversity,” is broader than the corrected O.6 results if “diversity” denotes the paper’s conditional PCMD endpoint. A precise later replacement is:

> Disabling explicit reasoning reduces overall correctness and mean verified mode counts, while conditional diversity changes differ by deployment.

The appendix already makes that distinction. This review did not modify the main body or expand the current page scope.
