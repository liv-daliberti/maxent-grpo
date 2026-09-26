# Source audit for the user's authoritative main

Audit date: 2026-09-22. The authoritative text is [user_main_exact.tex](user_main_exact.tex). User instructions are to repair links and citations only and to flag factual mismatches separately. The observations below do not authorize changes to prose, numbers, captions, or claims.

## Link and citation repairs

- The levels figure is present at `paper/figures/qwen_level_trends.pdf`; the supplied `iclr2027_overleaf/figures/qwen_level_trends.pdf` is not a valid path relative to `paper/`.
- `brown2024large` names the same paper already present as `brown2024monkeys`: *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*, [arXiv:2407.21787](https://arxiv.org/abs/2407.21787).
- `yue2025does` names the paper already present as `yue2025rlvrlimit`: *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, [arXiv:2504.13837](https://arxiv.org/abs/2504.13837).
- `anschel2025group` names the paper already present as `anschel2025groupaware`: *Group-Aware Reinforcement Learning for Output Diversity in Large Language Models*, [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1649/).
- The replay-method paragraph attributes Dr.GRPO to `shao2024deepseekmath`; its correct existing citation is `liu2025understanding`. The Shao citation is the GRPO/DeepSeekMath source.
- For the five-domain recovery claim, `app:withdrawal-recovery` is the relevant appendix, whereas `app:pantry-adaptation` covers a separate Pantry-only study. `app:portfolio-withdrawals` remains the appropriate portfolio-survival source.
- The matched reasoning-control table is `tab:hosted-reasoning-matched-levels` in `paper/results/hosted_reasoning_off_20260912_appendix.tex`. `tab:hosted-level-averages` and `fig:hosted-verified-breadth` describe broader hosted summaries, not the matched reasoning-on/off experiment.

## Ten bibliography entries added

All existing bibliography bytes were preserved as an unchanged prefix. No fabricated technical-report titles or metadata were used. Provider pages are cited as announcements or documentation; their existence does not independently authenticate the paper's experiment outputs.

| Key | Verified primary source and metadata |
| --- | --- |
| `DBLP:journals/corr/MnihKSGAWR13` | Mnih et al., *Playing Atari with Deep Reinforcement Learning*, submitted December 19, 2013, [arXiv:1312.5602](https://arxiv.org/abs/1312.5602). This is the 2013 paper, not the different 2015 Nature article. |
| `allal2025smollm2` | Ben Allal et al., *SmolLM2: When Smol Goes Big—Data-Centric Training of a Small Language Model*, submitted February 4, 2025, [arXiv:2502.02737](https://arxiv.org/abs/2502.02737). |
| `yang2024qwen25` | Qwen et al., *Qwen2.5 Technical Report*, first submitted December 19, 2024, [arXiv:2412.15115](https://arxiv.org/abs/2412.15115). |
| `falconllm2024falcon3` | Falcon-LLM Team, *The Falcon 3 Family of Open Models*, December 17, 2024. The title and collective author follow the provider's suggested citation on its [official release page](https://falcon-lm.github.io/blog/falcon-3/). |
| `olmo20242olmo2furious` | Team OLMo et al., *2 OLMo 2 Furious*, first submitted December 31, 2024, [arXiv:2501.00656](https://arxiv.org/abs/2501.00656). The arXiv identifier begins 2501 despite the 2024 first submission; the entry explicitly notes that date. |
| `openai2026gpt56` | OpenAI, *GPT-5.6: Frontier Intelligence That Scales with Your Ambition*, July 9, 2026, [official announcement](https://openai.com/index/gpt-5-6/). Explicitly includes Sol. |
| `openai2026gpt54` | OpenAI, *Introducing GPT-5.4*, March 5, 2026, [official announcement](https://openai.com/index/introducing-gpt-5-4/). |
| `xai2026grok43` | xAI, *Grok 4.3: Model Documentation*, [official model page](https://docs.x.ai/developers/models/grok-4.3). No invented release date or research-paper claim. |
| `kimiteam2026kimik3` | Moonshot AI, *Kimi K3: Open Frontier Intelligence*, [official technical blog](https://www.kimi.ai/blog/kimi-k3). July 16, 2026 date is listed beside its link on the [provider's research homepage](https://www.moonshot.ai/). |
| `anthropic2026opus48` | Anthropic, *Introducing Claude Opus 4.8*, May 28, 2026, [official announcement](https://www.anthropic.com/news/claude-opus-4-8). |

All web sources were checked on September 22, 2026; online-provider entries include that access date. The bibliography contains no duplicate entry keys. The three remaining originally missing keys are the aliases to existing entries listed above, to be repaired in the main by the integrating agent.

## Factual mismatches left unchanged

Locations below refer to the stable supplied `user_main_exact.tex`, rather than changing line numbers in the assembled paper.

1. **Frontier PCMD number, line 195.** The sentence states mean PCMD `0.991` and calls it below `0.57`. `paper/results/mode_diversity_coverage.tex` defines `MDfrontierpass=.991` and `MDfrontierpmdmean=.232`; the first value is correctness, not PCMD. This is both a numeric source mismatch and an internal numerical contradiction.

2. **Rollout-signal method attribution, line 361.** The supplied text attributes the 18.5% versus 1.5% example to Re:Max. The table under `app:degenerate-groups` explicitly labels these columns Re:Dr and Dr.GRPO, across all 3,072 Qwen2.5-0.5B updates and five seeds. The appendix supports the example for Re:Dr, not the stated Re:Max attribution.

3. **KL figure population and color meaning, lines 439 and 451–455.** The surrounding paragraph says four domains and the caption says five. `ops/plot_paper_reference_kl_plane.py` selects the three domains with reportable initial PCMD: Graph, MathIR, Pantry. `app:theory-kl-measured` explicitly confirms these three. The green area is above the initial policy on both pass@8 and PCMD; it does not encode comparison to the KL curve at fixed diversity. A separate all-five-domain matched-diversity calculation exists in `paper/results/reference_kl_macros.tex` (`MDklMatchedDomains=5`, beta `.04`, PCMD `.394` for KL and Re:Max, pass@8 `.641` versus `.772`), but it is a different aggregate from the displayed figure.

4. **Reasoning-off direction, line 216.** The phrase saying disabled reasoning costs correctness and PCMD together is not uniform across deployments. In `tab:hosted-reasoning-matched-levels`, Grok's overall PCMD increases `.154 → .364`, Kimi's `.293 → .426`, and Opus 4.8's `.202 → .209` when reasoning is disabled. GPT-5.6 Sol and GPT-5.4 decline. Correctness declines overall, but conditional-diversity directions vary.

5. **Hosted cohort, line 297.** The five listed deployments are real and have verified primary citations. However, `paper/results/frontier_comparison_20260911_protocol.tex` describes seven evaluated deployments, adding Claude Opus 5 and DeepSeek V4 Pro. The introduction's generated `MDfrontierdeployments` is also seven. The five-model list matches the principal matched reasoning-off study, not the entire main hosted cohort.

6. **Comprehensive solution catalogues, line 134.** Verifiers compute canonical keys from accepted outputs or execution; complete pre-enumeration is not generally required. `app:domain-overview`, `tab:tasks`, and the high-budget appendices distinguish certified catalogue counts from the verifier's full support. Countdown accepts unary negation beyond its enumerated binary-expression catalogue. The claim of an advance comprehensive set overstates the construction.

7. **Constant mode distributions across levels, lines 192–195.** `app:data-levels` states that construction reserves differ from native Level-1 terminal test sets. Displayed Graph, Countdown, and Python evaluations consequently have different valid-key histograms across levels. The key definition and verifier remain fixed; the evaluated support distribution need not. The appendix also explains calibration tolerances and the Level-4 MathIR exception rather than exact score equality.

8. **Effective-mode scope, line 102.** `app:hosted-mode-diversity` defines the headline effective-mode numbers as `1/(1 − macro PCMD)` over the thirteen domain–level cells reportable for every deployment. They are not bounds for each individual problem, averages of per-cell effective mode counts, or counts of all discovered solutions. The supplied unqualified wording does not carry that aggregate scope.

9. **No correctness difference in weighting ablation, line 382.** `app:frequency-replay-progress` reports the five-domain uniform-minus-frequency effect as `+.010` pass@8 with a nominal interval `[-.119,+.139]`, explicitly establishing neither a gain nor equivalence. The supplied wording can be read as establishing equivalence; the evidence is inconclusive.

10. **Correlation wording, line 144.** The figure illustrates that greater performance/scale does not guarantee more diversity. It does not establish zero association. For the broader frozen grid, `paper/results/mode_diversity_coverage.tex` reports `MDpmdcorr=-.332`; this is a different population from a formal test of the plotted subset, but it is another reason not to infer a universal zero correlation from the figure.

11. **Unspecified eligibility threshold, line 175.** The sentence stops at “a minimum of correct solutions” without specifying the number. `app:metric-sampling` distinguishes two verified responses needed per prompt from the thirty-eligible-prompt reporting threshold for frozen/hosted summaries (other comparisons have their own reporting rule). This remains a prose omission after correcting the appendix link.

12. **Fourteen of fifteen “domains,” line 321.** The figure has five domains crossed with three objectives. Fourteen of fifteen is the number of objective–domain comparisons for Qwen2.5-3B, not fifteen domains. The supporting figure itself is correctly present.

13. **Sampling-budget assertion, line 216.** The supplied claim that quadrupling the budget moves the mode count by less than `1/1000` is stronger and metrically different from the current appendix description. `app:decoding-objection` reports PCMD ranges (e.g. `.006–.031` in Countdown and `.037–.069` in Pantry across the eleven-setting grid), changes populations by eligibility, and uses a separate twelve-pass cohort for its larger-group settings. This does not establish a universal <.001 change in the number of modes.

14. **Conditional versus raw portfolio survival, lines 424–425.** The all-five-domain claim has support for pooled raw eight-response survival means in `tab:withdrawal-contrasts`; it should not be interpreted as a universal gain at fixed correct-draw budget or in every matched comparison. The appendix reports 39/40 positive raw contrasts and 28/39 positive four-verified-draw contrasts. At four verified draws, Re:Max Countdown and MathIR intervals include zero. The narrower raw interpretation is supported; no rewrite was applied.

15. **Literal drafting placeholder, line 265 — resolved under subsequent user authorization.** The user explicitly authorized filling both `------` fields. Discovery frequency describes how often a mode appears in fresh rollouts; rehearsal frequency describes how often a stored mode is replayed. This authorized clarification supersedes the original requirement to preserve those two placeholders. All other unresolved factual flags remain unchanged.

The user also explicitly authorized expressing the comparator paragraph's four raw gains as percentages: `.653`, `.497`, `.358`, and `.320` become **65.3, 49.7, 35.8, and 32.0 percentage points**, respectively. Both pass@8 and PCMD are probabilities on a zero-to-one scale, so multiplying their absolute differences by 100 gives percentage-point gains. These are not relative percentage increases, which would require division by the comparator's baseline value.

16. **Teaser sampling unit and runs, line 74.** The caption says “32 sampled questions, eight from each of four matched runs.” `paper/FIGURE_MANIFEST.md` identifies the active source as `ops/plot_paper_collapse_toy.py` and `var/artifacts/paper_graph_collapse_toy.json`. That JSON specifies a single selected Graph prompt (index 84, instance `eval_multi_answer-10000-84`) and one matched training seed, 70. Its `sampling` record gives four evaluation draws of eight responses, hence 32 responses to that one prompt per checkpoint and arm. Seeds 70–73 are the selection pool, not four displayed runs. The source also labels this as a post-hoc illustration.

17. **Additional training update, line 291.** `app:algorithm-control` explicitly says that fresh-task and replay gradients accumulate before the same optimizer step. The `app:algorithm-update` pseudocode also describes one optimizer update. Replay adds a likelihood-gradient contribution and associated scoring/backward computation; it does not add the optimizer step suggested by “an additional training update.” This conflicts with the matched optimizer-update budget in `tab:run-contract`.

18. **Mixtures versus either constituent, line 218.** `paper/results/inference_followups_20260912.tex`, under `app:cross-model-overlap`, reports all 21 pairs improving expected distinct modes over the average constituent by `.115–.266`. Only ten pairs improve over both constituents in point estimates, and nine have positive pointwise intervals against both. For example, `tab:cross-model-mixtures` reports Kimi K3 + Grok 4.3 at `-.080` relative to Kimi and `+.309` relative to Grok. The supplied “than either alone” needs a selected-pair interpretation; it does not hold for arbitrary evaluated pairs.

19. **Recovery conditioned on initial failure, line 427.** `tab:withdrawal-recovery` defines Calls over all withdrawals, assigning zero to an initially surviving portfolio and eight to an unresolved case. It therefore measures overall recovery burden, not calls conditional on none of the initial answers surviving. In the Pantry tuned-temperature row, the displayed values imply approximately `3.38/(1-.5312)=7.21` calls among replay's failed initial portfolios versus `4.81/(1-.3125)=7.00` among the control's; these condition on different failure populations and do not support a uniform conditional-speed claim. The reported unconditional replay reduction is supported. This issue is distinct from correcting the appendix destination to the five-domain recovery study.

20. **Reproducibility-statement description, lines 534–535.** The current `app:reproducibility` contains evaluation-comparison scope, dataset/model links, an experiment index, an analysis-input index, a model-loading guide, and the `paper/FIGURE_MANIFEST.md` pointer. It does not itself print hashes, validation receipts, or regeneration commands. Those may live in linked/local artifacts, but the supplied statement that this appendix “records protocols, hashes, validations, and regeneration commands” overstates the contents of the current section. The appendix link resolves correctly; this is a content-description mismatch, so it was not changed.

This audit is a source-consistency check of the supplied main and its immediate evidence, not a reanalysis of raw experiments or a claim that all scientific limitations in the entire appendix have been resolved.
