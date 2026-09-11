# E28 post-hoc matched free-form Dr.GRPO controls at 3B

**Status: FROZEN POST-HOC MATCHED-CONTROL EXTENSION, BEFORE E28 SUBMISSION OR
E28 OUTCOMES (2026-07-21).**

E28 adds the unrestricted free-form Dr.GRPO controls that were absent from
E25-v2's 3B graph-coloring and Countdown easy3 cohorts. E25-v2 had already
started before this extension was requested, so every E28-versus-E25-v2
comparison is explicitly post-hoc and exploratory. E28 does not alter,
restart, relabel, or pool any earlier E1, E22, or E25 trajectory.

## Matched contrast

E28 changes exactly the entropy treatment relative to E25-v2. It uses ordinary
free-form Dr.GRPO with `alpha=0`, `xdr_tau=inf`, `beta=0`, and every entropy or
aggregation controller disabled. The policy remains the unrestricted
`qwen_boxed` token policy; no canonical-action codec or learner-side canonical
sampling is enabled.

All non-treatment settings are inherited exactly from E25-v2:

- Qwen2.5-3B-Instruct revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`;
- seeds 43, 44, and 45 in graph coloring and Countdown (six jobs total);
- group size 16, one PPO epoch, learning rate `2e-7`, sampling temperature 1,
  `top_p=1`, and free-form responses of at most 192 tokens;
- graph coloring's 192/96 train/evaluation pools with evaluation and checkpoint
  every 48 prompts;
- Countdown easy3's 384/128 pools with evaluation and checkpoint every 96
  prompts;
- five complete prompt-pool passes;
- one node302 A100, 96 GiB host memory, ZeRO-2, CPU optimizer offload,
  activation offload, vLLM sleep, rollout batch one, global train batch 16,
  and backward microbatch one.

The immutable Python source and operations surfaces are the same snapshots as
E25-v2: `217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448`
and `05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce`.
Fresh prefixes are `gce28_freeform_drgrpo_3b_v1` and
`cde28_freeform_drgrpo_3b_v1`.

## Submission and reporting

Configuration-only validation must pass for both tasks before submission.
The launcher submits all six jobs held, audits method, seeds, model, datasets,
budgets, resources, and the complete absence of entropy treatment, and releases
only a complete valid cohort. A partial or invalid cohort remains held.

All figures and tables must label E28 as the matched free-form Dr.GRPO control
for E25-v2, retain all three paired seeds, and disclose that this control was
added post-hoc after E25-v2 began. Historical E1 3B Dr.GRPO is contextual only:
its group size and/or prompt pool differ and it must not be substituted for
E28.
