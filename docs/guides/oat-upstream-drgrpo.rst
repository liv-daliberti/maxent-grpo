OAT Upstream DR.GRPO
====================

This guide documents the exact upstream OAT/``understand-r1-zero`` stack that
reproduced the public README-style 1.5B DR.GRPO run in this repository, plus
the opt-in listwise MaxEnt explorer overlay built directly on top of it.

Working Baseline
----------------

The proven baseline path is:

- launcher: ``ops/run_oat_zero_exact_1p5b_upstream.sh``
- node302 Slurm wrapper:
  ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm``
- upstream checkout: ``understand-r1-zero/``
- Python env: ``var/seed_paper_eval/paper310``
- flash-attn: installed on the node at launch time into a local overlay

The README-flash runtime that worked is:

- ``python==3.10.20``
- ``torch==2.6.0``
- ``transformers==4.51.3``
- ``vllm==0.8.4``
- ``oat-llm==0.1.3.post1``
- ``deepspeed==0.16.8``
- ``flash-attn==2.7.4.post1`` via the runtime overlay

The baseline training geometry is intentionally left unchanged:

- ``objective=grpo``
- ``critic_type=drgrpo``
- ``prompt_template=r1``
- ``num_samples=8``
- ``train_batch_size=128``
- ``train_batch_size_per_device=1``
- ``rollout_batch_size=128``
- ``rollout_batch_size_per_device=16``
- ``pi_buffer_maxlen_per_device=128``
- ``beta=0.0``

Explorer Overlay
----------------

The listwise explorer path reuses that exact runtime and launcher, but switches
the learner onto the listwise MaxEnt objective through environment variables.

- launcher: ``ops/run_oat_zero_explorer_1p5b_upstream.sh``
- node302 Slurm wrapper:
  ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm``

The explorer defaults mirror the existing Dr.GRPO-Explorer settings already
used elsewhere in this repository:

- ``objective=maxent_listwise``
- ``beta=0.0`` by default; set it above zero only when you explicitly want the
  reference-weight term active
- ``maxent_tau=0.5``
- ``maxent_q_temperature=2.0``
- ``maxent_q_epsilon=1e-6``
- ``maxent_length_normalize_ref=true``
- ``maxent_length_normalize_policy=true``
- ``maxent_listwise_skip_zero_variance_groups=true``
- ``maxent_use_clip_objective=true``
- ``maxent_clip_objective_coef=1.0``
- ``maxent_reference_logprobs_source=model`` so a nonzero ``beta`` can reuse the
  model-reference path without changing the launcher
- ``maxent_logprob_chunk_size=2``
- ``train_batch_size_per_device=8`` so each microbatch contains one whole
  prompt group when ``num_samples=8``

Safety Guardrails
-----------------

The baseline DR.GRPO path stays the default. Nothing switches to listwise
unless the launch explicitly sets ``OAT_ZERO_OBJECTIVE=maxent_listwise``.

Additional guardrails are enforced in two places:

- shell launcher validation in ``ops/run_oat_zero_exact_1p5b_upstream.sh``
- learner validation in ``understand-r1-zero/train_zero_math.py``
- repo audit in ``tools/audit_oat_setup.py``

The listwise path now fails fast when:

- ``objective`` is unsupported
- ``critic_type`` is not ``drgrpo``
- ``num_samples <= 1``
- ``train_batch_size_per_device`` is not divisible by ``num_samples``
- ``maxent_tau <= 0``

Implementation Notes
--------------------

The local listwise extension lives in ``oat_zero_ext/listwise.py`` and is used
only by the patched ``understand-r1-zero/train_zero_math.py`` learner.

Key design choice: the baseline OAT PPO/Dr.GRPO update path is untouched. The
listwise branch activates only inside ``ZeroMathLearner.learning_step`` when
``objective=maxent_listwise`` and keeps whole prompt groups together during
minibatching.

That prompt-group preservation matters because stock OAT shuffles flat rollout
rows, which is correct for GRPO but wrong for a listwise prompt-group loss.

Guarantees And Limits
---------------------

This setup is engineered to avoid breaking the proven baseline:

- one shared runtime/bootstrap path
- separate baseline and explorer wrappers
- separate default run names and local scratch roots
- opt-in objective routing
- focused unit tests for the prompt-group and q/weight math

It is still not possible to promise an absolute guarantee against every future
cluster, dependency, or upstream behavior change. What we do guarantee here is
that the repository now has an isolated baseline path, an isolated explorer
overlay, explicit validation, and targeted tests for the failure modes that
would silently corrupt the objective.
