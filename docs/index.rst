MaxEnt-GRPO
===========

MaxEnt-GRPO is a clean GRPO training stack with optional maximum-entropy
weighting and a GRPO + entropy-bonus mode. It targets practical math training
and evaluation while keeping the implementation approachable for production
use. See the README_ on GitHub for the full project overview and quick start
instructions.

.. _README: https://github.com/huggingface/open-r1#readme

Paired GRPO/MaxEnt recipes live under ``configs/recipes/<model>/grpo/`` and
``configs/recipes/<model>/maxent-grpo/`` so runs stay comparable. The GRPO
pairs pin ``maxent_reference_logprobs_source: model`` to keep the objective and
frozen reference anchor aligned. The
``maxent-grpo`` recipes now default to GRPO + entropy bonus (and can still
switch to MaxEnt weighting via ``train_grpo_objective=false``).

Get Started
-----------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started

Guides
------

.. toctree::
   :maxdepth: 2
   :caption: Guides

   architecture
   guides/cli
   guides/training
   guides/generation
   guides/evaluation
   guides/runtime
   recipes

Reference
---------

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Project Links
-------------

- Source code: https://github.com/huggingface/open-r1
- README: https://github.com/huggingface/open-r1#readme
