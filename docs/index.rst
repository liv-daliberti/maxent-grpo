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
frozen reference anchor aligned. The Qwen 0.5B/1.5B ``maxent-grpo`` recipes
default to trainer-level entropy MaxEnt
(``objective=maxent_entropy``) with
``maxent_policy_entropy_mode=exact``; the frozen reference still enters through
the same KL term used by GRPO. The older Qwen 7B ``maxent-grpo`` math recipe
remains a GRPO + entropy-bonus run.

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

   methods
   architecture
   guides/cli
   guides/training
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
