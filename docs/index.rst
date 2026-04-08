MaxEnt-GRPO
===========

This documentation now treats the upstream OAT stack as the canonical training
surface for the repository. The active path is the README-flash
``understand-r1-zero`` baseline plus the local listwise maxent-explorer
overlay on top of it.

Older TRL/Hydra orchestration and pre-canonical launchers are still retained
under ``archive/trl/``, but they are no longer presented as the default way to
train in this repo.

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
   guides/oat-upstream-drgrpo
   guides/training
   guides/runtime
   guides/evaluation
   guides/cli
   recipes

Reference
---------

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Project Notes
-------------

- Active launchers live in ``ops/`` and are OAT-only.
- Retired launchers live in ``archive/trl/``.
- The runtime audit entrypoint is ``tools/audit_oat_setup.py``.
