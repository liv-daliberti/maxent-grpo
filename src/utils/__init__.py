"""Legacy compatibility namespace for the old ``utils.*`` modules.

The project now splits concerns across dedicated packages:

* ``core`` for data/model/evaluation/hub helpers.
* ``patches`` for TRL/vLLM compatibility shims.
* ``telemetry`` for logging integrations (e.g., Weights & Biases).

Importers should switch to those modules directly; ``utils`` no longer exposes
submodules beyond this placeholder.
"""

__all__: list[str] = []
