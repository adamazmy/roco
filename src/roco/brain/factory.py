"""Stable entry point for building the brain: ``create_brain(config, bus) -> Brain``.

Used by ``roco brain``, ``roco chat`` and ``roco run``. Returns a
:class:`~roco.brain.router.BrainRouter`: Claude by default, local Ollama as the fallback
(docs/ROCO_SPEC.md §9). Builds the standard tool registry (which also starts the
reminder scheduler) unless a registry is passed in.
"""

from roco.brain.router import create_brain

__all__ = ["create_brain"]
