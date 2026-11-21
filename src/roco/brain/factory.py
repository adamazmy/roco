"""Stable entry point for building the brain: ``create_brain(config, bus) -> Brain``.

Used by ``roco chat`` and ``roco run``. Builds the standard tool registry (which
also starts the reminder scheduler) unless a registry is passed in.
"""

from roco.brain.ollama_brain import create_brain

__all__ = ["create_brain"]
