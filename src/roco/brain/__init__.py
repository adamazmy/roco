"""The brain: persona, history, the Claude and Ollama backends and the router between them.

Owner: workstream A. Everything else codes against :class:`roco.brain.base.Brain`;
``create_brain(config, bus, registry)`` builds the real one: a
:class:`~roco.brain.router.BrainRouter` (Claude by default, local Ollama as the fallback).
"""

from roco.brain.base import Brain, SentenceCallback, TurnHooks
from roco.brain.ollama_brain import OllamaBrain
from roco.brain.router import BrainRouter, create_brain

__all__ = ["Brain", "BrainRouter", "OllamaBrain", "SentenceCallback", "TurnHooks", "create_brain"]
