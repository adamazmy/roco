"""The brain: persona, history, Ollama client and the tool-call loop.

Owner: workstream A. Everything else codes against :class:`roco.brain.base.Brain`;
``create_brain(config, bus, registry)`` builds the real one (Ollama).
"""

from roco.brain.base import Brain, SentenceCallback, TurnHooks
from roco.brain.ollama_brain import OllamaBrain, create_brain

__all__ = ["Brain", "OllamaBrain", "SentenceCallback", "TurnHooks", "create_brain"]
