"""The brain: persona, history, Ollama client and the tool-call loop.

Owner: workstream A. Everything else codes against :class:`roco.brain.base.Brain`.
"""

from roco.brain.base import Brain, SentenceCallback

__all__ = ["Brain", "SentenceCallback"]
