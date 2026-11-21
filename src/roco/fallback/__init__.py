"""Fallback brain for the body (workstream D): PC brain -> Claude -> canned line.

Light to import: neither the ``anthropic`` SDK nor faster-whisper is loaded until the
fallback answers its first turn. See :mod:`roco.fallback.chain` for the contract.
"""

from roco.fallback.chain import (
    BRAIN_RESTORED_LINE,
    CLOUD_SWITCH_LINE,
    OFFLINE_LINE,
    FallbackChain,
    create_fallback,
)

__all__ = [
    "BRAIN_RESTORED_LINE",
    "CLOUD_SWITCH_LINE",
    "OFFLINE_LINE",
    "FallbackChain",
    "create_fallback",
]
