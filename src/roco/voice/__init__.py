"""Voice pipeline: wake word, recording, speech-to-text and text-to-speech.

Owner: workstream B. Heavy imports (sounddevice, openwakeword, faster_whisper,
piper) must stay inside the implementation modules so text mode never needs them.
"""

from roco.voice.base import Listener, Speaker

__all__ = ["Listener", "Speaker"]
