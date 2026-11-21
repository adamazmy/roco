"""Strip the leading ``[emotion]`` tag from a streamed reply.

The model is asked to start every reply with a tag such as ``[happy]``. In a
stream the tag can arrive split across chunks (``"[ha"``, ``"ppy] Hi"``), in
another case (``"[Happy]"``), or not at all. :class:`EmotionTagParser` holds
back only the few characters it needs to decide, then passes text through.
"""

from __future__ import annotations

from roco.bus import Emotion

# Longest tag we wait for before deciding the "[" was just text.
MAX_TAG_LEN = 24


class EmotionTagParser:
    """Feed chunks; get text back with the leading tag removed.

    After the first non-whitespace text is decided, :attr:`emotion` holds the
    parsed tag (``Emotion.NEUTRAL`` for an unknown tag word) or None when the
    reply had no tag. :attr:`decided` tells whether that decision has been made.
    """

    def __init__(self) -> None:
        self._buf = ""
        self.decided = False
        self.emotion: Emotion | None = None

    def feed(self, chunk: str) -> str:
        if self.decided:
            return chunk
        self._buf += chunk
        stripped = self._buf.lstrip()
        if not stripped:
            return ""
        if not stripped.startswith("["):
            return self._decide(None, stripped)
        close = stripped.find("]")
        if close == -1:
            if len(stripped) > MAX_TAG_LEN or "\n" in stripped:
                return self._decide(None, stripped)  # a "[" that is not a tag
            return ""  # wait for the rest of the tag
        word = stripped[1:close]
        if len(word) > MAX_TAG_LEN or "\n" in word:
            return self._decide(None, stripped)
        return self._decide(Emotion.parse(word), stripped[close + 1 :].lstrip())

    def flush(self) -> str:
        """End of stream: release anything held back (an unfinished "[..." is text)."""
        if self.decided:
            return ""
        return self._decide(None, self._buf.strip())

    def _decide(self, emotion: Emotion | None, text: str) -> str:
        self.decided = True
        self.emotion = emotion
        self._buf = ""
        return text
