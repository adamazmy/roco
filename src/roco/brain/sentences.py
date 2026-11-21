"""Incremental sentence splitting for streamed LLM output.

Text-to-speech wants whole sentences as early as possible, so tokens are fed in
as they arrive and complete sentences come out as soon as we can be sure they
are complete. A boundary is sentence punctuation (``. ! ?`` or an ellipsis),
optionally followed by closing quotes/brackets, then whitespace, then a
character that does not look like the sentence continues:

- ``3.5`` / ``10.30``: no whitespace after the dot, never a boundary.
- ``Mr. Smith`` / ``e.g. Paris``: the word before the dot is a known abbreviation.
- ``J. Smith``: a single capital letter before the dot is an initial.
- ``Well... maybe``: the next word starts lowercase, so the sentence continues.

Because the decision needs the first character after the whitespace, the last
sentence of a stream is only released by :meth:`SentenceSplitter.flush`.
A line break also ends a sentence (models occasionally write lists anyway).

Long sentences are released clause by clause: once the pending text is at least
``clause_min_chars`` long, the first ``, `` or ``; `` after that point ends a chunk.
A spoken list ("you have the dentist at ten, lunch with Sam at one, and...") would
otherwise keep the voice silent until the whole list has been generated. Commas
inside numbers ("10,000") are never followed by a space, so they never split.
"""

from __future__ import annotations

import re

from roco.bus import Emotion

# Words (lowercased, without their final dot) that do not end a sentence.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "rev", "hon", "capt", "sgt",
        "lt", "col", "gen", "mt", "ft", "vs", "approx", "dept", "est", "fig", "cf",
        "e.g", "i.e", "eg", "ie", "a.k.a", "u.k", "u.s", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
)

_TERMINAL = ".!?…"
_CLAUSE_BREAKS = ",;"
CLAUSE_MIN_CHARS = 60
_CLOSERS = "\"'”’)]}»"
# Any known emotion tag anywhere in a sentence, e.g. a model repeating "[happy]" mid-reply.
_INLINE_TAG = re.compile(
    r"\s*\[\s*(?:" + "|".join(e.value for e in Emotion) + r")\s*\]\s*", re.IGNORECASE
)
# Short bracketed stage directions the model sometimes adds: "[chuckling]", "[laughs softly]".
# Also catches a tool name the model wrote in brackets ("[get_current_time]") and an
# email id copied from a tool result ("[id o1]").
_STAGE_DIRECTION = re.compile(r"\s*\[\s*[A-Za-z][A-Za-z0-9_ '\-]{0,30}\]\s*")
# A list bullet the model wrote anyway: at the start, or after a colon or full stop
# ("Here they are: - From Kiwi.com ... - From UNiDAYS ..."). A hyphen between words or
# numbers ("ten - fifteen") is left alone.
_BULLET = re.compile(r"(^|[:.!?]\s*)\s*(?:[-*•]|\d{1,2}[.)])\s+")
_SPACES = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")


def clean_sentence(text: str) -> str:
    """Remove emotion tags, stage directions and markdown asterisks; squeeze whitespace.

    Square brackets are never meant to be read aloud, so a short bracketed
    word or phrase is dropped wherever it appears.
    """
    text = _INLINE_TAG.sub(" ", text)
    text = _STAGE_DIRECTION.sub(" ", text)
    text = _BULLET.sub(r"\1 ", text.strip())
    text = text.replace("*", "").replace(" | ", ", ").replace("|", " ")
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", text))
    return text.strip()


class SentenceSplitter:
    """Feed chunks with :meth:`feed`; get complete sentences back. Call :meth:`flush` at the end."""

    def __init__(self, clause_min_chars: int | None = CLAUSE_MIN_CHARS) -> None:
        """``clause_min_chars=None`` turns clause splitting off (whole sentences only)."""
        self._buf = ""
        self.clause_min_chars = clause_min_chars

    def feed(self, chunk: str) -> list[str]:
        if not chunk:
            return []
        self._buf += chunk
        out: list[str] = []
        while True:
            cut = self._find_boundary()
            if cut is None:
                cut = self._find_clause()
            if cut is None:
                break
            sentence = clean_sentence(self._buf[:cut])
            self._buf = self._buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        """Return whatever is left (the final sentence), or None if nothing is."""
        rest = clean_sentence(self._buf)
        self._buf = ""
        return rest or None

    # ------------------------------------------------------------------ internals
    def _find_clause(self) -> int | None:
        """Index just after a clause break in a long unfinished sentence, or None."""
        if self.clause_min_chars is None:
            return None
        buf = self._buf
        for i in range(max(self.clause_min_chars - 1, 0), len(buf) - 1):
            if buf[i] in _CLAUSE_BREAKS and buf[i + 1] in " 	":
                return i + 1
        return None

    def _find_boundary(self) -> int | None:
        """Index just after the first complete sentence in the buffer, or None."""
        buf = self._buf
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch == "\n":
                if buf[:i].strip():
                    return i + 1
                i += 1
                continue
            if ch not in _TERMINAL:
                i += 1
                continue

            start = i
            while i < n and buf[i] in _TERMINAL:
                i += 1
            while i < n and buf[i] in _CLOSERS:
                i += 1
            end = i  # sentence ends here (exclusive)
            if i >= n:
                return None  # need to see what follows
            if not buf[i].isspace():
                continue  # "3.5", "e.g.x", "10.30": not a boundary
            while i < n and buf[i] in " \t":
                i += 1
            if i >= n:
                return None  # whitespace so far; wait for the next word
            if buf[i] == "\n":
                return end
            if self._continues(buf, start, buf[i]):
                continue
            return end
        return None

    @staticmethod
    def _continues(buf: str, punct_at: int, next_char: str) -> bool:
        """True when the punctuation at ``punct_at`` does not end the sentence."""
        if next_char.islower():
            return True  # "e.g. the", "Well... maybe", "3 p.m. tomorrow"
        if buf[punct_at] != "." or buf[punct_at : punct_at + 2] == "..":
            return False
        # The word right before the dot, e.g. "Mr" or "e.g".
        j = punct_at
        while j > 0 and (buf[j - 1].isalpha() or buf[j - 1] == "."):
            j -= 1
        word = buf[j:punct_at]
        if not word:
            return False
        if word.lower() in ABBREVIATIONS:
            return True
        return len(word) == 1 and word.isupper()  # an initial: "J. Smith"
