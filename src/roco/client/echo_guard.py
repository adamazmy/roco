"""Was that barge-in Adam, or ROCO hearing itself?

On laptop speakers some of ROCO's own voice survives echo cancellation, gets past
the VAD and is transcribed as a (garbled) user turn: ROCO says "it's just gone half
past three on Sunday" and the brain hears "House three", "on the song", "Just go".
After a barge-in the body compares the brain's transcript of the new turn with what
ROCO was saying around the cut (:func:`is_echo`). A match means echo: the turn is
cancelled before the brain answers it.

Matching is deliberately fuzzy and sound-based: words are reduced to a rough
phonetic skeleton (consonant classes, vowels dropped), then the transcript is
aligned against every stretch of the reference of similar length. Words people use
to interrupt ("stop", "wait", the wake word...) are never treated as echo.

Pure Python, no dependencies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

# A barge-in containing any of these is Adam, even if ROCO happens to say them too.
INTERRUPT_WORDS = frozenset({
    "stop", "wait", "hold", "pause", "shut", "quiet", "enough", "cancel", "never",
    "nevermind", "no", "nope", "hey", "jarvis", "roco", "rocco", "rocko", "sorry",
    "actually", "excuse",
    # Questions and corrections are what an interruption sounds like.
    "what", "what's", "whats", "why", "how", "who", "where", "when", "which", "wrong",
    "louder", "quieter", "slower", "faster", "please", "thanks", "thank", "repeat", "again",
    # Commands: losing one is worse than answering a rare echo of ROCO saying it.
    "turn", "play", "tell", "show", "open", "call", "set", "remind", "switch", "start",
    "read", "check", "skip", "mute",
})

_NUMBERS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "15": "fifteen", "20": "twenty", "30": "thirty", "45": "forty five",
}

# Consonants that sound alike map to one class; vowels and h/w/y vanish.
_CLASSES = {
    **dict.fromkeys("bp", "P"), **dict.fromkeys("fv", "F"), **dict.fromkeys("dt", "T"),
    **dict.fromkeys("gkcq", "K"), **dict.fromkeys("szx", "S"), **dict.fromkeys("jg", "K"),
    "l": "L", "r": "R", "m": "M", "n": "N",
}


def words(text: str) -> list[str]:
    out: list[str] = []
    for w in re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")):
        w = w.strip("'")
        if not w:
            continue
        if w.isdigit():
            out.extend(_NUMBERS.get(w, w).split())
        else:
            out.append(w)
    return out


def phonetic(word: str) -> str:
    """Rough sound skeleton: ``sunday`` -> ``SNT``, ``song`` -> ``SNK``."""
    w = word.replace("'", "")
    w = re.sub(r"th", "T", w)
    w = re.sub(r"ph", "f", w)
    w = re.sub(r"ck", "k", w)
    w = re.sub(r"ng\b", "nk", w)  # song ~ sunk
    w = re.sub(r"(?<!^)gh", "", w)  # night, though
    w = re.sub(r"^wr", "r", w)
    w = re.sub(r"^kn", "n", w)
    key = []
    for ch in w:
        cls = _CLASSES.get(ch, ch if ch.isupper() else "")
        if cls and (not key or key[-1] != cls):
            key.append(cls)
    return "".join(key) or (w[:1].upper() if w else "")


# Words too common to count as evidence on their own.
_STOP = """
a an the and or but so of to in on at for with by from up as is it its it's i i'm i've
you you're your we he she they them me my be am are was were do does did have has had
this that what which who how when where there here then than just very really not
oh ok okay yeah yes well uh um about can could would should will
"""
STOP_WORDS = frozenset(_STOP.split())


def word_similarity(a: str, b: str) -> float:
    """Spelling or sound, whichever is closer (0..1)."""
    if a == b:
        return 1.0
    spelled = SequenceMatcher(None, a, b, autojunk=False).ratio()
    pa, pb = phonetic(a), phonetic(b)
    sound = SequenceMatcher(None, pa, pb, autojunk=False).ratio() if pa and pb else 0.0
    if min(len(pa), len(pb)) < 2:
        sound *= 0.8  # one-consonant skeletons collide too easily
    return max(spelled, sound)


@dataclass(frozen=True)
class EchoVerdict:
    echo: bool
    score: float  # 0..1 similarity with the best-matching stretch of ROCO's speech
    matched: str = ""  # that stretch
    reason: str = ""


def echo_score(transcript: str, reference: Iterable[str]) -> tuple[float, str]:
    """Best phonetic similarity (0..1) of ``transcript`` with any stretch of the
    reference sentences of about the same length, and that stretch."""
    t_words = words(transcript)
    if not t_words:
        return 0.0, ""
    t_key = " ".join(phonetic(w) for w in t_words)
    best, best_text = 0.0, ""
    for sentence in reference:
        r_words = words(sentence)
        if not r_words:
            continue
        r_keys = [phonetic(w) for w in r_words]
        n = len(t_words)
        for size in range(max(1, n - 1), n + 3):
            for start in range(0, max(1, len(r_words) - size + 1)):
                window = r_keys[start : start + size]
                ratio = SequenceMatcher(None, t_key, " ".join(window), autojunk=False).ratio()
                if ratio > best:
                    best, best_text = ratio, " ".join(r_words[start : start + size])
    return best, best_text


def is_echo(transcript: str, reference: Iterable[str], threshold: float = 0.6,
            max_words: int = 8) -> EchoVerdict:
    """Echo verdict for a barge-in transcript against what ROCO was saying.

    - empty transcript: not decided here (the caller treats it as a false barge-in);
    - an interrupt word ("stop", "wait", the wake word...): always Adam;
    - longer than ``max_words``: Adam (echo fragments that survive AEC are short);
    - otherwise echo when the phonetic similarity reaches ``threshold``.
    """
    t_words = words(transcript)
    if not t_words:
        return EchoVerdict(False, 0.0, reason="empty")
    if INTERRUPT_WORDS.intersection(t_words):
        return EchoVerdict(False, 0.0, reason="interrupt word")
    if len(t_words) > max_words:
        return EchoVerdict(False, 0.0, reason="long")
    reference = list(reference)
    score, matched = echo_score(transcript, reference)
    if score < threshold:
        return EchoVerdict(False, score, matched, "different words")
    # Word evidence: a phrase can look alike by chance; echo repeats ROCO's words.
    ref_words = [w for sentence in reference for w in words(sentence)]
    content = [w for w in t_words if w not in STOP_WORDS] or t_words
    def best(w: str) -> float:
        return max((word_similarity(w, r) for r in ref_words), default=0.0)

    sims = [best(w) for w in content]
    strong = max(best(w) for w in t_words) >= 0.85  # one word ROCO really said
    covered = sum(s >= 0.6 for s in sims) / len(sims) >= 0.5
    if not (strong and covered):
        return EchoVerdict(False, score, matched, "no word of ROCO's in it")
    return EchoVerdict(True, score, matched, "matches ROCO's speech")
