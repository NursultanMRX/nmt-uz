"""Text handling around the model, shared by the PyTorch and ONNX translators.

Only the standard library and the alphabet converter: the ONNX runtime must not need torch
or polars.
"""

import re
import unicodedata
from collections.abc import Callable

from src.data_pipeline.alphabet import to_new_latin

MAX_SOURCE_TOKENS = 96  # longer sentences are translated in pieces; at 128+ the model drops clauses

_WS = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[\"“«(]?[A-ZĞÖŞÇ0-9])")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
# Pictographs plus the joiners and modifiers that glue them together. The tokenizer has
# never seen them; left in, they can send the decoder into a loop.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿‍︎️⃣]+")

# One text -> lines -> sentences; a sentence is (number of pieces sent to the model, emoji).
Layout = list[list[list[tuple[int, str]]]]


def normalize_en(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def normalize_uz(text: str) -> str:
    return _WS.sub(" ", to_new_latin(text)).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for line in text.splitlines() for s in _SENT_END.split(line) if s.strip()]


def split_emoji(text: str) -> tuple[str, str]:
    """Text without emoji, and the emoji in their original order."""
    found = "".join(_EMOJI.findall(text))
    return (" ".join(_EMOJI.sub(" ", text).split()), found) if found else (text, "")


def split_long(sentence: str, count: Callable[[str], int], limit: int) -> list[str]:
    """Cut a sentence over `limit` tokens at , ; : and, where that is not enough, between words."""
    if count(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    for part in _CLAUSE_END.split(sentence):
        if count(part) > limit:
            pieces += _split_words(part, count, limit)
        elif pieces and count(f"{pieces[-1]} {part}") <= limit:
            pieces[-1] = f"{pieces[-1]} {part}"
        else:
            pieces.append(part)
    return pieces


def _split_words(text: str, count: Callable[[str], int], limit: int) -> list[str]:
    pieces: list[str] = []
    for word in text.split():
        if pieces and count(f"{pieces[-1]} {word}") <= limit:
            pieces[-1] = f"{pieces[-1]} {word}"
        else:
            pieces.append(word)
    return pieces


def segment(texts: list[str], count: Callable[[str], int],
            limit: int | None = None) -> tuple[list[str], Layout]:
    """Everything the model has to translate, in one flat list, and how to put it back."""
    limit = limit or MAX_SOURCE_TOKENS
    segments, layout = [], []
    for text in texts:
        lines = []
        for line in text.splitlines():
            sentences = []
            for sentence in split_sentences(line):
                core, emoji = split_emoji(sentence)
                pieces = split_long(core, count, limit) if core else []
                sentences.append((len(pieces), emoji))
                segments += pieces
            lines.append(sentences)
        layout.append(lines)
    return segments, layout


def assemble(translated: list[str], layout: Layout) -> list[str]:
    it = iter(translated)

    def sentence(n: int, emoji: str) -> str:
        return " ".join([next(it) for _ in range(n)] + ([emoji] if emoji else []))

    return ["\n".join(" ".join(sentence(*s) for s in line) for line in lines).strip()
            for lines in layout]
