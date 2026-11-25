"""Translate with a released model, from a local directory or the Hugging Face Hub.

    from src.inference import Translator
    tr = Translator.from_pretrained("nickoo004/nmt-uz-76m")
    tr.translate("Good morning!", target="uz")
"""

import re
from collections.abc import Callable
from pathlib import Path

import torch

from src.data_pipeline.alphabet import to_old_latin
from src.data_pipeline.cleaning import normalize_uz
from src.data_pipeline.mono import split_sentences
from src.data_pipeline.tokenizer import Tokenizer
from src.evaluation.evaluate import translate as beam_translate
from src.export import load_release
from src.models.transformer import Seq2SeqTransformer

RELEASE_FILES = ["model.safetensors", "config.json", "spm.model"]
MAX_SOURCE_TOKENS = 96  # longer sentences are translated in pieces; at 128+ the model drops clauses
MAX_RATIO = 3  # an output longer than 3x its source is a decoding loop, not a translation

# Pictographs plus the joiners and modifiers that glue them together. The tokenizer has
# never seen them; left in, they can send the decoder into a loop.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿‍︎️⃣]+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


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


class Translator:
    def __init__(self, model: Seq2SeqTransformer, tok: Tokenizer, name: str = ""):
        self.model, self.tok, self.name = model, tok, name
        self.device = next(model.parameters()).device.type

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, device: str = "auto") -> "Translator":
        path = Path(name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download  # reads HF_TOKEN for private repos

            path = Path(snapshot_download(str(name_or_path), allow_patterns=RELEASE_FILES))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_release(path, device)
        return cls(model, Tokenizer(path / "spm.model"), name=str(name_or_path))

    def translate(self, texts: str | list[str], target: str, beam: int = 4,
                  max_len: int | None = None, batch_size: int = 32,
                  script: str = "new", autocast: bool = True) -> str | list[str]:
        if target not in ("uz", "en"):
            raise ValueError(f"target must be 'uz' or 'en', got {target!r}")
        if script not in ("new", "old"):
            raise ValueError(f"script must be 'new' or 'old', got {script!r}")
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if target == "en":
            texts = [normalize_uz(t) for t in texts]  # so Cyrillic splits into sentences too

        # Each text becomes lines of sentences, each sentence one or more pieces plus the emoji
        # taken out of it; all pieces go through the model in one pass.
        def count(s: str) -> int:
            return len(self.tok.encode([s])[0])

        segments, layout = [], []
        for text in texts:
            lines = []
            for line in text.splitlines():
                sentences = []
                for sentence in split_sentences(line):
                    core, emoji = split_emoji(sentence)
                    pieces = split_long(core, count, MAX_SOURCE_TOKENS) if core else []
                    sentences.append((len(pieces), emoji))
                    segments += pieces
                lines.append(sentences)
            layout.append(lines)

        max_len = min(max_len or self.model.cfg.max_len, self.model.cfg.max_len)
        out = beam_translate(self.model, self.tok, segments, target, beam=beam,
                             batch_size=batch_size, max_len=max_len, autocast=autocast,
                             max_ratio=MAX_RATIO) if segments else []
        if target == "uz" and script == "old":
            out = [to_old_latin(s) for s in out]

        it = iter(out)

        def sentence(n: int, emoji: str) -> str:
            return " ".join([next(it) for _ in range(n)] + ([emoji] if emoji else []))

        joined = ["\n".join(" ".join(sentence(*s) for s in line) for line in lines).strip()
                  for lines in layout]
        return joined[0] if single else joined
