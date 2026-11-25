"""Translate with a released model, from a local directory or the Hugging Face Hub.

    from src.inference import Translator
    tr = Translator.from_pretrained("nickoo004/nmt-uz-76m")
    tr.translate("Good morning!", target="uz")
"""

from pathlib import Path

import torch

from src import text
from src.data_pipeline.alphabet import to_old_latin
from src.data_pipeline.tokenizer import Tokenizer
from src.evaluation.evaluate import translate as beam_translate
from src.export import load_release
from src.models.transformer import Seq2SeqTransformer

RELEASE_FILES = ["model.safetensors", "config.json", "spm.model"]
MAX_RATIO = 3  # an output longer than 3x its source is a decoding loop, not a translation


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
            texts = [text.normalize_uz(t) for t in texts]  # so Cyrillic splits into sentences too

        # All sentences (and pieces of long ones) of all texts go through the model in one pass.
        segments, layout = text.segment(texts, lambda s: len(self.tok.encode([s])[0]))
        max_len = min(max_len or self.model.cfg.max_len, self.model.cfg.max_len)
        out = beam_translate(self.model, self.tok, segments, target, beam=beam,
                             batch_size=batch_size, max_len=max_len, autocast=autocast,
                             max_ratio=MAX_RATIO) if segments else []
        if target == "uz" and script == "old":
            out = [to_old_latin(s) for s in out]

        joined = text.assemble(out, layout)
        return joined[0] if single else joined
