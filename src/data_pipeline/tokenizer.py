"""Shared en+uz SentencePiece tokenizer with direction tags."""

from collections.abc import Iterable
from pathlib import Path

import sentencepiece as spm

PAD, BOS, EOS, UNK = 0, 1, 2, 3
LANG_TAGS = {"uz": "<2uz>", "en": "<2en>"}
BT_TAG = "<bt>"


def train_tokenizer(
    texts: Iterable[str], model_prefix: str | Path, vocab_size: int = 16384,
    max_sentences: int = 4_000_000,
) -> Path:
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=True,
        required_chars="ğöşçĞÖŞÇʼ",
        input_sentence_size=max_sentences,
        shuffle_input_sentence=True,
        pad_id=PAD, bos_id=BOS, eos_id=EOS, unk_id=UNK,
        user_defined_symbols=[*LANG_TAGS.values(), BT_TAG],
        normalization_rule_name="identity",  # text is already normalized by the cleaner
        num_threads=8,
        train_extremely_large_corpus=False,
    )
    return Path(f"{model_prefix}.model")


class Tokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.tag_ids = {lang: self.sp.piece_to_id(t) for lang, t in LANG_TAGS.items()}
        self.bt_id = self.sp.piece_to_id(BT_TAG)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, texts: list[str]) -> list[list[int]]:
        return self.sp.encode(texts)

    def source_ids(self, ids: list[int], tgt_lang: str, bt: bool = False) -> list[int]:
        prefix = [self.tag_ids[tgt_lang]] + ([self.bt_id] if bt else [])
        return prefix + list(ids) + [EOS]

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in (PAD, BOS)]
        if EOS in ids:
            ids = ids[: ids.index(EOS)]
        return self.sp.decode(ids)
