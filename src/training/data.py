"""Pre-tokenized pair storage and token-budget batching for bidirectional training."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data_pipeline.tokenizer import BOS, EOS, PAD

# Direction of an example: 0 = en->uz, 1 = uz->en.
EN2UZ, UZ2EN = 0, 1


@dataclass
class PairStore:
    """Flat int32 token arrays + offsets, so millions of pairs fit in RAM cheaply."""

    en: np.ndarray
    en_off: np.ndarray
    uz: np.ndarray
    uz_off: np.ndarray
    # -1: real pair (both directions); EN2UZ/UZ2EN: back-translated, only that direction.
    only_dir: np.ndarray
    weight: np.ndarray

    def __len__(self) -> int:
        return len(self.en_off) - 1

    @staticmethod
    def _flatten(seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        off = np.zeros(len(seqs) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        parts = [np.asarray(s, np.int32) for s in seqs]
        flat = np.concatenate(parts) if parts else np.zeros(0, np.int32)
        return flat, off

    @classmethod
    def build(cls, en_ids, uz_ids, only_dir=None, weight=None) -> "PairStore":
        en, en_off = cls._flatten(en_ids)
        uz, uz_off = cls._flatten(uz_ids)
        n = len(en_ids)
        only_dir = np.full(n, -1, np.int8) if only_dir is None else np.asarray(only_dir, np.int8)
        weight = np.ones(n, np.float32) if weight is None else np.asarray(weight, np.float32)
        return cls(en, en_off, uz, uz_off, only_dir, weight)

    @classmethod
    def concat(cls, stores: list["PairStore"]) -> "PairStore":
        def join(flat, off):
            offs, shift = [np.zeros(1, np.int64)], 0
            for st in stores:
                offs.append(getattr(st, off)[1:] + shift)
                shift += int(getattr(st, off)[-1])
            return np.concatenate([getattr(st, flat) for st in stores]), np.concatenate(offs)

        en, en_off = join("en", "en_off")
        uz, uz_off = join("uz", "uz_off")
        return cls(en, en_off, uz, uz_off,
                   np.concatenate([st.only_dir for st in stores]),
                   np.concatenate([st.weight for st in stores]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, arr in self.__dict__.items():
            np.save(path / f"{name}.npy", arr)

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> "PairStore":
        # Token arrays are memory-mapped: the full corpus is several GB.
        mode = "r" if mmap else None
        return cls(**{name: np.load(Path(path) / f"{name}.npy", mmap_mode=mode)
                      for name in cls.__dataclass_fields__})

    def side(self, i: int, lang: str) -> np.ndarray:
        flat, off = (self.en, self.en_off) if lang == "en" else (self.uz, self.uz_off)
        return flat[off[i] : off[i + 1]]

    def lengths(self, lang: str) -> np.ndarray:
        off = self.en_off if lang == "en" else self.uz_off
        return np.diff(off)


class TokenBatcher:
    """Yields lists of example ids (pair_idx * 2 + direction) under a token budget.

    Examples are length-sorted inside large shuffled pools, which keeps padding
    low while batches still look random across an epoch.
    """

    def __init__(self, store: PairStore, max_tokens: int, max_len: int,
                 seed: int = 0, pool_batches: int = 100):
        self.max_tokens, self.max_len = max_tokens, max_len
        self.pool_batches = pool_batches
        self.rng = np.random.default_rng(seed)

        ids = np.arange(len(store) * 2)
        pair, direction = ids // 2, ids % 2
        keep = (store.only_dir[pair] == -1) | (store.only_dir[pair] == direction)
        self.ids = ids[keep]
        # +3 covers the language tag, optional <bt> and EOS.
        lens = np.maximum(store.lengths("en"), store.lengths("uz")) + 3
        self.lens = np.minimum(lens[self.ids // 2], max_len)

    def __iter__(self):
        order = self.rng.permutation(len(self.ids))
        avg = max(int(self.lens.mean()), 1)
        pool = self.pool_batches * max(self.max_tokens // avg, 1)
        for start in range(0, len(order), pool):
            chunk = order[start : start + pool]
            chunk = chunk[np.argsort(self.lens[chunk], kind="stable")]
            batches, cur, cur_max = [], [], 0
            for j in chunk:
                n = int(self.lens[j])
                if cur and max(cur_max, n) * (len(cur) + 1) > self.max_tokens:
                    batches.append(cur)
                    cur, cur_max = [], 0
                cur.append(int(self.ids[j]))
                cur_max = max(cur_max, n)
            if cur:
                batches.append(cur)
            for k in self.rng.permutation(len(batches)):
                yield batches[k]


def collate(store: PairStore, example_ids: list[int], tag_ids: dict[str, int],
            bt_id: int, max_len: int) -> dict[str, torch.Tensor]:
    srcs, tgts, weights = [], [], []
    for ex in example_ids:
        pair, direction = divmod(ex, 2)
        src_lang, tgt_lang = ("en", "uz") if direction == EN2UZ else ("uz", "en")
        prefix = [tag_ids[tgt_lang]] + ([bt_id] if store.only_dir[pair] != -1 else [])
        src = prefix + store.side(pair, src_lang)[: max_len - len(prefix) - 1].tolist() + [EOS]
        tgt = store.side(pair, tgt_lang)[: max_len - 1].tolist()
        srcs.append(src)
        tgts.append(tgt)
        weights.append(float(store.weight[pair]))

    b = len(srcs)
    s_len = max(map(len, srcs))
    t_len = max(map(len, tgts)) + 1
    src = torch.full((b, s_len), PAD, dtype=torch.long)
    tgt_in = torch.full((b, t_len), PAD, dtype=torch.long)
    tgt_out = torch.full((b, t_len), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : len(s)] = torch.tensor(s)
        tgt_in[i, : len(t) + 1] = torch.tensor([BOS] + t)
        tgt_out[i, : len(t) + 1] = torch.tensor(t + [EOS])
    return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out,
            "weight": torch.tensor(weights)}
