"""Normalization, filtering and de-duplication of en-uz sentence pairs.

Filtering is per row and runs in worker processes. De-duplication works on
64-bit hashes in polars, so tens of millions of pairs fit in a few GB of RAM.
"""

import hashlib
import re
from collections.abc import Iterable

import polars as pl

from src.text import normalize_en, normalize_uz

_NON_WORD = re.compile(r"[\W_]+")
_MARKUP = re.compile(r"<[^>]{1,40}>|&[a-z]+;|&#\d+;")

def near_key(text: str) -> str:
    return _NON_WORD.sub("", text.lower())


def hash64(*parts: str) -> int:
    digest = hashlib.blake2b("\t".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def reject_reason(
    en: str, uz: str, max_words: int = 150, max_ratio: float = 3.0
) -> str | None:
    if not en or not uz:
        return "empty"
    if _MARKUP.search(en) or _MARKUP.search(uz):
        return "markup"
    if not any(c.isalpha() for c in en) or not any(c.isalpha() for c in uz):
        return "non_text"
    if near_key(en) == near_key(uz):
        return "copy"
    if len(en.split()) > max_words or len(uz.split()) > max_words:
        return "too_long"
    short, long_ = sorted((len(en), len(uz)))
    # Very short pairs ("OK" / "Mayli") naturally have wild length ratios.
    if long_ >= 15 and long_ / max(short, 1) > max_ratio:
        return "ratio"
    return None


def clean_pair(en: str, uz: str) -> tuple[str, str, int, int, int] | str:
    """Returns (en, uz, pair_key, en_key, uz_key) or the rejection reason."""
    en, uz = normalize_en(en), normalize_uz(uz)
    reason = reject_reason(en, uz)
    if reason:
        return reason
    ken, kuz = near_key(en), near_key(uz)
    return en, uz, hash64(ken, kuz), hash64("en", ken), hash64("uz", kuz)


def blocklist_keys(sentences: Iterable[str]) -> set[int]:
    """Keys of eval sentences, in both the en and uz key spaces."""
    keys = set()
    for s in sentences:
        for lang, norm in (("en", normalize_en), ("uz", normalize_uz)):
            keys.add(hash64(lang, near_key(norm(s))))
    return keys


def dedup_reasons(keys: pl.DataFrame, blocked: set[int], max_per_side: int = 3) -> pl.Series:
    """Rejection reason per row (null = keep) for a frame with k_pair, k_en, k_uz.

    Row order is priority order: the first occurrence of a duplicate wins, so
    cleaner sources should come first.
    """
    blocked_s = pl.Series(list(blocked), dtype=pl.Int64).implode()
    df = keys.with_row_index("rid").with_columns(
        leak=pl.col("k_en").is_in(blocked_s) | pl.col("k_uz").is_in(blocked_s)
    )
    live = df.filter(~pl.col("leak"))
    first = live.filter(pl.col("k_pair").is_first_distinct())
    capped = first.filter(
        (pl.col("rid").cum_count().over("k_en") <= max_per_side)
        & (pl.col("rid").cum_count().over("k_uz") <= max_per_side)
    )
    reason = (
        df.select("rid", "leak")
        .join(first.select("rid", is_first=pl.lit(True)), on="rid", how="left")
        .join(capped.select("rid", kept=pl.lit(True)), on="rid", how="left")
        .sort("rid")
        .select(
            pl.when(pl.col("leak")).then(pl.lit("leakage"))
            .when(pl.col("is_first").is_null()).then(pl.lit("duplicate"))
            .when(pl.col("kept").is_null()).then(pl.lit("over_cap"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("reason")
        )
    )
    return reason["reason"]
