"""Builds the training corpus end to end.

    raw sources -> cleaned parquet shards -> cross-source dedup -> train parquet
    -> shared SentencePiece model -> pre-tokenized PairStores (all + clean subset)

Run locally on a few sources for a smoke test, or on Modal for the full ~33M pairs:

    python -m src.data_pipeline.build --out data/processed --only ted2020,qed --limit 5000
"""

import argparse
import itertools
import json
import multiprocessing as mp
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import polars as pl

from src.data_pipeline.cleaning import blocklist_keys, clean_pair, dedup_reasons, normalize_uz
from src.data_pipeline.sources import SOURCES, Source, iter_pairs, load_flores
from src.data_pipeline.tokenizer import Tokenizer, train_tokenizer
from src.training.data import PairStore

CHUNK = 200_000
MAX_TOKENS = 250  # pairs longer than this would get their target truncated in training


def _clean_chunk(rows: list[tuple[str, str]]) -> list:
    return [clean_pair(en, uz) for en, uz in rows]


def _chunks(it: Iterable, size: int) -> Iterator[list]:
    it = iter(it)
    while chunk := list(itertools.islice(it, size)):
        yield chunk


def write_eval(out: Path, splits=("dev", "devtest")) -> list[str]:
    """FLORES+ in the new alphabet as JSONL. Returns every sentence for the blocklist."""
    out.mkdir(parents=True, exist_ok=True)
    seen = []
    for split in splits:
        rows = load_flores(split)
        with open(out / f"flores_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "en": r["en"], "uz": normalize_uz(r["uz"])},
                                   ensure_ascii=False) + "\n")
                seen += [r["en"], r["uz"]]
    return seen


def write_shards(src: Source, pairs: Iterable[tuple[str, str]], out: Path,
                 pool, limit: int | None = None) -> Counter:
    """Cleans one source into numbered parquet files. Returns raw/kept/reject counts."""
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    pairs = itertools.islice(pairs, limit)
    for part, chunk in enumerate(_chunks(pairs, CHUNK)):
        sub = [chunk[i : i + 5000] for i in range(0, len(chunk), 5000)]
        results = [r for rs in pool.map(_clean_chunk, sub) for r in rs]
        kept = [r for r in results if not isinstance(r, str)]
        counts["raw"] += len(results)
        counts.update(r for r in results if isinstance(r, str))
        counts["kept"] += len(kept)
        if not kept:
            continue
        pl.DataFrame(kept, schema=["en", "uz", "k_pair", "k_en", "k_uz"], orient="row").with_columns(
            source=pl.lit(src.name), weight=pl.lit(src.weight, pl.Float32),
            clean=pl.lit(src.clean),
        ).write_parquet(out / f"{part:05d}.parquet")
    return counts


def _shard_files(shard_root: Path, sources: list[Source]) -> list[Path]:
    files = []
    for i, src in enumerate(sources):
        files += sorted((shard_root / f"{i:02d}_{src.name}").glob("*.parquet"))
    return files


def dedup(files: list[Path], blocked: set[int], out: Path, max_per_side: int = 3) -> dict:
    """Drops duplicates (first file wins), over-cap repeats and eval leakage."""
    out.mkdir(parents=True, exist_ok=True)
    keys = pl.scan_parquet(files).select("source", "k_pair", "k_en", "k_uz").collect()
    reasons = dedup_reasons(keys.drop("source"), blocked, max_per_side)
    report = (
        keys.select("source").with_columns(reason=reasons.fill_null("kept"))
        .group_by("source", "reason").len()
    )
    keep = reasons.is_null().to_numpy()
    offset = 0
    for i, path in enumerate(files):
        df = pl.read_parquet(path, columns=["en", "uz", "source", "weight", "clean"])
        mask = keep[offset : offset + len(df)]
        offset += len(df)
        df.filter(pl.Series(mask)).write_parquet(out / f"{i:05d}.parquet")
    stats: dict = {}
    for row in report.iter_rows(named=True):
        stats.setdefault(row["source"], {})[row["reason"]] = row["len"]
    return stats


def tokenizer_sample(files: list[Path], per_lang: int, seed: int = 0) -> list[str]:
    """Equal numbers of en and uz sentences, so neither language dominates the vocab."""
    total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    frac = min(1.0, per_lang / max(total, 1))
    en, uz = [], []
    for i, f in enumerate(files):
        df = pl.read_parquet(f, columns=["en", "uz"]).sample(fraction=frac, seed=seed + i)
        en += df["en"].to_list()
        uz += df["uz"].to_list()
    return en + uz


def encode_stores(files: list[Path], tok, out: Path) -> dict:
    """Tokenizes the train parquet into an all-data store and a clean-only store."""
    stores, clean_stores = [], []
    dropped = 0
    for f in files:
        df = pl.read_parquet(f, columns=["en", "uz", "weight", "clean"])
        if df.is_empty():
            continue
        en_ids, uz_ids = tok.encode(df["en"].to_list()), tok.encode(df["uz"].to_list())
        ok = np.array([len(a) <= MAX_TOKENS and len(b) <= MAX_TOKENS
                       for a, b in zip(en_ids, uz_ids)])
        dropped += int((~ok).sum())
        clean = df["clean"].to_numpy() & ok
        weight = df["weight"].to_numpy()
        for mask, bucket in ((ok, stores), (clean, clean_stores)):
            idx = np.flatnonzero(mask)
            if len(idx):
                bucket.append(PairStore.build([en_ids[i] for i in idx], [uz_ids[i] for i in idx],
                                              weight=weight[idx]))
    stats = {"too_many_tokens": dropped}
    for name, bucket in (("train", stores), ("clean", clean_stores)):
        store = PairStore.concat(bucket)
        store.save(out / name)
        stats[name] = {"pairs": len(store), "en_tokens": int(store.en_off[-1]),
                       "uz_tokens": int(store.uz_off[-1])}
    return stats


def build(out: Path, cache: Path, sources: list[Source], workers: int, vocab_size: int,
          tok_sentences: int, limit: int | None = None, skip_eval: bool = False) -> dict:
    stats: dict = {"sources": {}}
    blocked = set() if skip_eval else blocklist_keys(write_eval(out / "eval"))
    stats["blocklist_keys"] = len(blocked)

    shard_root = out / "shards"
    with mp.Pool(workers) as pool:
        for i, src in enumerate(sources):
            print(f"[clean] {src.name}", flush=True)
            counts = write_shards(src, iter_pairs(src, cache), shard_root / f"{i:02d}_{src.name}",
                                  pool, limit)
            stats["sources"][src.name] = dict(counts)

    print("[dedup]", flush=True)
    for name, d in dedup(_shard_files(shard_root, sources), blocked, out / "train").items():
        stats["sources"].setdefault(name, {})["dedup"] = d
    train_files = sorted((out / "train").glob("*.parquet"))

    print("[tokenizer]", flush=True)
    model = train_tokenizer(tokenizer_sample(train_files, tok_sentences // 2),
                            out / "spm" / "spm", vocab_size=vocab_size,
                            max_sentences=tok_sentences)

    print("[encode]", flush=True)
    stats["stores"] = encode_stores(train_files, Tokenizer(model), out / "stores")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", help="comma-separated source names")
    ap.add_argument("--limit", type=int, help="max raw pairs per source (smoke tests)")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--tok-sentences", type=int, default=4_000_000)
    ap.add_argument("--skip-eval", action="store_true", help="no FLORES+ (no HF_TOKEN)")
    args = ap.parse_args(argv)

    sources = SOURCES
    if args.only:
        names = args.only.split(",")
        sources = [s for s in SOURCES if s.name in names]
    stats = build(args.out, args.cache, sources, args.workers, args.vocab_size,
                  args.tok_sentences, args.limit, args.skip_eval)
    print(json.dumps(stats["stores"], indent=2))


if __name__ == "__main__":
    main()
