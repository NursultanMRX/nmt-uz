"""Monolingual sentences for back-translation, cut from web documents.

    python -m src.data_pipeline.mono --lang uz --out data/processed/mono/uz.txt --max 3000000
"""

import argparse
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from src.data_pipeline.cleaning import hash64, near_key, normalize_en, normalize_uz

MONO = {
    "uz": ("HuggingFaceFW/fineweb-2", "uzn_Latn"),
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
}

_SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[\"“«(]?[A-ZĞÖŞÇ0-9])")
_URLISH = re.compile(r"https?://|www\.|@|\.(com|uz|ru|org)\b")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for line in text.splitlines() for s in _SENT_END.split(line) if s.strip()]


def good_sentence(s: str, min_words: int = 5, max_words: int = 50) -> bool:
    words = s.split()
    if not min_words <= len(words) <= max_words or _URLISH.search(s):
        return False
    if s[-1] not in ".!?…\"”»)":
        return False
    letters = sum(c.isalpha() for c in s)
    return letters / len(s) > 0.7


def mono_sentences(docs: Iterable[str], lang: str, blocked: set[int] = frozenset(),
                   limit: int | None = None) -> Iterator[str]:
    norm = normalize_uz if lang == "uz" else normalize_en
    seen: set[int] = set()
    n = 0
    for doc in docs:
        for s in split_sentences(norm(doc)):
            if not good_sentence(s):
                continue
            key = hash64(lang, near_key(s))
            if key in seen or key in blocked:
                continue
            seen.add(key)
            yield s
            n += 1
            if limit and n >= limit:
                return


def main(argv=None):
    from datasets import load_dataset

    from src.data_pipeline.cleaning import blocklist_keys
    from src.evaluation.evaluate import load_eval

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=list(MONO), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max", type=int, default=3_000_000)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/processed/eval"))
    args = ap.parse_args(argv)

    blocked = blocklist_keys(
        s for f in sorted(args.eval_dir.glob("*.jsonl")) for r in load_eval(f)
        for s in (r["en"], r["uz"])
    )
    repo, config = MONO[args.lang]
    docs = (row["text"] for row in load_dataset(repo, config, split="train", streaming=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, s in enumerate(mono_sentences(docs, args.lang, blocked, args.max), 1):
            f.write(s + "\n")
            if i % 500_000 == 0:
                print(f"{i:,} sentences", flush=True)


if __name__ == "__main__":
    main()
