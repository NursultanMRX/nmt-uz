"""Where the parallel data comes from. See data/SOURCES.md for the audit."""

import io
import os
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

OPUS = "https://object.pouta.csc.fi/OPUS-{}/moses/en-uz.txt.zip"


@dataclass(frozen=True)
class Source:
    name: str
    kind: str  # "opus" or "hf"
    ref: str  # OPUS "<corpus>/<version>" or HF repo id
    splits: tuple[str, ...] = ("train",)
    config: str | None = None
    weight: float = 1.0
    clean: bool = False  # used for the cooldown phase


# Order is priority: when a pair appears in several sources, the first one wins.
SOURCES = [
    Source("ted2020", "opus", "TED2020/v1", clean=True),
    Source("qed", "opus", "QED/v2.0a", clean=True),
    Source("ml_jonibek", "hf", "ML-Jonibek/English-Uzbek-Translation-1", clean=True),
    Source("jonibek21", "hf", "Jonibek21/English-Uzbek-Translation", clean=True),
    Source("pythoninverter", "hf", "pythoninverter/English-Uzbek-Translation", clean=True),
    Source("wikimedia", "opus", "wikimedia/v20260327", clean=True),
    Source("opus100", "hf", "Helsinki-NLP/opus-100", config="en-uz",
           splits=("train", "validation", "test")),
    Source("shakhzo_clean", "hf", "ShakhzoDavronov/mt-dataset-en-uz-cleaned",
           splits=("train", "validation")),
    Source("shakhzo", "hf", "ShakhzoDavronov/mt-dataset-en-uz", splits=("train", "validation")),
    Source("tanzil", "opus", "Tanzil/v1", weight=0.5),
    Source("kde4", "opus", "KDE4/v2", weight=0.5),
    Source("gnome", "opus", "GNOME/v1", weight=0.5),
    Source("ubuntu", "opus", "Ubuntu/v14.10", weight=0.5),
    Source("translatewiki", "opus", "translatewiki/v2026-07-01", weight=0.5),
    Source("tldr", "opus", "tldr-pages/v2026-07-07", weight=0.5),
    Source("nllb", "opus", "NLLB/v1"),
]


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, part)
    part.replace(dest)
    return dest


def _iter_opus(src: Source, cache: Path) -> Iterator[tuple[str, str]]:
    path = _download(OPUS.format(src.ref), cache / "opus" / f"{src.name}.zip")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        en_name = next(n for n in names if n.endswith(".en"))
        uz_name = next(n for n in names if n.endswith(".uz"))
        with zf.open(en_name) as fe, zf.open(uz_name) as fu:
            en_lines = io.TextIOWrapper(fe, encoding="utf-8", errors="replace")
            uz_lines = io.TextIOWrapper(fu, encoding="utf-8", errors="replace")
            for en, uz in zip(en_lines, uz_lines, strict=True):
                yield en.rstrip("\n"), uz.rstrip("\n")


def _text(value) -> str:
    # Some HF uploads wrap each side as {"text": ...}.
    return value.get("text", "") if isinstance(value, dict) else (value or "")


def _iter_hf(src: Source) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    for split in src.splits:
        ds = load_dataset(src.ref, src.config, split=split)
        for row in ds:
            if "translation" in row:
                yield row["translation"]["en"], row["translation"]["uz"]
            else:
                yield _text(row["en"]), _text(row["uz"])


def iter_pairs(src: Source, cache: str | Path) -> Iterator[tuple[str, str]]:
    if src.kind == "opus":
        return _iter_opus(src, Path(cache))
    return _iter_hf(src)


def load_flores(split: str) -> list[dict]:
    """FLORES+ eng_Latn/uzn_Latn rows joined by id. Gated: needs HF_TOKEN."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split, token=token)
    uz = load_dataset("openlanguagedata/flores_plus", "uzn_Latn", split=split, token=token)
    uz_by_id = {r["id"]: r["text"] for r in uz}
    return [{"id": r["id"], "en": r["text"], "uz": uz_by_id[r["id"]]}
            for r in en if r["id"] in uz_by_id]
