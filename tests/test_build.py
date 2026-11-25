import polars as pl

from src.data_pipeline.build import dedup, encode_stores, tokenizer_sample, write_shards
from src.data_pipeline.cleaning import blocklist_keys
from src.data_pipeline.sources import Source
from src.training.data import PairStore


class _SerialPool:
    def map(self, fn, items):
        return list(map(fn, items))


class _CharTokenizer:
    def encode(self, texts):
        return [[ord(c) % 1000 for c in t] for t in texts]


CLEAN = Source("ted", "opus", "x", clean=True)
NOISY = Source("web", "opus", "y", weight=0.5)


def _build(tmp_path, blocklist=()):
    shards = tmp_path / "shards"
    a = write_shards(CLEAN, [("Good morning", "Xayrli tong"), ("", "böş")],
                     shards / "00_ted", _SerialPool())
    b = write_shards(NOISY, [("good morning!", "xayrli tong"), ("Thank you", "Rahmat"),
                             ("The weather is nice", "Ob-havo yaxshi")],
                     shards / "01_web", _SerialPool())
    files = sorted(shards.glob("*/*.parquet"))
    report = dedup(files, blocklist_keys(blocklist), tmp_path / "train")
    return a, b, report, sorted((tmp_path / "train").glob("*.parquet"))


def test_shards_dedup_and_leakage(tmp_path):
    a, _, report, train = _build(tmp_path, blocklist=["Thank you."])
    assert a == {"raw": 2, "empty": 1, "kept": 1}
    assert report["ted"] == {"kept": 1}
    assert report["web"] == {"duplicate": 1, "leakage": 1, "kept": 1}
    df = pl.read_parquet(train)
    assert df["en"].to_list() == ["Good morning", "The weather is nice"]
    assert df["clean"].to_list() == [True, False]


def test_tokenizer_sample_is_balanced(tmp_path):
    *_, train = _build(tmp_path)
    sample = tokenizer_sample(train, per_lang=10)
    assert len(sample) == 6 and "Rahmat" in sample and "Thank you" in sample


def test_encode_stores_splits_clean_subset(tmp_path):
    *_, train = _build(tmp_path)
    stats = encode_stores(train, _CharTokenizer(), tmp_path / "stores")
    assert stats["train"]["pairs"] == 3 and stats["clean"]["pairs"] == 1
    store = PairStore.load(tmp_path / "stores" / "train")
    assert store.weight.tolist() == [1.0, 0.5, 0.5]
    clean = PairStore.load(tmp_path / "stores" / "clean")
    assert clean.side(0, "uz").tolist() == _CharTokenizer().encode(["Xayrli tong"])[0]
