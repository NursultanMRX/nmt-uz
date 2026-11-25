import polars as pl

from src.data_pipeline.cleaning import (
    blocklist_keys,
    clean_pair,
    dedup_reasons,
    near_key,
    normalize_en,
    reject_reason,
)


def test_normalize_en_collapses_whitespace():
    assert normalize_en("  Hello   world \n") == "Hello world"


def test_near_key_ignores_case_and_punctuation():
    assert near_key("Salom, dunyo!") == near_key("salom dunyo")
    assert near_key("Salom dunyo") != near_key("Salom döst")


def test_reject_reasons():
    assert reject_reason("", "salom") == "empty"
    assert reject_reason("hello", "hello") == "copy"
    assert reject_reason("hi", "bu juda uzun gap " * 10) == "ratio"
    assert reject_reason("word " * 300, "söz " * 300) == "too_long"
    assert reject_reason("<div>hi</div>", "salom") == "markup"
    assert reject_reason("1234 5678", "1234 5678 9") in {"copy", "non_text"}
    assert reject_reason("How are you?", "Qalaysiz?") is None


def _run(rows, blocklist=(), max_per_side=3):
    cleaned = [clean_pair(en, uz) for en, uz in rows]
    keys = pl.DataFrame(
        [c[2:] for c in cleaned], schema=["k_pair", "k_en", "k_uz"], orient="row"
    )
    reasons = dedup_reasons(keys, blocklist_keys(blocklist), max_per_side).to_list()
    return cleaned, reasons


def test_converts_uz_and_dedups_across_sources():
    cleaned, reasons = _run([
        ("Good morning", "Xayrli tong, o'rtoq"),
        ("good morning", "Хайрли тонг, ўртоқ"),
        ("Thank you", "Rahmat"),
    ])
    assert cleaned[0][:2] == ("Good morning", "Xayrli tong, örtoq")
    assert reasons == [None, "duplicate", None]


def test_drops_eval_leakage():
    _, reasons = _run([("thank you", "Katta rahmat")], blocklist=["Thank you!", "Rahmat"])
    assert reasons == ["leakage"]


def test_caps_translations_per_source_sentence():
    _, reasons = _run(
        [("Hello", "Salom"), ("Hello", "Assalom"), ("Hello", "Salom döstim")],
        max_per_side=2,
    )
    assert reasons == [None, None, "over_cap"]
