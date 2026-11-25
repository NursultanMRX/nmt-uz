import pytest

from src.data_pipeline.alphabet import detect_script, to_new_latin, to_old_latin


@pytest.mark.parametrize(
    "old, new",
    [
        ("o'zbek", "özbek"),
        ("Oʻzbekiston", "Özbekiston"),
        ("OʻZBEKISTON", "ÖZBEKISTON"),
        ("g'alaba", "ğalaba"),
        ("Gʻalaba", "Ğalaba"),
        ("shahar", "şahar"),
        ("Shahar", "Şahar"),
        ("SHAHAR", "ŞAHAR"),
        ("choy", "çoy"),
        ("Choy", "Çoy"),
        ("to'g'ri", "töğri"),
        ("tong", "tong"),
    ],
)
def test_old_latin(old, new):
    assert to_new_latin(old) == new


@pytest.mark.parametrize("apos", ["'", "ʼ", "’", "`", "‘", "ʻ"])
def test_tutuq_kept_as_single_form(apos):
    assert to_new_latin(f"ma{apos}no") == "maʼno"
    assert to_new_latin(f"ta{apos}lim") == "taʼlim"


def test_apostrophe_between_s_and_h_is_not_digraph():
    assert to_new_latin("as'hob") == "asʼhob"


@pytest.mark.parametrize(
    "cyr, new",
    [
        ("Ўзбекистон", "Özbekiston"),
        ("ЎЗБЕКИСТОН", "ÖZBEKISTON"),
        ("шаҳар", "şahar"),
        ("Шаҳар", "Şahar"),
        ("чой", "çoy"),
        ("ЧОЙ", "ÇOY"),
        ("ғалаба", "ğalaba"),
        ("қалам", "qalam"),
        ("ёзув", "yozuv"),
        ("ЁЗУВ", "YOZUV"),
        ("Юрак", "Yurak"),
        ("яхши", "yaxşi"),
        ("маъно", "maʼno"),
        ("цех", "tsex"),
    ],
)
def test_cyrillic(cyr, new):
    assert to_new_latin(cyr) == new


def test_urls_numbers_and_empty_untouched():
    assert to_new_latin("https://sh.example.com/ch 123") == "https://sh.example.com/ch 123"
    assert to_new_latin("") == ""


@pytest.mark.parametrize(
    "text", ["O'zbekiston shahar", "Ўзбекистон шаҳар", "maʼno ğalaba", "To'g'ri choy"]
)
def test_idempotent(text):
    once = to_new_latin(text)
    assert to_new_latin(once) == once


def test_detect_script():
    assert detect_script("Salom dunyo") == "latin"
    assert detect_script("Салом дунё") == "cyrillic"
    assert detect_script("Salom дунё") == "mixed"
    assert detect_script("123 !?") == "none"


@pytest.mark.parametrize(
    "new, old",
    [
        ("Ğalaba", "Gʻalaba"),
        ("öqituvçi", "oʻqituvchi"),
        ("Şahar", "Shahar"),
        ("ŞAHAR", "SHAHAR"),
        ("Çoy", "Choy"),
        ("ÖZBEKISTON", "OʻZBEKISTON"),
        ("maʼno", "maʼno"),
    ],
)
def test_to_old_latin(new, old):
    assert to_old_latin(new) == old


@pytest.mark.parametrize("text", ["Töğri, şahar çiroyli!", "ÖZBEKISTON", "Ş", "asʼhob"])
def test_old_latin_round_trip(text):
    assert to_new_latin(to_old_latin(text)) == text
