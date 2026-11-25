from src.data_pipeline.cleaning import blocklist_keys
from src.data_pipeline.mono import good_sentence, mono_sentences, split_sentences


def test_split_sentences():
    text = "Bugun havo yaxshi. Ertaga yomğir yoğadi! Şunday bölsa ham, biz boramiz."
    assert split_sentences(text) == [
        "Bugun havo yaxshi.", "Ertaga yomğir yoğadi!", "Şunday bölsa ham, biz boramiz."
    ]


def test_good_sentence_filters():
    assert good_sentence("Men har kuni ertalab kitob öqiyman.")
    assert not good_sentence("Juda qisqa.")
    assert not good_sentence("Batafsil malumot uçun www.example.uz saytiga kiring.")
    assert not good_sentence("Bu gap tugamagan va nuqtasi ham yöq")


def test_mono_converts_dedups_and_blocks_eval():
    docs = [
        "Oʻzbekiston Respublikasi poytaxti Toshkent shahridir. Bu shahar juda katta va gözal.",
        "O'zbekiston Respublikasi poytaxti Toshkent shahridir.",
        "Men har kuni ertalab kitob öqiyman.",
    ]
    blocked = blocklist_keys(["Men har kuni ertalab kitob o'qiyman."])
    out = list(mono_sentences(docs, "uz", blocked))
    assert out[0] == "Özbekiston Respublikasi poytaxti Toşkent şahridir."
    assert len(out) == 2  # cross-document duplicate and the eval sentence are dropped
