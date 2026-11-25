import pytest

from src.data_pipeline.alphabet import to_old_latin
from src.evaluation.evaluate import translate as beam_translate
from src.export import export
from src.inference import Translator, split_emoji, split_long


@pytest.fixture
def translator(tmp_path, checkpoint, spm_model):
    release = export(checkpoint[0], spm_model, tmp_path / "release")
    return Translator.from_pretrained(release, device="cpu")


def test_order_and_batching(translator):
    texts = ["good morning", "hello", "a good book about the city school"]
    batched = translator.translate(texts, "uz", beam=2, batch_size=2)
    assert batched == [translator.translate([t], "uz", beam=2)[0] for t in texts]


def test_single_string_and_empty_inputs(translator):
    assert isinstance(translator.translate("hello", "uz"), str)
    out = translator.translate(["", "hello", "   "], "uz")
    assert out[0] == "" and out[2] == ""
    assert out[1] == translator.translate("hello", "uz")


def test_uzbek_input_in_any_script(translator):
    new = translator.translate("Özbekiston şahar", "en")
    assert translator.translate("Oʻzbekiston shahar", "en") == new
    assert translator.translate("Ўзбекистон шаҳар", "en") == new


def test_long_input_is_split_into_sentences(translator):
    first, second = "Good morning.", "The city school is good."
    joined = translator.translate(f"{first} {second}", "uz")
    assert joined == " ".join(translator.translate([first, second], "uz"))


def test_old_script_output(translator):
    texts = ["good idea", "read the book"]
    new = translator.translate(texts, "uz")
    assert translator.translate(texts, "uz", script="old") == [to_old_latin(t) for t in new]


def test_emoji_are_moved_to_the_end_of_the_sentence(translator):
    plain = translator.translate("good morning", "uz")
    assert translator.translate("good morning ❤️😊", "uz") == plain + " ❤️😊"
    assert translator.translate("good 👍 morning", "uz") == plain + " 👍"
    assert translator.translate("👍🏽", "uz") == "👍🏽"


def test_split_emoji_leaves_other_symbols():
    assert split_emoji("Price $25 → €23 #AI @me 50%") == ("Price $25 → €23 #AI @me 50%", "")
    assert split_emoji("ok 👨‍👩‍👧 fine ❤️") == ("ok fine", "👨‍👩‍👧❤️")


def test_split_long_at_punctuation_then_words():
    words = lambda s: len(s.split())
    assert split_long("a b, c d", words, limit=4) == ["a b, c d"]
    assert split_long("a b c, d e f, g h", words, limit=4) == ["a b c,", "d e f,", "g h"]
    assert split_long("a b c d e f g h i", words, limit=4) == ["a b c d", "e f g h", "i"]


def test_long_sentence_is_translated_in_pieces(translator, monkeypatch):
    pieces = ["the good school in the city,", "a book about the morning"]
    limit = max(len(ids) for ids in translator.tok.encode(pieces))  # each piece fits, both do not
    monkeypatch.setattr("src.inference.MAX_SOURCE_TOKENS", limit)
    joined = translator.translate(" ".join(pieces), "uz")
    assert joined == " ".join(translator.translate(pieces, "uz"))


def test_output_length_is_capped_by_max_ratio(translator, monkeypatch):
    limits = []
    generate = translator.model.generate

    def spy(src, *args, max_len, **kwargs):
        limits.append((src.size(1), max_len))
        return generate(src, *args, max_len=max_len, **kwargs)

    monkeypatch.setattr(translator.model, "generate", spy)
    beam_translate(translator.model, translator.tok, ["hello"], "uz", max_ratio=2)
    beam_translate(translator.model, translator.tok, ["hello"], "uz")
    (width, capped), (_, default) = limits
    assert capped == 2 * width + 10
    assert default == translator.model.cfg.max_len


def test_bad_arguments(translator):
    with pytest.raises(ValueError):
        translator.translate("hello", "ru")
    with pytest.raises(ValueError):
        translator.translate("hello", "uz", script="cyrillic")
