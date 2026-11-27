import pytest

from src.text import detect_target


@pytest.mark.parametrize("text", [
    "Китоб ўқиш фойдали.",
    "Toshkent O'zbekistonning poytaxti.",
    "Toshkent Oʻzbekistonning poytaxti.",
    "Bugun havo juda yaxşi.",
    "Men kitob öqiyman",
    "salom",
    "qayerdasiz",
])
def test_uzbek_input_targets_english(text):
    assert detect_target(text) == "en"


@pytest.mark.parametrize("text", [
    "The weather is beautiful today.",
    "Please send me the report by Friday.",
    "hello",
    "What is your name?",
    "Quick question about the queue",
    "The dog's bowl is empty and I don't know why.",
    "Iraq is in the Middle East.",
])
def test_english_input_targets_uzbek(text):
    assert detect_target(text) == "uz"


@pytest.mark.parametrize("text", ["", "   ", "12345", "2026!", "Tashkent", "OK"])
def test_undecided_input_returns_none(text):
    assert detect_target(text) is None
