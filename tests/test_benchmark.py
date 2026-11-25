import json

import pytest
import torch

from src.benchmark import NLLB, render_table, score_all, text_metrics, to_reference_script


@pytest.mark.parametrize("ref, expected", [
    ("Oʻzbekiston shahri", "Oʻzbekiston shahri"),
    ("O'zbekiston shahri", "O'zbekiston shahri"),
    ("O‘zbekiston shahri", "O‘zbekiston shahri"),
])
def test_reference_script_follows_reference_apostrophe(ref, expected):
    assert to_reference_script(["Özbekiston şahri"], [ref]) == [expected]


def test_text_metrics_reports_chrf_and_chrfpp():
    perfect = text_metrics(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert perfect == {"bleu": 100.0, "chrf": 100.0, "chrfpp": 100.0}
    partial = text_metrics(["the cat sat on a mat"], ["the cat sat on the mat"])
    assert partial["chrf"] != partial["chrfpp"]


class _StubTok:
    def __init__(self):
        self.src_lang, self.calls = None, []

    def __call__(self, texts, **kwargs):
        self.calls.append((self.src_lang, list(texts)))
        return {"input_ids": torch.tensor([[len(t)] for t in texts])}

    def convert_tokens_to_ids(self, token):
        return {"eng_Latn": 7, "uzn_Latn": 8}[token]

    def batch_decode(self, ids, skip_special_tokens=True):
        return [f"out{int(i[0])}" for i in ids]


class _StubModel:
    device = torch.device("cpu")

    def __init__(self):
        self.kwargs = []

    def generate(self, input_ids, **kwargs):
        self.kwargs.append(kwargs)
        return input_ids


def test_nllb_language_codes_and_order():
    tok, model = _StubTok(), _StubModel()
    nllb = NLLB(tok, model)
    out = nllb.translate(["ccc", "a", "bb"], "uz", beam=5, batch_size=2)
    assert out == ["out3", "out1", "out2"]  # input order kept despite length sorting
    assert {src for src, _ in tok.calls} == {"eng_Latn"}
    assert all(k["forced_bos_token_id"] == 8 and k["num_beams"] == 5 for k in model.kwargs)
    nllb.translate(["x"], "en")
    assert tok.calls[-1][0] == "uzn_Latn" and model.kwargs[-1]["forced_bos_token_id"] == 7


def _rows(n):
    return [{"id": i, "en": f"good morning {i}", "uz": f"xayrli tong oʻqish {i}"} for i in range(n)]


def test_score_all_tracks_and_subsets():
    rows = _rows(4)
    hyps = {"en2uz": [f"xayrli tong öqiş {i}" for i in range(4)],
            "uz2en": [r["en"] for r in rows]}
    res = score_all("ours", rows, hyps, subsets={"full": None, "first2": 2},
                    comet=lambda src, mt, ref: [0.5, 0.7, 0.9, 0.9])
    assert res["first2"]["compat"]["en2uz"]["comet"] == pytest.approx(0.6)
    assert res["full"]["compat"]["uz2en"]["chrf"] == 100.0
    assert res["full"]["native"]["en2uz"]["chrf"] == 100.0
    assert res["full"]["compat"]["en2uz"]["chrf"] == 100.0  # öqiş -> oʻqish matches the reference
    assert res["full"]["sentences"] == 4 and res["first2"]["sentences"] == 2


def test_render_table(tmp_path):
    result = {"system": "ours", "model": "release", "split": "devtest",
              "first200": {"sentences": 200, "compat": {
                  "en2uz": {"bleu": 15.0, "chrf": 52.0, "chrfpp": 49.0, "comet": 0.85},
                  "uz2en": {"bleu": 25.0, "chrf": 55.0, "chrfpp": 53.0, "comet": 0.83}}}}
    (tmp_path / "ours_devtest.json").write_text(json.dumps(result))
    table = render_table(tmp_path, subset="first200")
    assert "ours" in table and "52.00" in table and "NLLB-200-uz-en-v1" in table


def test_render_table_shows_model_size(tmp_path):
    scores = {"bleu": 10.0, "chrf": 50.0, "chrfpp": 48.0}
    for system in ("ours", "nllb-1.3b"):
        result = {"system": system, "first200": {"compat": {"en2uz": scores, "uz2en": scores}}}
        (tmp_path / f"{system}_devtest.json").write_text(json.dumps(result))
    lines = render_table(tmp_path, subset="first200").splitlines()
    assert lines[0].startswith("| System | Params |")
    assert any("NLLB-200-distilled-1.3B" in line and "| 1.3B |" in line for line in lines)
    assert any("(ours)" in line and "| 76.5M |" in line for line in lines)
    assert any("Tilmoch" in line and "| n/a |" in line for line in lines)


def test_render_table_puts_ours_first(tmp_path):
    scores = {"bleu": 10.0, "chrf": 50.0, "chrfpp": 48.0}
    for system in ("nllb", "nllb-1.3b", "ours"):
        result = {"system": system, "first200": {"compat": {"en2uz": scores, "uz2en": scores}}}
        (tmp_path / f"{system}_devtest.json").write_text(json.dumps(result))
    rows = render_table(tmp_path, subset="first200").splitlines()[2:]
    assert "(ours)" in rows[0]
    assert "600M (our run)" in rows[1] and "1.3B (our run)" in rows[2]
