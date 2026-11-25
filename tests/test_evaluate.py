import pytest
import torch

from src.data_pipeline.tokenizer import BOS, EOS, PAD
from src.evaluation.evaluate import evaluate, score, translate
from src.models.transformer import ModelConfig, Seq2SeqTransformer
from src.train import load_config


class _StubTokenizer:
    def __init__(self):
        self.tag_ids = {"uz": 4, "en": 5}

    def encode(self, texts):
        return [[10 + (ord(c) % 20) for c in t] for t in texts]

    def source_ids(self, ids, tgt_lang, bt=False):
        return [self.tag_ids[tgt_lang], *ids, EOS]

    def decode(self, ids):
        ids = [i for i in ids if i not in (PAD, BOS)]
        ids = ids[: ids.index(EOS)] if EOS in ids else ids
        return " ".join(map(str, ids))


def _model():
    torch.manual_seed(0)
    return Seq2SeqTransformer(ModelConfig(
        vocab_size=40, d_model=32, n_heads=4, n_kv_heads=4, enc_layers=1, dec_layers=1,
        ffn_dim=64, dropout=0.0, max_len=12,
    ))


def test_score_perfect_and_empty():
    sent = "bugun havo juda yaxshi va quyoshli"
    perfect = score([sent], [sent])
    assert perfect == {"bleu": pytest.approx(100.0), "chrf": pytest.approx(100.0)}
    assert score([""], ["salom dunyo"])["chrf"] == 0.0


def test_translate_keeps_input_order_across_length_buckets():
    model, tok = _model(), _StubTokenizer()
    sents = ["a much longer sentence", "hi", "mid size"]
    batched = translate(model, tok, sents, "uz", beam=2, batch_size=2)
    single = [translate(model, tok, [s], "uz", beam=2)[0] for s in sents]
    assert batched == single
    assert model.training  # translate restores the mode it found


def test_translate_without_autocast_matches_default():
    model, tok = _model(), _StubTokenizer()
    sents = ["hello there", "hi"]
    assert translate(model, tok, sents, "uz", beam=2, autocast=False) == \
        translate(model, tok, sents, "uz", beam=2)


def test_evaluate_reports_both_directions():
    rows = [{"en": "hello", "uz": "salom"}, {"en": "thanks", "uz": "rahmat"}]
    metrics = evaluate(_model(), _StubTokenizer(), rows, beam=1)
    assert {"en2uz_bleu", "en2uz_chrf", "uz2en_bleu", "uz2en_chrf", "chrf"} <= set(metrics)


def test_config_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 0.001\n  stable_ce: true\n")
    cfg = load_config(path, ["train.stable_ce=false", "model.d_model=128"])
    assert cfg["train"] == {"lr": 0.001, "stable_ce": False}
    assert cfg["model"]["d_model"] == 128
