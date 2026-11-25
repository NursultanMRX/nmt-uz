import pytest
import torch

from src.efficiency import cast, render_table, sweep, weight_bytes
from src.models.transformer import ModelConfig, Seq2SeqTransformer


def _model():
    torch.manual_seed(0)
    return Seq2SeqTransformer(ModelConfig(
        vocab_size=40, d_model=32, n_heads=4, n_kv_heads=4, enc_layers=1, dec_layers=1,
        ffn_dim=64, dropout=0.0, max_len=12,
    ))


def test_weight_bytes_counts_tied_weights_once():
    model = _model()
    unique = {p.data_ptr(): p for p in model.parameters()}.values()
    assert weight_bytes(model) == sum(p.numel() * 4 for p in unique)


@pytest.mark.parametrize("precision, dtype", [("fp32", torch.float32), ("fp16", torch.float16)])
def test_cast_sets_weight_dtype(precision, dtype):
    model = cast(_model(), precision)
    assert {p.dtype for p in model.parameters()} == {dtype}


def test_cast_fp16_halves_the_weights():
    model = _model()
    full = weight_bytes(model)
    assert weight_bytes(cast(model, "fp16")) == full // 2


def test_cast_rejects_unknown_precision():
    with pytest.raises(ValueError):
        cast(_model(), "fp8")


def test_sweep_times_every_batch_size_and_keeps_quality_hyps():
    calls = []

    def translate(texts, target, beam, batch_size):
        calls.append((target, batch_size))
        return [t.upper() for t in texts]

    rows = [{"en": "hi", "uz": "salom"}, {"en": "bye", "uz": "xayr"}]
    out = sweep(translate, rows, batch_sizes=[1, 2], beam=4, quality_batch=2, warmup=False)
    assert set(out["seconds"]) == {1, 2}
    assert ("uz", 1) in calls and ("en", 2) in calls
    assert out["hyps"] == {"en2uz": ["HI", "BYE"], "uz2en": ["SALOM", "XAYR"]}


def test_render_table_has_a_row_per_variant_and_a_column_per_batch():
    results = [
        {"system": "ours", "precision": "fp16", "weights_mb": 153.0,
         "seconds": {"1": 120.4, "32": 9.8},
         "quality": {"en2uz": {"bleu": 19.9, "chrf": 57.5}, "uz2en": {"bleu": 27.7, "chrf": 55.9}}},
        {"system": "nllb", "precision": "fp32", "weights_mb": 2460.0,
         "seconds": {"1": 400.0, "32": 30.1},
         "quality": {"en2uz": {"bleu": 14.9, "chrf": 52.8}, "uz2en": {"bleu": 31.3, "chrf": 58.4}}},
    ]
    lines = render_table(results).splitlines()
    assert lines[0] == ("| Model | Precision | Weights | batch 1 | batch 32 "
                        "| en→uz BLEU / chrF | uz→en BLEU / chrF |")
    assert len(lines) == 4
    assert lines[2].startswith("| NLLB-200-distilled-600M | fp32 | 2460 MB | 400 s | 30.1 s")
    assert lines[3].startswith("| **nmt-uz-76m** | fp16 | 153 MB | 120 s | 9.8 s")
    assert lines[3].endswith("| 19.9 / 57.5 | 27.7 / 55.9 |")


def test_sweep_records_out_of_memory_and_skips_larger_batches():
    calls = []

    def translate(texts, target, beam, batch_size):
        calls.append(batch_size)
        if batch_size > 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return texts

    rows = [{"en": "hi", "uz": "salom"}]
    out = sweep(translate, rows, batch_sizes=[1, 2, 4], quality_batch=1, warmup=False)
    assert out["seconds"][2] is None and out["seconds"][4] is None
    assert 4 not in calls
    assert out["hyps"] == {"en2uz": ["hi"], "uz2en": ["salom"]}


def test_render_table_marks_out_of_memory():
    results = [{"system": "nllb", "precision": "fp32", "weights_mb": 2460.0,
                "seconds": {"1": 400.0, "256": None},
                "quality": {"en2uz": {"bleu": 14.9, "chrf": 52.8},
                            "uz2en": {"bleu": 31.3, "chrf": 58.4}}}]
    assert "| 400 s | OOM |" in render_table(results)
