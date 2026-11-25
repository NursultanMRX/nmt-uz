import pytest
import torch

from src.models.transformer import ModelConfig, Seq2SeqTransformer

PAD, BOS, EOS = 0, 1, 2


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=50, d_model=32, n_heads=4, n_kv_heads=2,
        enc_layers=2, dec_layers=2, ffn_dim=64, dropout=0.0, max_len=64,
    )
    return Seq2SeqTransformer(cfg).eval()


def _batch():
    src = torch.tensor([[5, 6, 7, 8, EOS], [9, 10, EOS, PAD, PAD]])
    tgt = torch.tensor([[BOS, 11, 12, 13], [BOS, 14, 15, PAD]])
    return src, tgt


def test_forward_shape_and_tied_embeddings(model):
    src, tgt = _batch()
    assert model(src, tgt).shape == (2, 4, 50)
    assert model.lm_head.weight is model.embed.weight


def test_decoder_is_causal(model):
    src, tgt = _batch()
    changed = tgt.clone()
    changed[:, -1] = 42
    a, b = model(src, tgt), model(src, changed)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_source_padding_has_no_effect(model):
    src, tgt = _batch()
    padded = torch.cat([src, torch.full((2, 3), PAD)], dim=1)
    assert torch.allclose(model(src, tgt), model(padded, tgt), atol=1e-5)


def test_kv_cache_matches_full_forward(model):
    src, tgt = _batch()
    full = model(src, tgt)
    memory, mask = model.encode(src)
    caches = model.new_caches()
    steps = [model.decode(tgt[:, t : t + 1], memory, mask, caches, start_pos=t)
             for t in range(tgt.size(1))]
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_greedy_generate_matches_manual_loop(model):
    src, _ = _batch()
    out = model.generate(src, bos_id=BOS, eos_id=EOS, max_len=6, beam=1)
    seq = torch.full((2, 1), BOS)
    for _ in range(6):
        nxt = model(src, seq)[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    manual = seq[:, 1:]
    for row_out, row_manual in zip(out, manual):
        n = (row_manual == EOS).nonzero()
        stop = n[0, 0] + 1 if len(n) else len(row_manual)
        assert row_out[:stop].tolist() == row_manual[:stop].tolist()


def test_beam_search_shapes(model):
    src, _ = _batch()
    out = model.generate(src, bos_id=BOS, eos_id=EOS, max_len=8, beam=4)
    assert out.shape[0] == 2 and out.shape[1] <= 8
