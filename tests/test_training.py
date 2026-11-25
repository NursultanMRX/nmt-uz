import numpy as np
import pytest
import torch

from src.data_pipeline.tokenizer import BOS, EOS, PAD
from src.models.transformer import ModelConfig, Seq2SeqTransformer
from src.training.data import EN2UZ, UZ2EN, PairStore, TokenBatcher, collate
from src.training.trainer import TrainConfig, Trainer, wsd_lr

TAGS = {"uz": 4, "en": 5}
BT = 6


def _store(n=40, seed=0, only_dir=None):
    rng = np.random.default_rng(seed)
    en = [rng.integers(10, 30, rng.integers(2, 8)).tolist() for _ in range(n)]
    uz = [[t + 20 for t in s] for s in en]  # learnable toy mapping
    return PairStore.build(en, uz, only_dir=only_dir)


def test_store_roundtrip(tmp_path):
    store = _store()
    store.save(tmp_path / "store")
    loaded = PairStore.load(tmp_path / "store")
    assert len(loaded) == len(store)
    assert loaded.side(3, "uz").tolist() == store.side(3, "uz").tolist()


def test_store_concat_keeps_pairs_and_flags():
    a, b = _store(n=3, seed=1), _store(n=2, seed=2, only_dir=[EN2UZ, UZ2EN])
    both = PairStore.concat([a, b])
    assert len(both) == 5
    assert both.side(4, "en").tolist() == b.side(1, "en").tolist()
    assert both.only_dir.tolist() == [-1, -1, -1, EN2UZ, UZ2EN]


def test_batcher_respects_budget_and_covers_both_directions():
    store = _store()
    batcher = TokenBatcher(store, max_tokens=40, max_len=16)
    seen = []
    for batch in batcher:
        longest = max(int(batcher.lens[np.searchsorted(batcher.ids, ex)]) for ex in batch)
        assert longest * len(batch) <= 40 or len(batch) == 1
        seen += batch
    assert sorted(seen) == list(range(len(store) * 2))


def test_backtranslated_pairs_only_train_their_direction():
    store = _store(n=4, only_dir=[-1, EN2UZ, UZ2EN, -1])
    ids = sorted(ex for b in TokenBatcher(store, 100, 16) for ex in b)
    assert ids == [0, 1, 2, 5, 6, 7]


def test_collate_layout():
    store = _store(n=2, only_dir=[-1, EN2UZ])
    batch = collate(store, [0, 2, 1], TAGS, BT, max_len=16)
    src, tin, tout = batch["src"], batch["tgt_in"], batch["tgt_out"]
    en0, uz0 = store.side(0, "en").tolist(), store.side(0, "uz").tolist()
    assert src[0, : len(en0) + 2].tolist() == [TAGS["uz"], *en0, EOS]
    assert src[2, 0].item() == TAGS["en"]
    assert src[1, :2].tolist() == [TAGS["uz"], BT]
    assert tin[0, : len(uz0) + 1].tolist() == [BOS, *uz0]
    assert tout[0, : len(uz0) + 1].tolist() == [*uz0, EOS]
    assert (tout[0, len(uz0) + 1 :] == PAD).all()


def test_wsd_schedule():
    cfg = TrainConfig(lr=1.0, warmup_steps=10, total_steps=100, decay_frac=0.2, min_lr_ratio=0.1)
    assert wsd_lr(0, cfg) == pytest.approx(0.1)
    assert wsd_lr(9, cfg) == pytest.approx(1.0)
    assert wsd_lr(50, cfg) == 1.0
    assert wsd_lr(79, cfg) == 1.0
    assert wsd_lr(99, cfg) == pytest.approx(0.1, abs=0.05)
    assert wsd_lr(90, cfg) < 1.0


def _trainer(tmp_path, steps):
    torch.manual_seed(0)
    model = Seq2SeqTransformer(ModelConfig(
        vocab_size=60, d_model=32, n_heads=4, n_kv_heads=4, enc_layers=1, dec_layers=1,
        ffn_dim=64, dropout=0.0, max_len=32,
    ))
    cfg = TrainConfig(
        max_tokens=200, accum=1, lr=3e-3, warmup_steps=5, total_steps=steps,
        eval_every=10_000, save_every=steps, fp16=False, label_smoothing=0.0,
    )
    store = _store(n=8)
    return Trainer(model, cfg, store, TAGS, BT, run_dir=tmp_path, device="cpu")


def test_trainer_overfits_tiny_corpus(tmp_path):
    trainer = _trainer(tmp_path, steps=150)
    first = trainer.train_step()
    trainer.fit()
    assert trainer.last_loss < first * 0.3


def test_checkpoint_resume(tmp_path):
    trainer = _trainer(tmp_path, steps=20)
    trainer.fit()
    resumed = _trainer(tmp_path, steps=20)
    assert resumed.load_checkpoint(tmp_path / "last.pt")
    assert resumed.step == 20
    for a, b in zip(trainer.model.parameters(), resumed.model.parameters()):
        assert torch.equal(a, b)
