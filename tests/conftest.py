import random
from dataclasses import asdict

import pytest
import torch

from src.data_pipeline.tokenizer import train_tokenizer
from src.models.transformer import ModelConfig, Seq2SeqTransformer

_WORDS = [
    "salom", "dunyo", "kitob", "öqiş", "yaxşi", "ğoya", "çiroyli", "şahar", "maktab", "tong",
    "rahmat", "hello", "world", "book", "read", "good", "idea", "city", "school", "morning",
    "thanks",
]


@pytest.fixture(scope="session")
def spm_model(tmp_path_factory):
    rng = random.Random(0)
    texts = [" ".join(rng.choices(_WORDS, k=rng.randint(3, 9))) + "." for _ in range(3000)]
    return train_tokenizer(texts, tmp_path_factory.mktemp("spm") / "spm", vocab_size=300)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=300, d_model=32, n_heads=4, n_kv_heads=2, enc_layers=1,
                      dec_layers=1, ffn_dim=64, dropout=0.0, max_len=32)
    model = Seq2SeqTransformer(cfg).eval()
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "model_cfg": asdict(cfg), "step": 7}, path)
    return path, model
