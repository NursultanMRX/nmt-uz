import json

import pytest
import torch
from safetensors import safe_open

from src.export import export, load_release


def _logits(model):
    src = torch.tensor([[4, 10, 11, 12, 2], [5, 13, 14, 2, 0]])
    tgt = torch.tensor([[1, 20, 21], [1, 22, 0]])
    with torch.no_grad():
        return model(src, tgt)


def test_export_writes_release_files(tmp_path, checkpoint, spm_model):
    ckpt, _ = checkpoint
    out = export(ckpt, spm_model, tmp_path / "release")
    assert {p.name for p in out.iterdir()} >= {"model.safetensors", "config.json", "spm.model"}

    config = json.loads((out / "config.json").read_text())
    assert config["model"]["d_model"] == 32
    assert config["tags"] == {"uz": "<2uz>", "en": "<2en>"}

    with safe_open(out / "model.safetensors", "pt") as f:
        names = set(f.keys())
        assert all(f.get_tensor(n).dtype == torch.float16 for n in names)
    assert "lm_head.weight" not in names  # tied to embed.weight, restored on load


def test_fp32_round_trip_is_exact(tmp_path, checkpoint, spm_model):
    ckpt, model = checkpoint
    out = export(ckpt, spm_model, tmp_path / "release", dtype=torch.float32)
    loaded = load_release(out)
    assert loaded.lm_head.weight is loaded.embed.weight
    assert torch.allclose(_logits(loaded), _logits(model), atol=1e-6)


def test_fp16_round_trip_is_close(tmp_path, checkpoint, spm_model):
    ckpt, model = checkpoint
    loaded = load_release(export(ckpt, spm_model, tmp_path / "release"))
    assert next(loaded.parameters()).dtype == torch.float32
    assert torch.allclose(_logits(loaded), _logits(model), atol=1e-2)


def test_vocab_mismatch_is_rejected(tmp_path, checkpoint, spm_model):
    ckpt = checkpoint[0]
    state = torch.load(ckpt, weights_only=False)
    state["model_cfg"]["vocab_size"] = 512
    state["model"]["embed.weight"] = torch.zeros(512, 32)
    torch.save(state, ckpt)
    with pytest.raises(ValueError, match="vocab"):
        export(ckpt, spm_model, tmp_path / "release")
