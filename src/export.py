"""Turn a training checkpoint into release files: safetensors weights, config.json, tokenizer.

    python -m src.export --checkpoint runs/base/best.pt --tokenizer data/processed/spm/spm.model \
        --out release/
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from src.data_pipeline.tokenizer import BOS, BT_TAG, EOS, LANG_TAGS, PAD, UNK, Tokenizer
from src.models.transformer import ModelConfig, Seq2SeqTransformer

TIED = "lm_head.weight"


def export(checkpoint: str | Path, tokenizer: str | Path, out: str | Path,
           dtype: torch.dtype = torch.float16) -> Path:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    vocab = Tokenizer(tokenizer).vocab_size
    if cfg.vocab_size != vocab:
        raise ValueError(f"model vocab {cfg.vocab_size} != tokenizer vocab {vocab}")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.to(dtype).contiguous() for k, v in ckpt["model"].items()}
    if cfg.tie_embeddings:
        state.pop(TIED, None)  # safetensors refuses shared storage
    save_file(state, out / "model.safetensors", metadata={"format": "pt"})
    shutil.copy(tokenizer, out / "spm.model")
    config = {
        "model": asdict(cfg),
        "tags": LANG_TAGS,
        "bt_tag": BT_TAG,
        "special_ids": {"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK},
        "step": ckpt.get("step"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out


def load_release(path: str | Path, device: str = "cpu") -> Seq2SeqTransformer:
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Seq2SeqTransformer(ModelConfig(**config["model"]))
    state = {k: v.float() for k, v in load_file(path / "model.safetensors").items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) - {TIED}:
        raise ValueError(f"weights do not match config: missing={missing} unexpected={unexpected}")
    return model.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("release"))
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args(argv)
    out = export(args.checkpoint, args.tokenizer, args.out,
                 torch.float32 if args.fp32 else torch.float16)
    for f in sorted(out.iterdir()):
        print(f"{f.stat().st_size / 2**20:8.1f} MB  {f}")


if __name__ == "__main__":
    main()
