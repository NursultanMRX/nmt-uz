"""Score a trained checkpoint on a FLORES+ split and save metrics plus translations.

    python -m src.evaluate_run --run-dir runs/base --split devtest --beam 5
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data_pipeline.tokenizer import Tokenizer
from src.evaluation.evaluate import evaluate, load_eval
from src.models.transformer import ModelConfig, Seq2SeqTransformer


def load_model(ckpt_path: str | Path, device: str) -> Seq2SeqTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Seq2SeqTransformer(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer(args.data / cfg["data"]["tokenizer"])
    model = load_model(args.run_dir / args.checkpoint, device)
    rows = load_eval(args.data / "eval" / f"flores_{args.split}.jsonl")

    metrics, hyps = evaluate(model, tok, rows, beam=args.beam, return_hyps=True)
    name = f"{args.split}_{Path(args.checkpoint).stem}_beam{args.beam}"
    out = args.run_dir / "eval"
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(
        {"split": args.split, "checkpoint": args.checkpoint, "beam": args.beam,
         "sentences": len(rows), **metrics}, indent=2))
    for direction, lines in hyps.items():
        (out / f"{name}.{direction}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
