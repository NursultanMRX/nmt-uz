"""Translate lines from stdin.

    echo "Good morning, friend." | python -m src.translate --checkpoint runs/base/best.pt \
        --tokenizer data/processed/spm/spm.model --to uz
"""

import argparse
import sys

import torch

from src.data_pipeline.tokenizer import Tokenizer
from src.evaluate_run import load_model
from src.evaluation.evaluate import translate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--to", choices=["uz", "en"], required=True)
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    tok = Tokenizer(args.tokenizer)
    lines = [line.rstrip("\n") for line in sys.stdin]
    for out in translate(model, tok, lines, args.to, beam=args.beam):
        print(out)


if __name__ == "__main__":
    main()
