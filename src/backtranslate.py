"""Tagged back-translation (Caswell et al., 2019).

Monolingual text in language L is translated into the other language with the
current model. The synthetic pair trains only the direction whose target is the
real L text, and its source gets a <bt> tag so the model can tell it apart.

    python -m src.backtranslate --checkpoint runs/base/best.pt --data data/processed \
        --mono mono/uz.txt --lang uz --out stores/bt_uz
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data_pipeline.cleaning import normalize_en, normalize_uz, reject_reason
from src.data_pipeline.tokenizer import Tokenizer
from src.evaluate_run import load_model
from src.evaluation.evaluate import translate
from src.training.data import EN2UZ, UZ2EN, PairStore


def synthetic_store(real: list[str], synth: list[str], lang: str, tok) -> tuple[PairStore, int]:
    """Pairs real `lang` text with its synthetic translation; drops broken outputs."""
    en, uz = (synth, real) if lang == "uz" else (real, synth)
    keep = [i for i in range(len(real)) if reject_reason(en[i], uz[i]) is None]
    en_ids = tok.encode([en[i] for i in keep])
    uz_ids = tok.encode([uz[i] for i in keep])
    only = EN2UZ if lang == "uz" else UZ2EN
    store = PairStore.build(en_ids, uz_ids, only_dir=np.full(len(keep), only, np.int8))
    return store, len(real) - len(keep)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--tokenizer", default="spm/spm.model")
    ap.add_argument("--mono", type=Path, required=True, help="one sentence per line")
    ap.add_argument("--lang", choices=["uz", "en"], required=True)
    ap.add_argument("--out", required=True, help="store dir, relative to --data")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=192)
    args = ap.parse_args(argv)

    norm = normalize_uz if args.lang == "uz" else normalize_en
    with open(args.mono, encoding="utf-8") as f:
        real = [s for s in (norm(line) for line in f) if s][: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    tok = Tokenizer(args.data / args.tokenizer)
    other = "en" if args.lang == "uz" else "uz"
    synth = translate(model, tok, real, other, beam=args.beam, batch_size=args.batch_size)

    store, dropped = synthetic_store(real, synth, args.lang, tok)
    out = args.data / args.out
    store.save(out)
    pl.DataFrame({args.lang: real, other: synth}).write_parquet(out / "text.parquet")
    print(f"kept {len(store):,} synthetic pairs, dropped {dropped:,}")


if __name__ == "__main__":
    main()
