"""End-to-end smoke test on a small local build: train a tiny model, resume, evaluate.

    python -m src.data_pipeline.build --out data/smoke --only ted2020,qed --vocab-size 2000 --skip-eval
    python scripts/smoke.py
"""

import json
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate_run, train

DATA, RUN = Path("data/smoke"), Path("runs/smoke")

# Without HF_TOKEN there is no FLORES+, so a slice of the training data stands in.
(DATA / "eval").mkdir(parents=True, exist_ok=True)
rows = pl.read_parquet(DATA / "train" / "*.parquet").head(20).select("en", "uz").rows()
with open(DATA / "eval" / "flores_dev.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps({"id": i, "en": en, "uz": uz}, ensure_ascii=False) + "\n"
                 for i, (en, uz) in enumerate(rows))

tiny = ["model.d_model=64", "model.n_heads=4", "model.n_kv_heads=4", "model.enc_layers=1",
        "model.dec_layers=1", "model.ffn_dim=128", "model.max_len=64", "train.max_tokens=1000",
        "train.total_steps=20", "train.warmup_steps=5", "train.eval_every=10",
        "train.save_every=10", "train.log_every=5", "train.eval_max_sents=8",
        "train.fp16=false", "train.eval_beam=2"]
args = ["--config", "configs/base.yaml", "--data", str(DATA), "--run-dir", str(RUN)]
for s in tiny:
    args += ["--set", s]

train.main(args)
train.main([*args, "--resume"])
evaluate_run.main(["--run-dir", str(RUN), "--data", str(DATA), "--split", "dev",
                   "--beam", "2", "--checkpoint", "best.pt"])
print(sorted(os.listdir(RUN)), sorted(os.listdir(RUN / "eval")))
