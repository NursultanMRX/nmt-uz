"""Train the bidirectional model from a YAML config.

    python -m src.train --config configs/base.yaml --data data/processed --run-dir runs/base
    python -m src.train ... --set train.stable_ce=false --set train.total_steps=20000
"""

import argparse
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import torch
import yaml

from src.data_pipeline.tokenizer import Tokenizer
from src.evaluation.evaluate import evaluate, load_eval
from src.models.transformer import ModelConfig, Seq2SeqTransformer
from src.training.data import PairStore
from src.training.trainer import TrainConfig, Trainer


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        cfg.setdefault(section, {})[name] = yaml.safe_load(value)
    return cfg


def _pick(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def build_trainer(cfg: dict, data_dir: Path, run_dir: Path, device: str,
                  on_save: Callable[[], None] | None = None) -> Trainer:
    d = cfg["data"]
    tok = Tokenizer(data_dir / d["tokenizer"])
    store = PairStore.load(data_dir / d["train"])
    if d.get("extra"):
        store = PairStore.concat([store, *(PairStore.load(data_dir / p) for p in d["extra"])])
    cooldown = PairStore.load(data_dir / d["cooldown"]) if d.get("cooldown") else None

    model_cfg = _pick(ModelConfig, {**cfg["model"], "vocab_size": tok.vocab_size})
    train_cfg = _pick(TrainConfig, cfg["train"])
    model = Seq2SeqTransformer(model_cfg)
    print(f"params: {model.num_params() / 1e6:.1f}M, pairs: {len(store):,}", flush=True)

    rows = load_eval(data_dir / d["eval"])

    def evaluate_fn(m):
        return evaluate(m, tok, rows, beam=train_cfg.eval_beam,
                        max_sents=train_cfg.eval_max_sents)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Trainer(model, train_cfg, store, tok.tag_ids, tok.bt_id, run_dir, device=device,
                   cooldown_store=cooldown, evaluate_fn=evaluate_fn, on_save=on_save)


def main(argv=None, on_save=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = build_trainer(cfg, args.data, args.run_dir, device, on_save)
    if args.resume and trainer.load_checkpoint(args.run_dir / "last.pt"):
        print(f"resumed at step {trainer.step}", flush=True)
    if args.max_steps:
        trainer.cfg.total_steps = min(trainer.cfg.total_steps, args.max_steps)
    trainer.fit()


if __name__ == "__main__":
    main()
