"""Training loop: fp16 AMP, token-budget batches, WSD schedule with a clean-data cooldown."""

import json
import math
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.transformer import Seq2SeqTransformer
from src.training.data import PairStore, TokenBatcher, collate
from src.training.losses import stable_cross_entropy


@dataclass
class TrainConfig:
    max_tokens: int = 12_000
    accum: int = 2
    lr: float = 7e-4
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.1
    warmup_steps: int = 4000
    total_steps: int = 100_000
    decay_frac: float = 0.1
    min_lr_ratio: float = 0.1
    cooldown_clean_ratio: float = 0.7
    label_smoothing: float = 0.1
    stable_ce: bool = True
    clip: float = 1.0
    fp16: bool = True
    eval_every: int = 2000
    save_every: int = 2000
    eval_beam: int = 4
    eval_max_sents: int = 500
    log_every: int = 100
    seed: int = 0


def wsd_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup-stable-decay: linear warmup, flat, then linear decay over the last decay_frac."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_steps = max(int(cfg.total_steps * cfg.decay_frac), 1)
    decay_start = cfg.total_steps - decay_steps
    if step < decay_start:
        return cfg.lr
    progress = min((step - decay_start + 1) / decay_steps, 1.0)
    return cfg.lr * (1.0 - progress * (1.0 - cfg.min_lr_ratio))


def _forever(batcher: TokenBatcher) -> Iterator[list[int]]:
    while True:
        yield from batcher


class Trainer:
    def __init__(
        self,
        model: Seq2SeqTransformer,
        cfg: TrainConfig,
        store: PairStore,
        tag_ids: dict[str, int],
        bt_id: int,
        run_dir: str | Path,
        device: str = "cuda",
        cooldown_store: PairStore | None = None,
        evaluate_fn: Callable[[Seq2SeqTransformer], dict] | None = None,
        on_save: Callable[[], None] | None = None,
        logger: Callable[[dict], None] | None = None,
    ):
        self.model, self.cfg = model.to(device), cfg
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tag_ids, self.bt_id = tag_ids, bt_id
        self.evaluate_fn, self.on_save, self.logger = evaluate_fn, on_save, logger
        self.max_len = model.cfg.max_len

        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.store, self.cooldown_store = store, cooldown_store
        self._make_batchers(cfg.seed)

        decay, no_decay = [], []
        for name, p in model.named_parameters():
            (no_decay if p.ndim < 2 or "embed" in name else decay).append(p)
        self.optim = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=cfg.betas, fused=device.startswith("cuda"),
        )
        self.use_amp = cfg.fp16 and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.step = 0
        self.best = -math.inf
        self.last_loss = math.nan

    def _make_batchers(self, seed: int) -> None:
        cfg = self.cfg
        self.batches = _forever(TokenBatcher(self.store, cfg.max_tokens, self.max_len, seed))
        self.cooldown_batches = (
            _forever(TokenBatcher(self.cooldown_store, cfg.max_tokens, self.max_len, seed + 1))
            if self.cooldown_store is not None else None
        )

    def _next_batch(self) -> dict[str, torch.Tensor]:
        in_cooldown = self.step >= self.cfg.total_steps * (1 - self.cfg.decay_frac)
        if (self.cooldown_batches is not None and in_cooldown
                and self.rng.random() < self.cfg.cooldown_clean_ratio):
            store, ids = self.cooldown_store, next(self.cooldown_batches)
        else:
            store, ids = self.store, next(self.batches)
        batch = collate(store, ids, self.tag_ids, self.bt_id, self.max_len)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _loss(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            logits = self.model(batch["src"], batch["tgt_in"])
        # Loss in fp32: fp16 softmax hits the collapse cliff at tiny margins.
        logits = logits.float().flatten(0, 1)
        target = batch["tgt_out"].flatten()
        mask = target.ne(self.model.cfg.pad_id)
        logits, target = logits[mask], target[mask]
        if self.cfg.stable_ce:
            per_tok = stable_cross_entropy(logits, target, self.cfg.label_smoothing)
        else:
            per_tok = F.cross_entropy(logits, target, reduction="none",
                                      label_smoothing=self.cfg.label_smoothing)
        weights = batch["weight"][:, None].expand_as(batch["tgt_out"]).flatten()[mask]
        return (per_tok * weights).sum(), weights.sum()

    def train_step(self) -> float:
        self.model.train()
        for group in self.optim.param_groups:
            group["lr"] = wsd_lr(self.step, self.cfg)

        losses = []
        for _ in range(self.cfg.accum):
            batch = self._next_batch()
            loss_sum, n_tok = self._loss(batch)
            losses.append((loss_sum.item(), n_tok.item()))
            self.scaler.scale(loss_sum).backward()

        total_tokens = sum(n for _, n in losses)
        # Gradients were summed over tokens; turn them into a per-token mean.
        self.scaler.unscale_(self.optim)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(total_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip)
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad(set_to_none=True)

        self.step += 1
        self.last_loss = sum(s for s, _ in losses) / total_tokens
        self.last_grad_norm = float(grad_norm)
        return self.last_loss

    def fit(self) -> None:
        t0 = time.time()
        window = []
        while self.step < self.cfg.total_steps:
            window.append(self.train_step())
            if self.step % self.cfg.log_every == 0:
                # Mean over the interval: single-batch losses swing with source and length.
                loss, window = sum(window) / len(window), []
                self._log({"step": self.step, "loss": round(loss, 4),
                           "lr": wsd_lr(self.step, self.cfg),
                           "grad_norm": round(self.last_grad_norm, 3),
                           "elapsed_s": round(time.time() - t0, 1)})
            if self.evaluate_fn and self.step % self.cfg.eval_every == 0:
                metrics = self.evaluate_fn(self.model)
                self._log({"step": self.step, **metrics})
                if metrics.get("chrf", -math.inf) > self.best:
                    self.best = metrics["chrf"]
                    self.save_checkpoint(self.run_dir / "best.pt")
            if self.step % self.cfg.save_every == 0 or self.step == self.cfg.total_steps:
                self.save_checkpoint(self.run_dir / "last.pt")

    def _log(self, record: dict) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self.logger:
            self.logger(record)
        else:
            print(record, flush=True)

    def save_checkpoint(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "best": self.best,
            "model_cfg": asdict(self.model.cfg),
            "train_cfg": asdict(self.cfg),
            "rng": {"py": self.rng.getstate(), "np": np.random.get_state(),
                    "torch": torch.get_rng_state()},
        }, tmp)
        tmp.replace(path)  # never leave a half-written checkpoint behind
        if self.on_save:
            self.on_save()

    def load_checkpoint(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optim"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step, self.best = ckpt["step"], ckpt["best"]
        self.rng.setstate(ckpt["rng"]["py"])
        np.random.set_state(ckpt["rng"]["np"])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())  # map_location moved it to the GPU
        # A fresh data order, otherwise a resumed run replays the start of the epoch.
        self._make_batchers(self.cfg.seed + self.step)
        return True
