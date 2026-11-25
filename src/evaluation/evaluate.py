"""Batched beam-search translation and BLEU / chrF++ scoring on FLORES+."""

import json
from pathlib import Path

import sacrebleu
import torch

from src.data_pipeline.cleaning import normalize_en, normalize_uz
from src.data_pipeline.tokenizer import BOS, EOS, PAD, Tokenizer
from src.models.transformer import Seq2SeqTransformer


def load_eval(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def translate(model: Seq2SeqTransformer, tok: Tokenizer, sentences: list[str], tgt_lang: str,
              beam: int = 4, batch_size: int = 64, max_len: int | None = None,
              autocast: bool = True, max_ratio: float | None = None) -> list[str]:
    """max_ratio caps each output at max_ratio x its batch's source length + 10 tokens, so a
    decoder stuck in a loop stops early. None (the default, used by the benchmarks) does not."""
    src_lang_norm = normalize_uz if tgt_lang == "en" else normalize_en
    device = next(model.parameters()).device
    max_len = max_len or model.cfg.max_len
    ids = [tok.source_ids(s[: max_len - 2], tgt_lang)
           for s in tok.encode([src_lang_norm(s) for s in sentences])]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = [""] * len(ids)

    was_training = model.training
    model.eval()
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        width = max(len(ids[i]) for i in chunk)
        src = torch.full((len(chunk), width), PAD, dtype=torch.long)
        for row, i in enumerate(chunk):
            src[row, : len(ids[i])] = torch.tensor(ids[i])
        # autocast=False keeps the weights' own dtype; the efficiency table measures fp32 that way
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=autocast and device.type == "cuda"):
            limit = max_len if max_ratio is None else min(max_len, int(max_ratio * width) + 10)
            hyp = model.generate(src.to(device), BOS, EOS, max_len=limit, beam=beam)
        for row, i in enumerate(chunk):
            out[i] = tok.decode(hyp[row].tolist())
    model.train(was_training)
    return out


def score(hyps: list[str], refs: list[str]) -> dict:
    return {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),  # chrF++
    }


def evaluate(model, tok, rows: list[dict], beam: int = 4, max_sents: int | None = None,
             return_hyps: bool = False) -> dict:
    """Scores both directions. `chrf` is the mean of the two and picks the best checkpoint."""
    rows = rows[:max_sents] if max_sents else rows
    metrics, hyps = {}, {}
    for name, src, tgt in (("en2uz", "en", "uz"), ("uz2en", "uz", "en")):
        hyps[name] = translate(model, tok, [r[src] for r in rows], tgt, beam=beam)
        for k, v in score(hyps[name], [r[tgt] for r in rows]).items():
            metrics[f"{name}_{k}"] = v
    metrics["chrf"] = round((metrics["en2uz_chrf"] + metrics["uz2en_chrf"]) / 2, 2)
    return (metrics, hyps) if return_hyps else metrics
