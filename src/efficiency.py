"""Speed and quality of each weight precision across batch sizes, ours against NLLB-200.

    modal run modal_app.py::efficiency                 # every variant on its own T4
    python -m src.efficiency --table results/efficiency

Each time is wall-clock seconds to translate the full FLORES+ devtest in both directions,
beam 4, after one warm-up batch; loading is not timed. int8 and int4 are bitsandbytes
(LLM.int8 and NF4) and need CUDA.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

PRECISIONS = ("fp32", "fp16", "int8", "int4")
BATCH_SIZES = (1, 8, 32, 64, 128, 256)
NAMES = {"ours": "**nmt-uz-76m**", "nllb": "NLLB-200-distilled-600M"}


def weight_bytes(model: nn.Module) -> int:
    """Bytes of the parameters as stored, counting tied weights once."""
    unique = {p.data_ptr(): p for p in model.parameters()}.values()
    return sum(p.numel() * p.element_size() for p in unique)


def _swap_linears(module: nn.Module, make: Callable[[nn.Linear], nn.Module]) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name != "lm_head":  # tied to the embedding
            setattr(module, name, make(child))
        else:
            _swap_linears(child, make)


def cast(model: nn.Module, precision: str) -> nn.Module:
    """fp32 / fp16 on any device; int8 / int4 swap the linear layers for bitsandbytes ones,
    which quantize when the model moves to CUDA."""
    if precision == "fp32":
        return model.float()
    if precision == "fp16":
        return model.half()
    if precision not in ("int8", "int4"):
        raise ValueError(f"unknown precision {precision!r}")
    import bitsandbytes as bnb

    def make(linear: nn.Linear) -> nn.Module:
        args = (linear.in_features, linear.out_features, linear.bias is not None)
        if precision == "int8":
            new = bnb.nn.Linear8bitLt(*args, has_fp16_weights=False, threshold=6.0)
            new.weight = bnb.nn.Int8Params(linear.weight.data.half(), requires_grad=False,
                                           has_fp16_weights=False)
        else:
            new = bnb.nn.Linear4bit(*args, compute_dtype=torch.float16, quant_type="nf4")
            new.weight = bnb.nn.Params4bit(linear.weight.data.half(), requires_grad=False,
                                           quant_type="nf4")
        if linear.bias is not None:
            new.bias = nn.Parameter(linear.bias.data.half(), requires_grad=False)
        return new

    model = model.half()
    _swap_linears(model, make)
    return model


def sweep(translate: Callable, rows: list[dict], batch_sizes=BATCH_SIZES, beam: int = 4,
          quality_batch: int = 32, warmup: bool = True) -> dict:
    """translate(texts, target, beam, batch_size) -> list[str]. A batch size that runs out of
    GPU memory is recorded as None, and the larger ones are skipped."""
    en, uz = [r["en"] for r in rows], [r["uz"] for r in rows]
    if warmup:
        translate(en[:8], "uz", beam, 8)
    seconds, hyps = {}, {}
    for i, bs in enumerate(batch_sizes):
        start = time.perf_counter()
        try:
            en2uz = translate(en, "uz", beam, bs)
            uz2en = translate(uz, "en", beam, bs)
        except torch.OutOfMemoryError:
            seconds.update(dict.fromkeys(batch_sizes[i:]))
            torch.cuda.empty_cache()
            break
        seconds[bs] = round(time.perf_counter() - start, 1)
        if bs == quality_batch:
            hyps = {"en2uz": en2uz, "uz2en": uz2en}
    return {"seconds": seconds, "hyps": hyps}


def load(system: str, model: str, precision: str) -> tuple[Callable, nn.Module]:
    if system == "ours":
        from src.inference import Translator

        tr = Translator.from_pretrained(model, device="cpu")
        tr.model = cast(tr.model, precision).cuda()
        return (lambda texts, target, beam, bs: tr.translate(
            texts, target=target, beam=beam, batch_size=bs, autocast=False)), tr.model

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

    from src.benchmark import NLLB

    kwargs = {"torch_dtype": torch.float32 if precision == "fp32" else torch.float16}
    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    net = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="cuda", **kwargs).eval()
    nllb = NLLB(AutoTokenizer.from_pretrained(model), net)
    return (lambda texts, target, beam, bs: nllb.translate(
        texts, target, beam=beam, batch_size=bs)), net


def run(system: str, model: str, precision: str, batch_sizes=BATCH_SIZES) -> dict:
    from src.benchmark import score_all
    from src.data_pipeline.sources import load_flores

    rows = load_flores("devtest")
    translate, net = load(system, model, precision)
    out = sweep(translate, rows, list(batch_sizes))
    compat = score_all(system, rows, out["hyps"], subsets={"full": None})["full"]["compat"]
    return {"system": system, "precision": precision, "gpu": torch.cuda.get_device_name(),
            "weights_mb": round(weight_bytes(net) / 2**20, 1),
            "seconds": {str(bs): s for bs, s in out["seconds"].items()},
            "quality": {d: {k: compat[d][k] for k in ("bleu", "chrf")} for d in compat}}


def _seconds(s: float | None) -> str:
    if s is None:
        return "OOM"
    return f"{s:.0f} s" if s >= 100 else f"{s:.1f} s"


def render_table(results: list[dict]) -> str:
    batches = sorted({b for r in results for b in r["seconds"]}, key=int)
    lines = ["| Model | Precision | Weights | " + " | ".join(f"batch {b}" for b in batches)
             + " | en→uz BLEU / chrF | uz→en BLEU / chrF |",
             "|---|---|---:|" + "---:|" * len(batches) + "---:|---:|"]
    order = {s: i for i, s in enumerate(("nllb", "ours"))}
    for r in sorted(results, key=lambda r: (order.get(r["system"], 9),
                                            PRECISIONS.index(r["precision"]))):
        times = [_seconds(r["seconds"][b]) if b in r["seconds"] else "-" for b in batches]
        q = [f"{r['quality'][d]['bleu']:.1f} / {r['quality'][d]['chrf']:.1f}"
             for d in ("en2uz", "uz2en")]
        lines.append(f"| {NAMES.get(r['system'], r['system'])} | {r['precision']} | "
                     f"{r['weights_mb']:.0f} MB | " + " | ".join(times + q) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True, help="directory of result json files")
    args = ap.parse_args(argv)
    results = [json.loads(p.read_text()) for p in sorted(args.table.glob("*.json"))]
    print(render_table(results))


if __name__ == "__main__":
    main()
