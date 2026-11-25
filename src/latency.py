"""Translation latency on FLORES+ devtest sentences: one sentence at a time, and as a batch.

    python -m src.latency --system ours --model release/ --device cpu
    modal run modal_app.py::latency                  # ours and NLLB-600M, 4-core CPU and T4
"""

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch


def percentile(times: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q% of the data at or below it."""
    ordered = sorted(times)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def measure(translate: Callable, sentences: list[str], target: str, beam: int,
            warmup: int = 3) -> dict:
    for s in sentences[:warmup]:
        translate([s], target, beam)
    times = []
    for s in sentences:
        start = time.perf_counter()
        translate([s], target, beam)
        times.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    translate(sentences, target, beam)
    batch = (time.perf_counter() - start) * 1000 / len(sentences)
    return {"p50_ms": round(percentile(times, 50), 1), "p95_ms": round(percentile(times, 95), 1),
            "batch_ms_per_sentence": round(batch, 1), "sentences": len(sentences)}


def load_system(system: str, model: str, device: str, threads: int | None = None) -> Callable:
    if system.startswith("nllb"):
        from src.benchmark import NLLB

        nllb = NLLB.from_pretrained(model, device)
        return lambda texts, target, beam: nllb.translate(texts, target, beam=beam)
    if system.startswith("onnx-"):
        from src.onnx_translator import OnnxTranslator

        providers = ["CUDAExecutionProvider"] if device == "cuda" else None
        tr = OnnxTranslator.from_pretrained(model, system.removeprefix("onnx-"), providers,
                                            threads)
        return lambda texts, target, beam: tr.translate(texts, target=target, beam=beam)
    from src.inference import Translator

    tr = Translator.from_pretrained(model, device=device)
    return lambda texts, target, beam: tr.translate(texts, target=target, beam=beam)


def run(system: str, model: str, device: str, n: int = 100, threads: int | None = None) -> dict:
    from src.data_pipeline.sources import load_flores

    if threads:
        torch.set_num_threads(threads)
    rows = load_flores("devtest")[:n]
    translate = load_system(system, model, device, threads)
    hardware = torch.cuda.get_device_name() if device == "cuda" else f"CPU x{torch.get_num_threads()}"
    result = {"system": system, "device": device, "hardware": hardware}
    for beam in (4, 1):
        result[f"beam{beam}"] = {
            "en2uz": measure(translate, [r["en"] for r in rows], "uz", beam),
            "uz2en": measure(translate, [r["uz"] for r in rows], "en", beam),
        }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ours")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result = run(args.system, args.model, args.device, args.n, args.threads)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
