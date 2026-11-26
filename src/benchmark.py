"""FLORES+ benchmark: our model and NLLB-200 through the same script and the same scoring.

    python -m src.benchmark --system ours --model release/ --split devtest
    python -m src.benchmark --system nllb
    python -m src.benchmark --system nllb-1.3b
    python -m src.benchmark --table results/benchmark

Two tracks. `compat` scores against the raw FLORES+ references (1995 Latin alphabet), which is
how published en<->uz numbers are reported; our Uzbek output is converted back for it.
`native` scores Uzbek in the 2026 alphabet. NLLB needs `transformers`; COMET (wmt22-comet-da)
needs `unbabel-comet`, which pins older libraries, so on Modal it runs in its own image.
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import sacrebleu
import torch

from src.data_pipeline.alphabet import to_old_latin
from src.data_pipeline.cleaning import normalize_uz

NLLB_CODES = {"en": "eng_Latn", "uz": "uzn_Latn"}
SUBSETS = {"full": None, "first200": 200}

# https://huggingface.co/OvozifyLabs/nllb-200-uz-en-v1#benchmark-results (200 FLORES+ samples)
# name: (params, en2uz (bleu, chrf, comet), uz2en (...))
PUBLISHED = {
    "NLLB-200-uz-en-v1 (fine-tuned)": ("600M", (20.22, 59.3, 0.906), (34.47, 62.19, 0.874)),
    "Tahrirchi Tilmoch (commercial)": ("n/a", (19.83, 58.01, 0.91), (33.76, 61.97, 0.876)),
    "NLLB-200-distilled-600M": ("600M", (13.07, 51.73, 0.881), (30.28, 58.28, 0.856)),
}
# system: (display name, params, default model)
SYSTEMS = {
    "ours": ("nmt-uz-76m (ours)", "76.5M", "release/"),
    **{f"onnx-{p}": (f"nmt-uz-76m, ONNX {p}", "76.5M", "release/")
       for p in ("fp32", "fp16", "int8", "int4")},
    "nllb": ("NLLB-200-distilled-600M (our run)", "600M", "facebook/nllb-200-distilled-600M"),
    "nllb-1.3b": ("NLLB-200-distilled-1.3B (our run)", "1.3B", "facebook/nllb-200-distilled-1.3B"),
}

_OKINA = re.compile(r"[OoGg]([ʻ‘'’`ʼ])")


def to_reference_script(hyps: list[str], refs: list[str]) -> list[str]:
    """2026-alphabet output -> 1995 Latin, with the oʻ/gʻ mark the references use."""
    marks = Counter(m.group(1) for r in refs for m in _OKINA.finditer(r))
    mark = marks.most_common(1)[0][0] if marks else "ʻ"
    return [to_old_latin(h).replace("ʻ", mark) for h in hyps]


def text_metrics(hyps: list[str], refs: list[str]) -> dict:
    return {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        "chrfpp": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
    }


class NLLB:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model

    @classmethod
    def from_pretrained(cls, name: str, device: str) -> "NLLB":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype)
        return cls(AutoTokenizer.from_pretrained(name), model.to(device).eval())

    @torch.no_grad()
    def translate(self, texts: list[str], target: str, beam: int = 5, batch_size: int = 32,
                  max_len: int = 256) -> list[str]:
        self.tok.src_lang = NLLB_CODES["uz" if target == "en" else "en"]
        bos = self.tok.convert_tokens_to_ids(NLLB_CODES[target])
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            enc = self.tok([texts[i] for i in chunk], return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            gen = self.model.generate(**enc, forced_bos_token_id=bos, num_beams=beam,
                                      max_new_tokens=max_len, max_length=None)
            for i, text in zip(chunk, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = text
        return out


def comet_scorer(batch_size: int = 64) -> Callable[[list, list, list], list[float]]:
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    def score(src, mt, ref):
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, mt, ref)]
        gpus = 1 if torch.cuda.is_available() else 0
        return model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False).scores

    return score


def writes_2026(system: str) -> bool:
    return system == "ours" or system.startswith("onnx-")


def score_all(system: str, rows: list[dict], hyps: dict[str, list[str]],
              subsets: dict[str, int | None] = SUBSETS, comet=None) -> dict:
    en, uz = [r["en"] for r in rows], [r["uz"] for r in rows]
    uz_new = [normalize_uz(t) for t in uz]
    if writes_2026(system):
        en2uz_old, en2uz_new = to_reference_script(hyps["en2uz"], uz), hyps["en2uz"]
    else:
        en2uz_old, en2uz_new = hyps["en2uz"], [normalize_uz(h) for h in hyps["en2uz"]]

    seg = {}
    if comet:
        seg["en2uz"] = comet(en, en2uz_old, uz)
        seg["uz2en"] = comet(uz, hyps["uz2en"], en)

    out = {}
    for name, n in subsets.items():
        n = n or len(rows)
        compat = {"en2uz": text_metrics(en2uz_old[:n], uz[:n]),
                  "uz2en": text_metrics(hyps["uz2en"][:n], en[:n])}
        for direction, scores in seg.items():
            compat[direction]["comet"] = round(sum(scores[:n]) / n, 4)
        out[name] = {"sentences": n, "compat": compat,
                     "native": {"en2uz": text_metrics(en2uz_new[:n], uz_new[:n])}}
    return out


def _row(name: str, params: str, en2uz: tuple, uz2en: tuple) -> str:
    cells = [params]
    for bleu, chrf, comet in (en2uz, uz2en):
        cells += [f"{bleu:.2f}", f"{chrf:.2f}", "-" if comet is None else f"{comet:.3f}"]
    return f"| {name} | " + " | ".join(cells) + " |"


def render_table(results_dir: str | Path, subset: str = "first200",
                 published: bool = True) -> str:
    lines = [
        "| System | Params | en→uz BLEU | chrF | COMET | uz→en BLEU | chrF | COMET |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = list(SYSTEMS)  # ours first, then the baselines; unknown systems last
    paths = Path(results_dir).glob("*.json")
    for path in sorted(paths, key=lambda p: (order.index(s) if (s := p.name.split("_")[0])
                                             in order else len(order), p.name)):
        res = json.loads(path.read_text())
        if subset not in res:
            continue
        c = res[subset]["compat"]
        name, params, _ = SYSTEMS.get(res["system"], (res["system"], "?", None))
        lines.append(_row(name, params, *((c[d]["bleu"], c[d]["chrf"], c[d].get("comet"))
                            for d in ("en2uz", "uz2en"))))
    for name, (params, en2uz, uz2en) in PUBLISHED.items() if published else ():
        lines.append(_row(f"{name}, reported", params, en2uz, uz2en))
    return "\n".join(lines) + "\n"


def translate_flores(system: str, model: str, rows: list[dict], beam: int) -> dict[str, list]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if system.startswith("nllb"):
        translator = NLLB.from_pretrained(model, device)
    elif system.startswith("onnx-"):
        from src.onnx_translator import OnnxTranslator

        translator = OnnxTranslator.from_pretrained(model, precision=system.removeprefix("onnx-"))
    else:
        from src.inference import Translator

        translator = Translator.from_pretrained(model, device=device)
    return {"en2uz": translator.translate([r["en"] for r in rows], "uz", beam=beam),
            "uz2en": translator.translate([r["uz"] for r in rows], "en", beam=beam)}


def run(system: str, model: str, split: str, out: Path, beam: int = 5, comet=None) -> dict:
    from src.data_pipeline.sources import load_flores

    rows = load_flores(split)
    stem = f"{system}_{split}"
    paths = {d: out / f"{stem}.{d}.txt" for d in ("en2uz", "uz2en")}
    if all(p.exists() for p in paths.values()):
        # Translations are the slow part; re-scoring reuses them.
        hyps = {d: p.read_text(encoding="utf-8").splitlines() for d, p in paths.items()}
        assert all(len(h) == len(rows) for h in hyps.values()), "stale hypotheses, delete them"
    else:
        hyps = translate_flores(system, model, rows, beam)
        out.mkdir(parents=True, exist_ok=True)
        for d, lines in hyps.items():
            paths[d].write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"system": system, "model": model, "split": split, "beam": beam,
              **score_all(system, rows, hyps, comet=comet)}
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(SYSTEMS))
    ap.add_argument("--model", help="defaults to the system's usual checkpoint")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/benchmark"))
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--table", type=Path, help="only render the table from saved results")
    args = ap.parse_args(argv)

    if args.table:
        path = args.table / "README.md"
        path.write_text(
            "# Benchmark\n\n"
            "Beam 5. Uzbek scored against the original FLORES+ references (1995 alphabet);\n"
            "chrF is sacrebleu's default (not chrF++), COMET is wmt22-comet-da.\n\n"
            "## FLORES+ devtest, first 200 sentences\n\n"
            "Same size as the published results in the OvozifyLabs model card "
            "(https://huggingface.co/OvozifyLabs/nllb-200-uz-en-v1).\n\n"
            f"{render_table(args.table, 'first200')}\n"
            "## FLORES+ devtest, all 1012 sentences\n\n"
            f"{render_table(args.table, 'full', published=False)}",
            encoding="utf-8")
        print(f"wrote {path}")
        return
    comet = None if args.no_comet else comet_scorer()
    model = args.model or SYSTEMS[args.system][2]
    result = run(args.system, model, args.split, args.out, args.beam, comet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
