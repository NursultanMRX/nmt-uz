"""LaTeX tables for the technical report, generated from results/ so the paper cannot drift.

    uv run python scripts/paper_tables.py         # writes paper/tables/*.tex
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import PUBLISHED

SYSTEMS = [  # results file stem, row label, parameters
    ("ours", "nmt-uz-76m (ours)", "76.5M"),
    ("nllb", "NLLB-200-distilled-600M", "600M"),
    ("nllb-1.3b", "NLLB-200-distilled-1.3B", "1.3B"),
]
METRICS = [("bleu", "{:.2f}"), ("chrf", "{:.2f}"), ("comet", "{:.3f}")]
DIRECTIONS = [("en2uz", r"en$\to$uz"), ("uz2en", r"uz$\to$en")]
ONNX = [("ours", "PyTorch fp32"), ("onnx-fp32", "ONNX fp32"), ("onnx-fp16", "ONNX fp16"),
        ("onnx-int8", "ONNX int8"), ("onnx-int4", "ONNX int4")]
ONNX_MB = {"ours": 292, "onnx-fp32": 372, "onnx-fp16": 186, "onnx-int8": 94, "onnx-int4": 59}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tabular(cols: str, head: list[str], rows: list[str]) -> str:
    return "\n".join([rf"\begin{{tabular}}{{{cols}}}", r"\toprule", *head, r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}", ""])


def _scores(results: Path, subset: str) -> dict[str, dict]:
    return {stem: _load(results / "benchmark" / f"{stem}_devtest.json")[subset]["compat"]
            for stem, *_ in SYSTEMS}


METRIC_HEAD = [
    r" & & \multicolumn{3}{c}{en$\to$uz} & \multicolumn{3}{c}{uz$\to$en} \\",
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
    r"System & Params & BLEU & chrF & COMET & BLEU & chrF & COMET \\",
]


def main_table(results: Path) -> str:
    """FLORES+ devtest, all 1012 sentences; the best score in each column in bold."""
    scores = _scores(results, "full")
    best = {(d, m): max(s[d][m] for s in scores.values()) for d, _ in DIRECTIONS for m, _ in METRICS}
    rows = []
    for stem, label, params in SYSTEMS:
        cells = [label, params]
        for d, _ in DIRECTIONS:
            for m, fmt in METRICS:
                v = scores[stem][d][m]
                text = fmt.format(v)
                cells.append(rf"\textbf{{{text}}}" if v == best[d, m] else text)
        rows.append(" & ".join(cells) + r" \\")
    return _tabular("lrrrrrrr", METRIC_HEAD, rows)


def _delta(entry: dict, digits: int) -> str:
    lo, hi = entry["ci"]
    p = "" if entry["p"] < 0.001 else f", p = {entry['p']:.3f}"
    return f"${entry['delta']:+.{digits}f}$ $[{lo:+.{digits}f}, {hi:+.{digits}f}]${p}"


def significance_table(results: Path) -> str:
    """Paired bootstrap: our score minus the baseline's, with its 95% interval."""
    sig = _load(results / "significance" / "devtest.json")
    head = [r"vs. & Direction & $\Delta$BLEU & $\Delta$chrF & $\Delta$COMET \\"]
    rows = []
    for stem, label, _ in SYSTEMS[1:]:
        for d, name in DIRECTIONS:
            e = sig["deltas"][stem][d]
            rows.append(" & ".join([label.replace("NLLB-200-distilled-", "NLLB "), name,
                                    _delta(e["bleu"], 2), _delta(e["chrf"], 2),
                                    _delta(e["comet"], 3)]) + r" \\")
    return _tabular("llrrr", head, rows)


def reported_table(results: Path) -> str:
    """First 200 devtest sentences: our runs next to the numbers other systems report."""
    scores = _scores(results, "first200")
    rows = []
    for stem, label, params in SYSTEMS:
        cells = [label, params] + [fmt.format(scores[stem][d][m])
                                   for d, _ in DIRECTIONS for m, fmt in METRICS]
        rows.append(" & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    for name, (params, en2uz, uz2en) in PUBLISHED.items():
        values = [f"{en2uz[0]:.2f}", f"{en2uz[1]:.2f}", f"{en2uz[2]:.3f}",
                  f"{uz2en[0]:.2f}", f"{uz2en[1]:.2f}", f"{uz2en[2]:.3f}"]
        rows.append(" & ".join([f"{name}, reported", params, *values]) + r" \\")
    return _tabular("lrrrrrrr", METRIC_HEAD, rows)


def latency_table(results: Path) -> str:
    """Milliseconds per sentence, en->uz / uz->en, first 100 devtest sentences."""
    head = [r"Hardware & System & Beam & Single, median & Single, p95 & Batch of 100 \\"]
    rows = []
    for device, hw in [("cpu", "CPU, 4 cores"), ("cuda", "T4")]:
        for beam in ("beam4", "beam1"):
            if device == "cuda" and beam == "beam1":
                continue
            for stem, label in [("ours", "nmt-uz-76m"), ("onnx-int8", "nmt-uz-76m, ONNX int8"),
                                ("nllb", "NLLB 600M")]:
                path = results / "latency" / f"{stem}_{device}.json"
                if not path.exists():
                    continue
                r = _load(path)[beam]
                pairs = [f"{r['en2uz'][k]:.0f} / {r['uz2en'][k]:.0f}"
                         for k in ("p50_ms", "p95_ms", "batch_ms_per_sentence")]
                rows.append(" & ".join([hw, label, beam[-1], *pairs]) + r" \\")
    return _tabular("llrrrr", head, rows)


def onnx_table(results: Path) -> str:
    """Size, quality and CPU latency of every export."""
    head = [r"Model & Size (MB) & en$\to$uz BLEU / chrF & uz$\to$en BLEU / chrF"
            + r" & Beam 4, ms & Beam 1, ms \\"]
    rows = []
    for stem, label in ONNX:
        q = _load(results / "benchmark" / f"{stem}_devtest.json")["full"]["compat"]
        lat = _load(results / "latency" / f"{stem}_cpu.json")
        cells = [label, str(ONNX_MB[stem])]
        cells += [f"{q[d]['bleu']:.2f} / {q[d]['chrf']:.2f}" for d, _ in DIRECTIONS]
        cells += [f"{lat[b]['en2uz']['p50_ms']:.0f} / {lat[b]['uz2en']['p50_ms']:.0f}"
                  for b in ("beam4", "beam1")]
        rows.append(" & ".join(cells) + r" \\")
    return _tabular("lrrrrr", head, rows)


def _n(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def data_table(results: Path) -> str:
    """Per-source pairs: raw, after the rule filters, after deduplication."""
    stats = _load(results / "data_stats.json")
    head = [r"Source & Raw & Filtered & Deduplicated \\"]
    rows = []
    total = [0, 0, 0]
    for name, s in sorted(stats["sources"].items(), key=lambda kv: -kv[1]["raw"]):
        dedup = s.get("dedup", {}).get("kept", 0)
        for i, v in enumerate((s["raw"], s["kept"], dedup)):
            total[i] += v
        rows.append(" & ".join([name.replace("_", r"\_"), _n(s["raw"]), _n(s["kept"]),
                                _n(dedup)]) + r" \\")
    rows += [r"\midrule", " & ".join(["Total", *map(_n, total)]) + r" \\", r"\midrule",
             rf"Training set (after length filter) & & & {_n(stats['stores']['train']['pairs'])} \\",
             rf"Clean subset & & & {_n(stats['stores']['clean']['pairs'])} \\"]
    return _tabular("lrrr", head, rows)


TABLES = {"main": main_table, "significance": significance_table, "reported": reported_table,
          "latency": latency_table, "onnx": onnx_table, "data": data_table}


def write_all(results: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (out / f"{name}.tex").write_text(fn(results), encoding="utf-8")


if __name__ == "__main__":
    write_all(ROOT / "results", ROOT / "paper" / "tables")
    print("wrote paper/tables/*.tex")
