"""95% confidence intervals and paired-bootstrap significance for the FLORES+ comparison.

    modal run modal_app.py::significance     # writes results/significance/devtest.{json,md}

Paired bootstrap resampling (Koehn, 2004): every sample draws the same sentence indices for
every system and metric. BLEU and chrF are recomputed from summed sentence statistics, so each
sample is a true corpus score; COMET is the mean of its segment scores.
"""

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

SAMPLES, SEED = 1000, 12345
METRICS = {"bleu": BLEU, "chrf": CHRF}
DIRECTIONS = {"en2uz": "en→uz", "uz2en": "uz→en"}


def bootstrap_indices(n: int, samples: int = SAMPLES, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(samples, n))


def _corpus_scores(metric, hyps: list[str], refs: list[str], idx: np.ndarray):
    stats = np.array(metric._extract_corpus_statistics(hyps, [refs]), dtype=float)
    point = metric._compute_score_from_stats(stats.sum(0).tolist()).score
    samples = [metric._compute_score_from_stats(s.tolist()).score for s in stats[idx].sum(1)]
    return point, np.array(samples)


def _ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _check(name: str, values, n: int) -> None:
    if len(values) != n:
        raise ValueError(f"{name}: {len(values)} lines, expected {n}")


def compare(systems: dict[str, dict[str, list[str]]], refs: dict[str, list[str]],
            comet: dict[str, dict[str, list[float]]] | None = None, baseline: str = "ours",
            samples: int = SAMPLES, seed: int = SEED) -> dict:
    """Scores with 95% CIs per system, and baseline-minus-other deltas with CIs and p-values.

    p is the share of samples where the delta does not keep its sign (one-sided)."""
    scores, draws = {}, {}
    for direction, ref in refs.items():
        n = len(ref)
        idx = bootstrap_indices(n, samples, seed)
        for system, hyps in systems.items():
            _check(f"{system} {direction}", hyps[direction], n)
            for name, cls in METRICS.items():
                point, s = _corpus_scores(cls(), hyps[direction], ref, idx)
                scores.setdefault(system, {}).setdefault(direction, {})[name] = \
                    {"score": point, "ci": _ci(s)}
                draws[system, direction, name] = s
            if comet:
                seg = np.asarray(comet[system][direction], dtype=float)
                _check(f"{system} {direction} COMET", seg, n)
                s = seg[idx].mean(1)
                scores[system][direction]["comet"] = {"score": float(seg.mean()), "ci": _ci(s)}
                draws[system, direction, "comet"] = s

    deltas = {}
    for other in systems:
        if other == baseline:
            continue
        for direction in refs:
            for name in scores[baseline][direction]:
                delta = (scores[baseline][direction][name]["score"]
                         - scores[other][direction][name]["score"])
                d = draws[baseline, direction, name] - draws[other, direction, name]
                p = float(np.mean(d <= 0) if delta > 0 else np.mean(d >= 0))
                deltas.setdefault(other, {}).setdefault(direction, {})[name] = \
                    {"delta": delta, "ci": _ci(d), "p": p}
    return {"samples": samples, "seed": seed, "baseline": baseline,
            "systems": scores, "deltas": deltas}


def _fmt(d: dict, digits: int, samples: int) -> str:
    lo, hi = d["ci"]
    p = f"p<{1 / samples:g}" if d["p"] == 0 else f"p={d['p']:.3f}"
    return f"{d['delta']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] {p}"


def render_markdown(result: dict, names: dict[str, str] | None = None) -> str:
    names = names or {}
    metrics = [m for m in ("bleu", "chrf", "comet")
               if any(m in d for o in result["deltas"].values() for d in o.values())]
    header = {"bleu": "Δ BLEU", "chrf": "Δ chrF", "comet": "Δ COMET"}
    lines = [f"| {names.get(result['baseline'], result['baseline'])} vs | Direction | "
             + " | ".join(header[m] for m in metrics) + " |",
             "|---|---|" + "---:|" * len(metrics)]
    for other, by_dir in result["deltas"].items():
        for direction, d in by_dir.items():
            cells = [_fmt(d[m], 3 if m == "comet" else 2, result["samples"]) for m in metrics]
            lines.append(f"| {names.get(other, other)} | {DIRECTIONS[direction]} | "
                         + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
