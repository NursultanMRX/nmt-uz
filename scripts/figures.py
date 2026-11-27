"""README figures from the files in results/, in light and dark variants.

    uv run python scripts/figures.py            # writes docs/img/*.svg
    uv run python scripts/figures.py --paper    # writes paper/figures/*.pdf, light, untitled
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834",
              "other": "#b9b8b0", "band": "#f0efec",
              "ramp": ["#104281", "#256abf", "#3987e5", "#86b6ef"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "blue": "#3987e5", "orange": "#d95926",
             "other": "#5c5b57", "band": "#242423",
             "ramp": ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]},
}


def _style(t: dict, paper: bool = False) -> None:
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans"] if paper else ["Segoe UI", "DejaVu Sans"],
        "font.size": 10.5,
        "pdf.fonttype": 42,  # TrueType in the PDF, not Type 3
        "svg.fonttype": "path",  # glyphs as paths: renders the same on GitHub and HF
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _title(fig, title: str, subtitle: str, t: dict) -> None:
    fig.text(0.02, 0.97, title, fontsize=13, weight="semibold", va="top", color=t["ink"])
    fig.text(0.02, 0.905, subtitle, fontsize=10, va="top", color=t["ink2"])


def _save(fig, out: Path, name: str, mode: str, paper: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if paper:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    else:
        suffix = "" if mode == "light" else "-dark"
        fig.savefig(out / f"{name}{suffix}.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def load_metrics(path: Path) -> tuple[dict, dict]:
    train, evals = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        (evals if "chrf" in r else train)[r["step"]] = r  # a resumed run repeats steps
    return train, evals


def training_curve(metrics: Path, mode: str, out: Path = OUT, paper: bool = False) -> None:
    t = THEMES[mode]
    _, evals = load_metrics(metrics)
    steps = sorted(evals)
    fig, ax = plt.subplots(figsize=(6.4, 3.2) if paper else (8, 4.2))
    if not paper:
        fig.subplots_adjust(top=0.78)
        _title(fig, "Translation quality during training",
               "chrF++ on FLORES+ dev (500 sentences, beam 4), evaluated every 2k steps", t)

    for lo, hi, label in ((0, 4000, "warmup"), (45000, 50000, "decay + clean data")):
        ax.axvspan(lo, hi, color=t["band"], lw=0, zorder=0)
        ax.text((lo + hi) / 2, 1.01, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color=t["muted"])

    for key, name, color in (("en2uz_chrf", "en→uz", t["blue"]),
                             ("uz2en_chrf", "uz→en", t["orange"])):
        ys = [evals[s][key] for s in steps]
        ax.plot(steps, ys, color=color, lw=2, marker="o", ms=4.5,
                markeredgecolor=t["surface"], markeredgewidth=1.2, label=name, zorder=3)
        ax.annotate(f"{name}  {ys[-1]:.1f}", (steps[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=t["ink"])

    ax.set_xlim(0, 50000)
    ax.set_ylim(10, max(max(e["uz2en_chrf"] for e in evals.values()) + 4, 55))
    ax.set_xticks(range(0, 50001, 10000), [f"{s // 1000}k" for s in range(0, 50001, 10000)])
    ax.set_xlabel("training step")
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", frameon=False, labelcolor=t["ink2"])
    _save(fig, out, "training_curve", mode, paper)


def data_funnel(stats: Path, mode: str, out: Path = OUT, paper: bool = False) -> None:
    t = THEMES[mode]
    s = json.loads(stats.read_text())
    src = s["sources"].values()
    stages = [
        ("raw pairs, 16 sources", sum(v["raw"] for v in src)),
        ("after rule filters", sum(v["kept"] for v in src)),
        ("after dedup and FLORES+ blocklist", s["stores"]["train"]["pairs"]),
        ("clean subset (cooldown)", s["stores"]["clean"]["pairs"]),
    ]
    nllb = s["sources"]["nllb"]["dedup"]["kept"] / s["stores"]["train"]["pairs"]

    fig, ax = plt.subplots(figsize=(6.4, 2.2) if paper else (8, 3.2))
    fig.subplots_adjust(left=0.3, top=0.95 if paper else 0.72)
    if not paper:
        _title(fig, "From 34M raw pairs to 26M training pairs",
               f"{nllb:.0%} of the final pairs are web-mined NLLB bitext", t)
    labels, values = zip(*stages)
    y = range(len(stages))[::-1]
    ax.barh(y, values, color=t["ramp"], height=0.62, edgecolor=t["surface"], linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v + 4e5, yi, f"{v / 1e6:.2f}M", va="center", fontsize=10, color=t["ink"])
    ax.set_yticks(list(y), labels, color=t["ink2"], fontsize=10)
    ax.set_xlim(0, max(values) * 1.12)
    ax.xaxis.set_visible(False)
    ax.spines[["bottom", "left"]].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, out, "data_funnel", mode, paper)


def hero() -> None:
    raw = OUT / "raw" / "hero.jpg"
    if not raw.exists():
        return
    img = plt.imread(raw)
    h, w = img.shape[:2]
    top = (h - w // 3) // 2
    plt.imsave(OUT / "hero.jpg", img[top:top + w // 3], pil_kwargs={"quality": 90})


def main(paper: bool = False, out: Path | None = None) -> None:
    out = out or (ROOT / "paper" / "figures" if paper else OUT)
    for mode, t in THEMES.items():
        if paper and mode != "light":
            continue
        _style(t, paper)
        metrics = ROOT / "results" / "base" / "metrics.jsonl"
        if metrics.exists():
            training_curve(metrics, mode, out, paper)
        data_funnel(ROOT / "results" / "data_stats.json", mode, out, paper)
    if not paper:
        hero()
    print(f"wrote {sorted(p.name for p in out.glob('*.pdf' if paper else '*.svg'))}")


if __name__ == "__main__":
    main(paper="--paper" in sys.argv)
