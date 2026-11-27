"""Speed against quality for every precision of ours and NLLB-600M, one panel per direction.

    uv run python scripts/speed_quality.py            # writes docs/img/speed_quality.{png,svg}
    uv run python scripts/speed_quality.py --paper    # writes paper/figures/speed_quality.pdf

x is the fastest batch size of each variant (seconds for the whole devtest, both directions,
beam 4, one T4); y is chrF of the batch-32 translations. Marker area follows the weight size.
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#dcdfe5"
STYLE = {"ours": ("nmt-uz-76m", "#2a78d6"), "nllb": ("NLLB-200-distilled-600M", "#eb6834")}
PANELS = [("en2uz", "English → Uzbek"), ("uz2en", "Uzbek → English")]


def load_points(results: Path) -> list[dict]:
    points = []
    for path in sorted(results.glob("*.json")):
        r = json.loads(path.read_text())
        best = min(s for s in r["seconds"].values() if s is not None)
        points.append({"system": r["system"], "precision": r["precision"],
                       "seconds": best, "mb": r["weights_mb"],
                       **{d: r["quality"][d]["chrf"] for d, _ in PANELS}})
    return points


def draw(points: list[dict], out: Path, paper: bool = False) -> None:
    """paper=True: a plain white, untitled PDF for the report; the caption carries the title."""
    mpl.rcParams.update({"font.family": ["DejaVu Sans"] if paper else ["Segoe UI", "DejaVu Sans"],
                         "svg.fonttype": "path", "pdf.fonttype": 42})
    # a smaller canvas for the paper: the text stays readable once scaled to the page width
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4) if paper else (14, 6.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92 if paper else 0.76, bottom=0.14,
                        wspace=0.2)

    if not paper:
        bg = fig.add_axes((0, 0, 1, 1), zorder=-1)
        bg.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", extent=(0, 1, 0, 1),
                  cmap=LinearSegmentedColormap.from_list("bg", ["#ffffff", "#eef3fa"]))
        bg.axis("off")
        fig.text(0.035, 0.95, "Speed and quality, every precision", fontsize=24, weight="bold",
                 color=INK, va="top")
        fig.text(0.035, 0.875, "Whole FLORES+ devtest, both directions, beam 4, one T4 · "
                 "fastest batch size of each variant · marker area = weight size", fontsize=13,
                 color=INK2, va="top")

    for ax, (direction, title) in zip(axes, PANELS):
        for system, (name, color) in STYLE.items():
            pts = [p for p in points if p["system"] == system]
            ax.scatter([p["seconds"] for p in pts], [p[direction] for p in pts],
                       s=[40 + p["mb"] / 6 for p in pts], color=color, alpha=0.85,
                       edgecolor="white", linewidth=2, zorder=3, label=name)
            # alternate above / below along x, so neighbouring labels never overlap; NLLB starts
            # below because its two fastest points are close in x and the faster one is lower
            for i, p in enumerate(sorted(pts, key=lambda p: p["seconds"])):
                up = (i % 2 == 0) == (system == "ours")
                gap = (40 + p["mb"] / 6) ** 0.5 / 2 + 5  # marker radius in points, plus a margin
                ax.annotate(f"{p['precision']}\n{p['mb']:.0f} MB", (p["seconds"], p[direction]),
                            xytext=(0, gap if up else -gap), textcoords="offset points",
                            ha="center", va="bottom" if up else "top", fontsize=10,
                            linespacing=1.1, color=INK2,
                            weight="bold" if system == "ours" else "normal")
        ax.set_xscale("log")
        ax.set_xlim(15, 400)
        ax.set_xticks([20, 50, 100, 200])
        ax.set_xticklabels(["20 s", "50 s", "100 s", "200 s"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))  # room for the labels
        ax.set_title(title, fontsize=15, color=INK, loc="left", pad=10)
        ax.set_xlabel("seconds (log scale) · faster is left", fontsize=12, color=MUTED)
        ax.grid(color=GRID, lw=1, zorder=0)
        ax.tick_params(length=0, labelsize=11.5, labelcolor=MUTED)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b9bcc4")
        ax.set_facecolor("none")
    axes[0].set_ylabel("chrF · higher is better", fontsize=12, color=MUTED)
    handles = [Line2D([], [], ls="", marker="o", ms=10, color=c, label=n)
               for n, c in STYLE.values()]
    axes[0].legend(handles=handles, loc="center right", frameon=False, fontsize=11.5)

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf",) if paper else ("png", "svg"):
        fig.savefig(out / f"speed_quality.{ext}", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    paper = "--paper" in sys.argv
    out = ROOT / "paper" / "figures" if paper else ROOT / "docs" / "img"
    draw(load_points(ROOT / "results" / "efficiency"), out, paper)
    print("wrote", out / ("speed_quality.pdf" if paper else "speed_quality.png"))
