"""Benchmark chart: every system on FLORES+ devtest, grouped by metric and direction.

    uv run python scripts/benchmark_chart.py    # writes docs/img/benchmark.{png,svg}
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import PUBLISHED, SYSTEMS

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#dcdfe5"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]  # fixed per system, ours blue
GROUPS = [("en2uz", "chrf", "English → Uzbek\nchrF"), ("uz2en", "chrf", "Uzbek → English\nchrF"),
          ("en2uz", "comet", "English → Uzbek\nCOMET"), ("uz2en", "comet", "Uzbek → English\nCOMET")]


def load_rows(results: Path, subset: str = "first200") -> list[dict]:
    rows = []
    for system, (name, params, _) in SYSTEMS.items():
        path = results / f"{system}_devtest.json"
        if path.exists():
            c = json.loads(path.read_text())[subset]["compat"]
            rows.append({"name": name.replace(" (our run)", ""), "params": params,
                         "reported": False, "ours": system == "ours",
                         **{d: c[d] for d in ("en2uz", "uz2en")}})
    for name, (params, en2uz, uz2en) in PUBLISHED.items():
        if not name.startswith("NLLB-200-distilled"):  # we ran it ourselves
            rows.append({"name": name, "params": params, "reported": True, "ours": False,
                         **{d: dict(zip(("bleu", "chrf", "comet"), s))
                            for d, s in (("en2uz", en2uz), ("uz2en", uz2en))}})
    names = [r["name"] for r in rows]
    for r in rows:  # the color follows the system, not its position in the chart
        r["color"] = PALETTE[names.index(r["name"]) % len(PALETTE)]
    return rows


def _value(row: dict, direction: str, metric: str) -> float | None:
    v = row[direction].get(metric)
    return None if v is None else (v * 100 if metric == "comet" else v)


def _gradient_bar(ax, y, width, height, color):
    light = tuple(c + (1 - c) * 0.45 for c in to_rgb(color))
    cmap = LinearSegmentedColormap.from_list("bar", [light, color])
    bar = Rectangle((0, y - height / 2), width, height, fc="none", ec="none")
    ax.add_patch(bar)
    img = ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto", zorder=2,
                    extent=(0, width, y - height / 2, y + height / 2))
    img.set_clip_path(bar)


def draw(rows: list[dict], out: Path) -> None:
    mpl.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "svg.fonttype": "path"})
    n, bar, gap = len(rows), 0.8, 1.6
    group_h = n * bar + gap
    fig = plt.figure(figsize=(14, 2.9 + 0.33 * group_h * len(GROUPS)))
    h = fig.get_figheight()
    ax = fig.add_axes((0.2, 0.75 / h, 0.72, 1 - 3.05 / h))

    bg = fig.add_axes((0, 0, 1, 1), zorder=-1)
    bg.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", extent=(0, 1, 0, 1),
              cmap=LinearSegmentedColormap.from_list("bg", ["#ffffff", "#eef3fa"]))
    bg.axis("off")

    fig.text(0.035, 1 - 0.35 / h, "Translation benchmarks", fontsize=26, weight="bold",
             color=INK, va="top")
    fig.text(0.035, 1 - 0.95 / h, "FLORES+ devtest, first 200 sentences · higher is better · "
             "COMET shown × 100", fontsize=14, color=INK2, va="top")
    renderer = fig.canvas.get_renderer()
    x, line = 0.035, 0
    for i, r in enumerate(rows):
        size = "" if r["params"] == "n/a" else f" · {r['params']}"
        label = r["name"] + size + ("*" if r["reported"] else "")
        t = fig.text(0, 0, label, fontsize=12.5, color=INK, va="center",
                     weight="bold" if r["ours"] else "normal")
        width = fig.transFigure.inverted().transform(
            (t.get_window_extent(renderer).width, 0))[0] - fig.transFigure.inverted().transform(
            (0, 0))[0]
        if x + 0.016 + width > 0.97:  # wrap onto the next legend line
            x, line = 0.035, line + 1
        y = 1 - (1.47 + 0.38 * line) / h
        t.set_position((x + 0.016, y))
        fig.patches.append(Rectangle((x, y - 0.08 / h), 0.011, 0.16 / h, color=r["color"],
                                     transform=fig.transFigure, figure=fig))
        x += 0.016 + width + 0.022

    for g, (direction, metric, title) in enumerate(GROUPS):
        base = (len(GROUPS) - 1 - g) * group_h
        ax.text(-2, base + n * bar / 2 - bar / 2, title, fontsize=15, ha="right",
                va="center", color=INK, linespacing=1.35)
        for i, r in enumerate(rows):
            v = _value(r, direction, metric)
            if v is None:
                continue
            y = base + i * bar  # first row (ours) sits at the bottom of each group
            _gradient_bar(ax, y, v, bar * 0.72, r["color"])
            ax.text(v + 0.7, y, f"{v:.1f}", fontsize=12.5, va="center", color=INK2,
                    weight="bold" if r["ours"] else "normal")

    for xv in range(0, 101, 20):
        ax.axvline(xv, color=GRID, lw=1, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-bar, len(GROUPS) * group_h - gap)
    ax.set_xticks(range(0, 101, 20))
    ax.tick_params(axis="x", length=0, labelsize=13, labelcolor=MUTED, pad=8)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b9bcc4")
    ax.set_facecolor("none")

    fig.text(0.035, 0.22 / h, "* reported in the OvozifyLabs/nllb-200-uz-en-v1 model card; "
             "all other rows were run with the same script (beam 5).", fontsize=11.5,
             color=MUTED)

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"benchmark.{ext}", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    draw(load_rows(ROOT / "results" / "benchmark"), ROOT / "docs" / "img")
    print("wrote docs/img/benchmark.png")
