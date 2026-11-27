"""Page-one teaser of the technical report: quality, CPU latency and size at a glance.

    uv run python scripts/paper_teaser.py        # writes paper/figures/teaser.pdf
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# results stem: label, colour, parameters
SYSTEMS = {
    "ours": ("nmt-uz-76m (ours)", "#2a78d6", 76.5e6),
    "nllb": ("NLLB-200-distilled-600M", "#eb6834", 600e6),
    "nllb-1.3b": ("NLLB-200-distilled-1.3B", "#a8a79f", 1.3e9),
}
LATENCY = {"ours": ("nmt-uz-76m, PyTorch", "#2a78d6"),
           "onnx-int8": ("nmt-uz-76m, ONNX int8", "#86b6ef"),
           "nllb": ("NLLB-200-distilled-600M", "#eb6834")}
DIRS = [("en2uz", "English→Uzbek"), ("uz2en", "Uzbek→English")]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load(results: Path) -> dict:
    full = {s: _json(results / "benchmark" / f"{s}_devtest.json")["full"]["compat"]
            for s in SYSTEMS}
    return {
        "quality": {s: {d: full[s][d]["bleu"] for d, _ in DIRS} for s in SYSTEMS},
        "chrf": {s: {d: full[s][d]["chrf"] for d, _ in DIRS} for s in SYSTEMS},
        "latency": {s: _json(results / "latency" / f"{s}_cpu.json")["beam4"]["en2uz"]["p50_ms"]
                    for s in LATENCY},
    }


def _axes(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.tick_params(length=0, labelsize=9, labelcolor=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")


def draw(data: dict, out: Path) -> None:
    mpl.rcParams.update({"font.family": ["DejaVu Sans"], "pdf.fonttype": 42})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.3),
                                  gridspec_kw={"width_ratios": [1.15, 1.1, 1]})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.2, wspace=0.42)

    # (a) BLEU, both directions: the weak direction is shown, not hidden
    width = 0.26
    for i, (stem, (label, color, _)) in enumerate(SYSTEMS.items()):
        xs = [k + (i - 1) * width for k in range(len(DIRS))]
        ys = [data["quality"][stem][d] for d, _ in DIRS]
        a.bar(xs, ys, width, color=color, edgecolor="white", linewidth=1.5, label=label)
        for x, y in zip(xs, ys):
            a.text(x, y + 0.6, f"{y:.1f}", ha="center", va="bottom", fontsize=8,
                   color=INK, weight="bold" if stem == "ours" else "normal")
    a.set_xticks(range(len(DIRS)), [name for _, name in DIRS])
    a.set_ylim(0, 40)
    a.set_ylabel("BLEU", fontsize=9, color=MUTED)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    _axes(a, "(a) FLORES+ devtest BLEU")

    # (b) CPU latency of one sentence
    stems = list(LATENCY)[::-1]
    secs = [data["latency"][s] / 1000 for s in stems]
    b.barh(range(len(stems)), secs, height=0.42, color=[LATENCY[s][1] for s in stems])
    for y, (stem, v) in enumerate(zip(stems, secs)):
        b.text(0, y + 0.3, LATENCY[stem][0], va="bottom", fontsize=8.5, color=INK2)
        b.text(v + 0.12, y, f"{v:.2f} s", va="center", fontsize=9, color=INK,
               weight="bold" if stem != "nllb" else "normal")
    b.set_yticks([])
    b.set_ylim(-0.5, len(stems) - 0.1)
    b.set_xlim(0, max(secs) * 1.22)
    b.set_xlabel("median seconds per sentence, 4-core CPU", fontsize=9, color=MUTED)
    b.grid(axis="x", color=GRID, lw=0.8)
    b.set_axisbelow(True)
    _axes(b, "(b) CPU latency, English→Uzbek, beam 4")

    # (c) chrF against model size
    for stem, (label, color, params) in SYSTEMS.items():
        for (d, _), marker in zip(DIRS, ("o", "s")):
            c.scatter(params, data["chrf"][stem][d], s=60, marker=marker, color=color,
                      edgecolor="white", linewidth=1.2, zorder=3)
    c.set_xscale("log")
    c.set_xlim(4e7, 2.5e9)
    c.set_xticks([1e8, 3e8, 1e9], ["100M", "300M", "1B"])
    c.minorticks_off()
    c.set_ylim(50, 64)
    c.set_xlabel("parameters (log scale)", fontsize=9, color=MUTED)
    c.set_ylabel("chrF", fontsize=9, color=MUTED)
    c.grid(color=GRID, lw=0.8)
    c.set_axisbelow(True)
    c.legend(handles=[Line2D([], [], ls="", marker=m, color=INK2, label=n)
                      for m, (_, n) in zip(("o", "s"), DIRS)],
             loc="upper left", frameon=False, fontsize=8.5)
    _axes(c, "(c) Quality against size")

    a.legend(loc="upper center", bbox_to_anchor=(1.75, -0.13), ncol=3, frameon=False,
             fontsize=9, handlelength=1.2)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "teaser.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    draw(load(ROOT / "results"), ROOT / "paper" / "figures")
    print("wrote paper/figures/teaser.pdf")
