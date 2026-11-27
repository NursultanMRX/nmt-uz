"""Architecture diagram of the model, drawn from its real config.

    uv run python scripts/architecture.py            # writes docs/img/architecture.{png,svg}
    uv run python scripts/architecture.py --paper    # writes paper/figures/architecture.pdf
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, DARK = "#2a78d6", "#2f2f2d"
PANEL, PAPER = "#e8e7e2", "#ffffff"
ENC_FILL, ENC_EDGE = "#9ec5f4", "#184f95"
DEC_FILL, DEC_EDGE = "#f7c1a8", "#b9481c"
TEXT, SMALL, HEAD = 15, 14.5, 19

mpl.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "svg.fonttype": "path",
                     "pdf.fonttype": 42})
fig = plt.figure(figsize=(23.4, 13.7))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 234)
ax.set_ylim(0, 137)
ax.axis("off")


def panel(x0, y0, x1, y1, fc=PANEL, ec=INK, ls="-", lw=1.8, r=2.5, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, zorder=z, fc=fc, ec=ec, ls=ls,
                                lw=lw, boxstyle=f"round,pad=0,rounding_size={r}"))


def layer_stack(x0, y0, x1, y1, fc, ec):
    for d in (1.6, 0.8):  # sheets behind the layer: it repeats
        panel(x0 + d, y0 + d, x1 + d, y1 + d, fc, ec, lw=1.3, r=2, z=2)
    panel(x0, y0, x1, y1, fc, ec, lw=1.8, r=2, z=3)


def box(x, y, label, w=31, h=4.6, dark=False, size=TEXT):
    panel(x - w / 2, y - h / 2, x + w / 2, y + h / 2, DARK if dark else PAPER, INK,
          lw=1.4, r=0.8, z=5)
    ax.text(x, y, label, ha="center", va="center", fontsize=size, zorder=6,
            color=PAPER if dark else INK, linespacing=1.2)


def plus(x, y, r=1.3):
    ax.add_patch(Circle((x, y), r, fc=PAPER, ec=INK, lw=1.4, zorder=5))
    ax.text(x, y + 0.05, "+", ha="center", va="center", fontsize=16, zorder=6)


def times(x, y, r=1.3):
    ax.add_patch(Circle((x, y), r, fc=PAPER, ec=INK, lw=1.4, zorder=5))
    ax.text(x, y, "×", ha="center", va="center", fontsize=15, zorder=6)


def path(*pts, color=INK, lw=1.6):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-2],), ys[:-1] + (ys[-2],), color=color, lw=lw, zorder=4,
            solid_capstyle="butt", solid_joinstyle="miter")
    ax.annotate("", pts[-1], pts[-2], zorder=4, arrowprops={
        "arrowstyle": "-|>", "color": color, "lw": lw, "shrinkA": 0, "shrinkB": 0,
        "mutation_scale": 15})


def dotted(p0, p1):
    ax.plot(*zip(p0, p1), color=INK, lw=1.6, ls=(0, (1, 2.4)), zorder=4)


def rich(x, y, parts, size=TEXT, weight="bold", ha="left"):
    """One line of text where some words are blue, the way the numbers stand out."""
    renderer = fig.canvas.get_renderer()
    texts = []
    for s, color in parts:
        t = ax.text(x, y, s, fontsize=size, weight=weight, color=color, va="center", zorder=6)
        x = ax.transData.inverted().transform((t.get_window_extent(renderer).x1, 0))[0]
        texts.append(t)
    if ha == "right":
        shift = x - texts[0].get_position()[0]
        for t in texts:
            t.set_x(t.get_position()[0] - shift)


def label(x, y, s, size=TEXT, color=INK2, **kw):
    ax.text(x, y, s, fontsize=size, color=color, va="center", **kw)


def residual(x, y_branch, y_plus, offset=18):
    ax.add_patch(Circle((x, y_branch), 0.5, color=INK, zorder=5))
    path((x, y_branch), (x + offset, y_branch), (x + offset, y_plus), (x + 1.3, y_plus))


def layer_title(x, y, name, times_n, edge):
    ax.text(x - 18.8, y, name, fontsize=TEXT, weight="bold", va="center", zorder=6)
    ax.text(x + 18.8, y, times_n, fontsize=22, weight="bold", color=edge, ha="right",
            va="center", zorder=6)


ax.text(117, 132, "nmt-uz-76m  (76.5M)", ha="center", va="center", fontsize=38,
        weight="bold", color=BLUE)
ax.text(117, 126.8, "The encoder reads the sentence, the decoder writes the translation",
        ha="center", va="center", fontsize=HEAD, color=INK2)

# encoder
ex = 76
panel(ex - 32, 16, ex + 32, 94)
ax.text(ex - 30, 91.5, "Encoder", fontsize=HEAD + 2, weight="bold", va="center")
label(ex - 30, 88.2, "reads the whole sentence", SMALL)
ax.text(ex, 3.5, "<2uz> The library opens at nine.", ha="center", va="center",
        fontsize=TEXT, family="Consolas", color=INK)
path((ex, 5.4), (ex, 8.2))
box(ex, 10.5, "Tokenized text")
path((ex, 12.8), (ex, 18.7))
box(ex, 21, "Token embedding")
layer_stack(ex - 20.5, 26, ex + 20.5, 78.5, ENC_FILL, ENC_EDGE)
layer_title(ex, 76.2, "One encoder layer", "× 8", ENC_EDGE)
path((ex, 23.3), (ex, 28.7))
box(ex, 31, "RMSNorm")
path((ex, 33.3), (ex, 37))
box(ex, 41, "Self-attention", h=8, dark=True)
path((ex, 45), (ex, 50.2))
plus(ex, 51.5)
path((ex, 52.8), (ex, 56.2))
box(ex, 58.5, "RMSNorm")
path((ex, 60.8), (ex, 64.2))
box(ex, 66.5, "Feed forward")
path((ex, 68.8), (ex, 71.9))
plus(ex, 73.2)
residual(ex, 27.6, 51.5)
residual(ex, 54.5, 73.2)
box(ex - 26.5, 41, "RoPE", w=9)
label(ex - 26.5, 36.6, "word order", SMALL - 1, ha="center")
path((ex - 22, 41), (ex - 15.5, 41))
path((ex, 74.5), (ex, 81.7))
box(ex, 84, "Final RMSNorm")

# decoder
dx = 156
panel(dx - 32, 16, dx + 32, 122.5)
ax.text(dx - 30, 120, "Decoder", fontsize=HEAD + 2, weight="bold", va="center")
label(dx - 30, 116.6, "writes one token at a time", SMALL)
label(dx, 3.5, "starts with <s>", TEXT, ha="center")
path((dx, 5.4), (dx, 8.2))
box(dx, 10.5, "Tokens written so far")
path((dx, 12.8), (dx, 18.7))
box(dx, 21, "Token embedding")
layer_stack(dx - 20.5, 26, dx + 20.5, 99.5, DEC_FILL, DEC_EDGE)
layer_title(dx, 97.2, "One decoder layer", "× 4", DEC_EDGE)
path((dx, 23.3), (dx, 28.7))
box(dx, 31, "RMSNorm")
path((dx, 33.3), (dx, 37))
box(dx, 41, "Masked\nself-attention", h=8, dark=True)
path((dx, 45), (dx, 50.2))
plus(dx, 51.5)
path((dx, 52.8), (dx, 56.2))
box(dx, 58.5, "RMSNorm")
path((dx, 60.8), (dx, 63.3))
box(dx, 67, "Cross-attention", h=7, dark=True)
path((dx, 70.5), (dx, 73.2))
plus(dx, 74.5)
path((dx, 75.8), (dx, 78.7))
box(dx, 81, "RMSNorm")
path((dx, 83.3), (dx, 85.7))
box(dx, 88, "Feed forward")
path((dx, 90.3), (dx, 92.7))
plus(dx, 94)
residual(dx, 27.6, 51.5)
residual(dx, 54.5, 74.5)
residual(dx, 77.2, 94)
box(dx - 26.5, 41, "RoPE", w=9)
label(dx - 26.5, 36.6, "word order", SMALL - 1, ha="center")
path((dx - 22, 41), (dx - 15.5, 41))
path((dx, 95.3), (dx, 101.7))
box(dx, 104, "Final RMSNorm")
path((dx, 106.3), (dx, 108.7))
box(dx, 111, "Linear output layer")
path((dx, 113.3), (dx, 124))

# the encoder output feeds the cross-attention of every decoder layer
path((ex, 86.3), (ex, 98.5), (116, 98.5), (116, 67), (dx - 15.5, 67), color=BLUE, lw=2.8)
ax.text((ex + 116) / 2, 101, "Encoder output → every decoder layer", ha="center",
        va="center", fontsize=TEXT, weight="bold", color=BLUE)
dotted((ex + 15.5, 21), (dx - 15.5, 21))
label(116, 23.2, "same weights", SMALL, ha="center",
      bbox={"fc": PAPER, "ec": "none", "pad": 1.5})

# left column
ax.text(2, 112, "One model, both directions", fontsize=HEAD, weight="bold", color=BLUE,
        va="center")
rich(2, 106.5, [("<2uz>", BLUE), ("  →  English to Uzbek", INK)], weight="normal")
rich(2, 102, [("<2en>", BLUE), ("  →  Uzbek to English", INK)], weight="normal")

ax.text(2, 88, "Parameters", fontsize=HEAD, weight="bold", color=BLUE, va="center")
for i, (name, value) in enumerate([("Shared embedding", "10.5M"),
                                   ("Encoder, 8 layers", "39.7M"),
                                   ("Decoder, 4 layers", "26.4M")]):
    label(2, 83 - 4.3 * i, name)
    label(38, 83 - 4.3 * i, value, ha="right")
ax.plot([2, 38], [72.6, 72.6], color=INK2, lw=0.9)
label(2, 69.8, "Total", TEXT + 1, INK, weight="bold")
label(38, 69.8, "76.5M", TEXT + 1, BLUE, weight="bold", ha="right")

rich(40, 36.5, [("Up to ", INK), ("256", BLUE), (" tokens", INK)], ha="right")
dotted((ex - 31, 38.6), (40.6, 36.5))
rich(40, 13, [("Vocabulary", INK)], ha="right")
rich(40, 9.3, [("16,384", BLUE), (" tokens", INK)], ha="right")
dotted((ex - 15.5, 10.5), (40.6, 11.2))

# right column
rich(194, 113, [("Next token", INK)], size=TEXT + 1)
ax.text(194, 106.8, "best of 4 beams,\nfed back in below", fontsize=TEXT, color=INK2,
        va="center", linespacing=1.4)
dotted((dx + 15.5, 111), (192.5, 111))

panel(192, 56, 233, 99, ls=(0, (1, 2.4)))
ax.text(194.5, 95.5, "Feed forward (SwiGLU)", fontsize=TEXT + 1, weight="bold", va="center")
box(203, 62.5, "Linear", w=14, h=4.4)
box(203, 72, "SiLU", w=14, h=4.4)
box(224, 72, "Linear", w=14, h=4.4)
times(213.5, 80.5)
box(213.5, 88.5, "Linear", w=14, h=4.4)
path((203, 52.5), (203, 60.3))
path((203, 57.5), (224, 57.5), (224, 69.8))
path((203, 64.7), (203, 69.8))
path((203, 74.2), (203, 80.5), (212.2, 80.5))
path((224, 74.2), (224, 80.5), (214.8, 80.5))
path((213.5, 81.8), (213.5, 86.3))
path((213.5, 90.7), (213.5, 93))
dotted((dx + 15.5, 88), (192, 88))
rich(192, 50, [("Width ", INK), ("640", BLUE), ("   Hidden ", INK), ("1,728", BLUE)])

rich(192, 25, [("Shared embedding", INK)])
rich(192, 21, [("16,384 × 640", BLUE)])
dotted((dx + 15.5, 21), (191, 23))

if "--paper" in sys.argv:
    OUT = OUT.parents[1] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf",) if "--paper" in sys.argv else ("png", "svg"):
    fig.savefig(OUT / f"architecture.{ext}", dpi=130, facecolor=PAPER, bbox_inches="tight",
                pad_inches=0.3)
print("wrote", OUT / "architecture.*")
