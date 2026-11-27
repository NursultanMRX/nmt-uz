"""The page-one teaser of the report: its numbers come from results/, never typed by hand."""

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts import paper_teaser as pt

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _json(*parts: str) -> dict:
    return json.loads(RESULTS.joinpath(*parts).read_text())


def test_quality_panel_matches_benchmark_files():
    data = pt.load(RESULTS)
    for stem in ("ours", "nllb", "nllb-1.3b"):
        full = _json("benchmark", f"{stem}_devtest.json")["full"]["compat"]
        for d in ("en2uz", "uz2en"):
            assert data["quality"][stem][d] == pytest.approx(full[d]["bleu"])


def test_latency_panel_matches_latency_files():
    data = pt.load(RESULTS)
    for stem in ("ours", "onnx-int8", "nllb"):
        cpu = _json("latency", f"{stem}_cpu.json")["beam4"]
        assert data["latency"][stem] == pytest.approx(cpu["en2uz"]["p50_ms"])


def test_draw_writes_one_pdf(tmp_path):
    pt.draw(pt.load(RESULTS), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["teaser.pdf"]
